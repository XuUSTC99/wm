"""Fix for the OOD probe extrapolation flaw.

The original `probe_ood.py` fits a Ridge on ID-partition training frames, then
applies that same Ridge to OOD partitions. If OOD target values fall outside
the ID training range, the linear probe **cannot extrapolate** even when the
encoder has the right features — so R² drop is confounded between "encoder
fails on OOD" and "probe trained on ID can't reach OOD region of feature space".

This script runs TWO protocols on the same cached embedding:
  A) Original protocol — Ridge fit on ID-train fold, tested on each partition's test fold
  B) Per-partition protocol — Ridge fit on EACH partition's own train fold (80/20 by episode),
     tested on its own test fold. No extrapolation involved.

Diagnostic:
  A low + B low   → encoder genuinely fails on OOD (conclusion preserved)
  A low + B high  → probe couldn't extrapolate; encoder is fine (conclusion overturned)
  A high + B low  → rare (would mean encoder happened to align with ID Ridge weights even though
                    its OOD features are scrambled; almost never seen in practice)

Run on cached embedding files produced by encode_*_eval.py companion scripts.

Targets:
  pos_x       Ridge R²    proprio[:, [0,2]]
  vel_x       Ridge R²    state[:, [0,2]]
  mass        Ridge R²    mass[:, [0,1]]
  mass_ratio  Ridge R²    mass[:,0]/mass[:,1]
  speed       Ridge R²    sqrt(vx1² + vx2²)
  collision   LogReg AUC  collision_event 0/1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score


PART_NAMES = {0: "ID", 1: "r-OOD", 2: "v-OOD", 3: "both-OOD"}


def split_episodes_in_set(ep_ids: np.ndarray, train_frac=0.8, seed=0):
    """Given a set of episode IDs, return (train_eps, test_eps) sorted."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ep_ids)
    n_train = int(round(len(ep_ids) * train_frac))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def fit_eval_set(
    emb_train: np.ndarray,
    emb_test: np.ndarray,
    targets_reg: dict,
    targets_cls: dict,
    mask_train_global: np.ndarray,
    mask_test_global: np.ndarray,
    label: str,
):
    """Standardize emb on train, fit, evaluate. Returns dict of {target: score}."""
    if len(emb_train) < 20 or len(emb_test) < 20:
        return {name: np.nan for name in list(targets_reg) + list(targets_cls)}
    mu, sigma = emb_train.mean(0), emb_train.std(0) + 1e-6
    e_tr = (emb_train - mu) / sigma
    e_te = (emb_test - mu) / sigma

    results = {}
    for name, y in targets_reg.items():
        y_tr = y[mask_train_global]
        y_te = y[mask_test_global]
        if len(y_tr) < 20 or len(y_te) < 20:
            results[name] = np.nan
            continue
        m = Ridge(alpha=1.0)
        m.fit(e_tr, y_tr)
        r2 = r2_score(y_te, m.predict(e_te))
        results[name] = float(r2)

    for name, y in targets_cls.items():
        y_tr = y[mask_train_global]
        y_te = y[mask_test_global]
        if y_tr.sum() < 2 or y_tr.sum() >= len(y_tr) - 1 or len(y_te) < 20:
            results[name] = np.nan
            continue
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
        clf.fit(e_tr, y_tr)
        p = clf.predict_proba(e_te)[:, 1]
        results[name] = float(roc_auc_score(y_te, p))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="cached .npy embedding of eval frames (52320, D)")
    ap.add_argument("--eval-data", default=os.path.expanduser("~/.stable_worldmodel/phyworld_collision_eval.h5"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional JSON to dump full results")
    args = ap.parse_args()

    emb = np.load(args.emb)
    print(f"[load] emb {args.emb}  shape={emb.shape}")

    with h5py.File(args.eval_data, "r") as f:
        prop = f["proprio"][:]
        state = f["state"][:]
        mass = f["mass"][:]
        coll = f["collision_event"][:].astype(np.int64)
        ep_idx = f["episode_idx"][:]
        part = f["partition"][:]
    N = emb.shape[0]
    assert N == len(prop), f"emb {N} != frames {len(prop)}"

    pos_x = prop[:, [0, 2]].astype(np.float32)
    vel_x = state[:, [0, 2]].astype(np.float32)
    speed = np.linalg.norm(vel_x, axis=1, keepdims=True).astype(np.float32)
    mass_2d = mass[:, [0, 1]].astype(np.float32)
    mass_ratio = (mass[:, 0] / (mass[:, 1] + 1e-6)).astype(np.float32).reshape(-1, 1)

    targets_reg = {
        "pos_x": pos_x,
        "vel_x": vel_x,
        "speed": speed,
        "mass": mass_2d,
        "mass_ratio": mass_ratio,
    }
    targets_cls = {"collision_event": coll}

    # Build per-partition episode lists
    part_eps = {}
    for p_val in [0, 1, 2, 3]:
        mask = part == p_val
        eps = np.unique(ep_idx[mask])
        part_eps[p_val] = eps
        print(f"  partition {p_val} ({PART_NAMES[p_val]}): {len(eps)} eps / {mask.sum()} frames")

    # Per-partition 80/20 episode split
    part_train_eps, part_test_eps = {}, {}
    for p_val, eps in part_eps.items():
        tr, te = split_episodes_in_set(eps, train_frac=0.8, seed=args.seed)
        part_train_eps[p_val] = tr
        part_test_eps[p_val] = te

    all_results = {}

    print("\n========== PROTOCOL A: Ridge fit on ID-train, tested per-partition (original, confounded) ==========")
    id_train_eps = part_train_eps[0]
    mask_id_train = np.isin(ep_idx, id_train_eps)
    for p_val in [0, 1, 2, 3]:
        test_eps = part_test_eps[p_val]
        mask_test = np.isin(ep_idx, test_eps)
        res = fit_eval_set(
            emb[mask_id_train], emb[mask_test], targets_reg, targets_cls,
            mask_id_train, mask_test, f"A: ID-train → {PART_NAMES[p_val]}-test",
        )
        n_tr = mask_id_train.sum(); n_te = mask_test.sum()
        print(f"  A.{PART_NAMES[p_val]:9s} (train_n={n_tr}, test_n={n_te}):  " +
              "  ".join(f"{k}={v:+.3f}" for k, v in res.items()))
        all_results[f"A_{PART_NAMES[p_val]}"] = res

    print("\n========== PROTOCOL B: Ridge fit per-partition (each partition's own train fold) ==========")
    for p_val in [0, 1, 2, 3]:
        tr_eps = part_train_eps[p_val]
        te_eps = part_test_eps[p_val]
        mask_tr = np.isin(ep_idx, tr_eps)
        mask_te = np.isin(ep_idx, te_eps)
        res = fit_eval_set(
            emb[mask_tr], emb[mask_te], targets_reg, targets_cls,
            mask_tr, mask_te, f"B: {PART_NAMES[p_val]}-train → {PART_NAMES[p_val]}-test",
        )
        n_tr = mask_tr.sum(); n_te = mask_te.sum()
        print(f"  B.{PART_NAMES[p_val]:9s} (train_n={n_tr}, test_n={n_te}):  " +
              "  ".join(f"{k}={v:+.3f}" for k, v in res.items()))
        all_results[f"B_{PART_NAMES[p_val]}"] = res

    print("\n========== PROTOCOL C: single Ridge fit on COMBINED train, tested per-partition ==========")
    print("  (no probe extrapolation since probe sees ID + all OOD value ranges; same probe across partitions)")
    # Combined training mask = union of train folds across all partitions
    all_train_eps = np.concatenate([part_train_eps[p] for p in [0, 1, 2, 3]])
    mask_combined_train = np.isin(ep_idx, all_train_eps)
    print(f"  combined train_n = {mask_combined_train.sum()}", flush=True)
    # Fit one Ridge (and one LogReg per cls target) on combined train, evaluate per-partition test
    # We use the same standardization (mu/sigma on combined train) for fair scaling across partitions.
    mu_c, sigma_c = emb[mask_combined_train].mean(0), emb[mask_combined_train].std(0) + 1e-6
    e_combined_train = (emb[mask_combined_train] - mu_c) / sigma_c
    fitted_reg = {}
    for name, y in targets_reg.items():
        m = Ridge(alpha=1.0); m.fit(e_combined_train, y[mask_combined_train])
        fitted_reg[name] = m
    fitted_cls = {}
    for name, y in targets_cls.items():
        y_tr = y[mask_combined_train]
        if y_tr.sum() > 1 and y_tr.sum() < len(y_tr) - 1:
            clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
            clf.fit(e_combined_train, y_tr)
            fitted_cls[name] = clf
    # Evaluate per partition's test fold using the same fitted probes
    for p_val in [0, 1, 2, 3]:
        te_eps = part_test_eps[p_val]
        mask_te = np.isin(ep_idx, te_eps)
        e_te = (emb[mask_te] - mu_c) / sigma_c
        res = {}
        for name, m in fitted_reg.items():
            y_te = targets_reg[name][mask_te]
            r2 = r2_score(y_te, m.predict(e_te)) if len(y_te) >= 20 else np.nan
            res[name] = float(r2)
        for name, clf in fitted_cls.items():
            y_te = targets_cls[name][mask_te]
            if len(y_te) >= 20 and y_te.sum() > 0:
                p = clf.predict_proba(e_te)[:, 1]
                res[name] = float(roc_auc_score(y_te, p))
            else:
                res[name] = np.nan
        print(f"  C.{PART_NAMES[p_val]:9s} (test_n={mask_te.sum()}):  " +
              "  ".join(f"{k}={v:+.3f}" for k, v in res.items()))
        all_results[f"C_{PART_NAMES[p_val]}"] = res

    print("\n========== DIAGNOSTIC: protocol comparison per partition ==========")
    print("  Δ B−A : large = probe extrapolation was the issue (Protocol A's penalty was probe, not encoder)")
    print("  Δ C−B : non-zero = sample-size effect (B over-fits each partition independently)")
    print("  C 上 ID vs OOD : same probe, so direct test of encoder consistency across partitions")
    for p_val in [1, 2, 3]:
        name = PART_NAMES[p_val]
        a = all_results[f"A_{name}"]
        b = all_results[f"B_{name}"]
        c = all_results[f"C_{name}"]
        diff_ba = {k: (b[k] - a[k]) if (not np.isnan(a[k]) and not np.isnan(b[k])) else np.nan
                   for k in a}
        diff_cb = {k: (c[k] - b[k]) if (not np.isnan(c[k]) and not np.isnan(b[k])) else np.nan
                   for k in b}
        print(f"  Δ B−A on {name:9s}:  " + "  ".join(f"{k}={v:+.3f}" for k, v in diff_ba.items()))
        print(f"  Δ C−B on {name:9s}:  " + "  ".join(f"{k}={v:+.3f}" for k, v in diff_cb.items()))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        # convert NaN to None for JSON serialization
        def _conv(x):
            if isinstance(x, dict):
                return {k: _conv(v) for k, v in x.items()}
            if isinstance(x, float) and np.isnan(x):
                return None
            return x
        with open(args.out, "w") as f:
            json.dump(_conv(all_results), f, indent=2)
        print(f"\n[json] saved → {args.out}")


if __name__ == "__main__":
    main()
