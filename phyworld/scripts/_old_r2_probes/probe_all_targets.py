"""Unified probe-only step: take a cached embedding .npy + dataset, run all
probe targets at K=1 and K=4. Used to retroactively complete probes that were
written with only (pos_x, vel_x, collision) targets.

Targets:
  · pos_x       Ridge R²    proprio[:, [0,2]]              both balls' x position
  · vel_x       Ridge R²    state[:, [0,2]]                both balls' x velocity
  · speed       Ridge R²    sqrt(vx1² + vx2²)              alt velocity formulation (norm)
  · mass        Ridge R²    mass[:, [0,1]]                 individual masses (m1, m2)
  · mass_ratio  Ridge R²    mass[:, 0] / mass[:, 1]        m1/m2 — historical target
  · accel_x     Ridge R²    forward-diff of vel_x, per ep  within-episode acceleration
  · collision   LogReg AUC  collision_event 0/1            collision indicator
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score


def split_episodes(n_episodes, train_frac=0.8, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    n_train = int(round(n_episodes * train_frac))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def build_multiframe_feats(emb_per_frame, ep_idx, step_idx, K):
    N, D = emb_per_frame.shape
    feats = np.zeros((N, K * D), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)
    by_ep = {}
    for i in range(N):
        by_ep.setdefault(int(ep_idx[i]), []).append((int(step_idx[i]), i))
    for ep, lst in by_ep.items():
        lst.sort()
        ordered = [idx for _, idx in lst]
        for k, idx in enumerate(ordered):
            if k >= K - 1:
                ctx = ordered[k - K + 1:k + 1]
                feats[idx] = np.concatenate([emb_per_frame[j] for j in ctx])
                valid[idx] = True
    return feats, valid


def compute_acceleration(vel_x, ep_idx, step_idx):
    """Forward-diff a[t] = v[t+1] - v[t] within each episode. Last frame of
    each ep gets the previous step's diff (so the array stays dense)."""
    N = vel_x.shape[0]
    accel = np.zeros_like(vel_x)
    valid = np.ones(N, dtype=bool)
    by_ep = {}
    for i in range(N):
        by_ep.setdefault(int(ep_idx[i]), []).append((int(step_idx[i]), i))
    for ep, lst in by_ep.items():
        lst.sort()
        ordered = [idx for _, idx in lst]
        L = len(ordered)
        for k in range(L - 1):
            accel[ordered[k]] = vel_x[ordered[k + 1]] - vel_x[ordered[k]]
        # Last frame: copy previous step's accel (avoids leakage to test set
        # since the episode boundary is preserved)
        accel[ordered[-1]] = accel[ordered[-2]] if L >= 2 else 0.0
    return accel, valid


def fit_eval_targets(emb, targets_reg, targets_cls, mask_tr, mask_te, *, label):
    """Run Ridge / LogReg over a dict of regression/classification targets."""
    mu, sigma = emb[mask_tr].mean(0), emb[mask_tr].std(0) + 1e-6
    e_tr = (emb[mask_tr] - mu) / sigma
    e_te = (emb[mask_te] - mu) / sigma

    results = {}
    print(f"  [{label}] train={mask_tr.sum()}, test={mask_te.sum()}, feat_dim={emb.shape[1]}")
    for name, y in targets_reg.items():
        m = Ridge(alpha=1.0); m.fit(e_tr, y[mask_tr])
        r2 = r2_score(y[mask_te], m.predict(e_te))
        results[name] = r2
        print(f"    {name:20s} R²    {r2:+.4f}")
    for name, y in targets_cls.items():
        y_tr_v = y[mask_tr]
        if y_tr_v.sum() > 1 and y_tr_v.sum() < len(y_tr_v):
            clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
            clf.fit(e_tr, y_tr_v)
            p = clf.predict_proba(e_te)[:, 1]
            auc = roc_auc_score(y[mask_te], p)
            results[name] = auc
            print(f"    {name:20s} AUC   {auc:.4f}")
        else:
            print(f"    {name:20s} skipped (no positive in train)")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--emb", required=True, help=".npy embedding cache (N, D)")
    ap.add_argument("--data", default=os.path.expanduser("~/.stable_worldmodel/phyworld_collision.h5"))
    ap.add_argument("--n-frames", type=int, default=None, help="Cap frames; default = embedding length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, nargs="+", default=[1, 4])
    args = ap.parse_args()

    emb = np.load(args.emb)
    print(f"[load] emb {args.emb}  shape={emb.shape}")

    N = args.n_frames if args.n_frames else emb.shape[0]
    assert N <= emb.shape[0], f"requested {N} > emb {emb.shape[0]}"

    with h5py.File(args.data, "r") as f:
        prop = f["proprio"][:N]
        state = f["state"][:N]
        mass = f["mass"][:N]
        coll = f["collision_event"][:N].astype(np.int64)
        ep_idx = f["episode_idx"][:N]
        step_idx = f["step_idx"][:N]
    n_ep = len(np.unique(ep_idx))
    emb = emb[:N]

    pos_x = prop[:, [0, 2]].astype(np.float32)
    vel_x = state[:, [0, 2]].astype(np.float32)
    speed = np.linalg.norm(vel_x, axis=1, keepdims=True).astype(np.float32)
    mass_2d = mass[:, [0, 1]].astype(np.float32)
    mass_ratio = (mass[:, 0] / (mass[:, 1] + 1e-6)).astype(np.float32).reshape(-1, 1)
    accel_x, _ = compute_acceleration(vel_x, ep_idx, step_idx)

    train_eps, test_eps = split_episodes(n_ep, 0.8, args.seed)

    for K in args.K:
        print(f"\n=== K={K} ({'single-frame' if K == 1 else 'multi-frame'}) ===")
        if K == 1:
            feats = emb
            valid_K = np.ones(N, dtype=bool)
        else:
            feats, valid_K = build_multiframe_feats(emb, ep_idx, step_idx, K=K)
        base_tr = np.isin(ep_idx, train_eps) & valid_K
        base_te = np.isin(ep_idx, test_eps) & valid_K
        fit_eval_targets(
            feats,
            targets_reg={
                "pos_x": pos_x,
                "vel_x": vel_x,
                "speed": speed,
                "mass": mass_2d,
                "mass_ratio": mass_ratio,
                "accel_x": accel_x,
            },
            targets_cls={"collision_event": coll},
            mask_tr=base_tr, mask_te=base_te,
            label=f"K={K}",
        )


if __name__ == "__main__":
    main()
