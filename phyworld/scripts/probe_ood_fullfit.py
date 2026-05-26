"""ID+OOD-mixed probe on phyworld collision_eval, per-partition evaluation.

Protocol (different from previous OOD experiments):
  · Take collision_eval.hdf5 (1635 trajs, mixed ID + r-OOD + v-OOD + both-OOD)
  · 80/20 split BY TRAJ (mixing all partitions in both train and test)
  · Fit Ridge / class-balanced LogReg on ALL train data (probe sees ID + OOD)
  · Evaluate on the 20% test set, broken down per partition

This isolates ENCODER REPRESENTATION QUALITY per partition,
removing the "probe-extrapolation" handicap from the previous OOD experiments.

Compares two frozen encoders:
  · lewm-pusht (5.5M, frozen, official PushT weights)        [no_projector emb]
  · DiT-XL zero-shot (749M, frozen, ImageNet pretrained)     [last-block pooled]

Both never saw phyworld during training; this is a head-to-head zero-shot
representation comparison on phyworld collision ID + OOD.
"""
from __future__ import annotations

import os
import numpy as np
import h5py
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score


PART_NAMES = {0: "ID", 1: "r-OOD", 2: "v-OOD", 3: "both-OOD"}


def build_multiframe_feats(emb_per_frame, ep_idx, step_idx, K):
    """Concat K consecutive frames' embeddings (within each episode)."""
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


def fit_and_eval(emb, label_name, prop, state, coll, ep_idx, part, *,
                 K=1, step_idx=None, train_frac=0.8, seed=0):
    """K=1: use emb directly. K>=2: build multi-frame concat features."""
    if K > 1:
        assert step_idx is not None
        feats, valid = build_multiframe_feats(emb, ep_idx, step_idx, K)
    else:
        feats = emb
        valid = np.ones(len(emb), dtype=bool)

    n_ep = len(np.unique(ep_idx))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.unique(ep_idx))
    n_tr = int(round(n_ep * train_frac))
    train_eps = set(perm[:n_tr].tolist())
    base_tr = np.array([e in train_eps for e in ep_idx])
    base_te = ~base_tr
    mask_tr = base_tr & valid
    mask_te = base_te & valid

    mu, sigma = feats[mask_tr].mean(0), feats[mask_tr].std(0) + 1e-6
    e_tr = (feats[mask_tr] - mu) / sigma

    pos_x = prop[:, [0, 2]]
    vel_x = state[:, [0, 2]]

    print(f"\n=== {label_name}  [K={K}] ===  feat dim={feats.shape[1]}")
    print(f"  train: {mask_tr.sum()} frames (mixed partitions; valid mask drops K-1 first frames per traj)")
    print(f"  test : {mask_te.sum()} frames")

    ridge_pos = Ridge(alpha=1.0); ridge_pos.fit(e_tr, pos_x[mask_tr])
    ridge_vel = Ridge(alpha=1.0); ridge_vel.fit(e_tr, vel_x[mask_tr])
    if coll[mask_tr].sum() > 1 and coll[mask_tr].sum() < mask_tr.sum():
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
        clf.fit(e_tr, coll[mask_tr])
    else:
        clf = None

    print(f"  {'partition':12s} {'n_frames':>9s}  pos_x R²  vel_x R²  collision AUC")
    for p in sorted(np.unique(part)):
        pm = mask_te & (part == p)
        if pm.sum() < 50:
            continue
        n = pm.sum()
        e_p = (feats[pm] - mu) / sigma
        r2_pos = r2_score(pos_x[pm], ridge_pos.predict(e_p))
        r2_vel = r2_score(vel_x[pm], ridge_vel.predict(e_p))
        if clf is not None and coll[pm].sum() > 1 and coll[pm].std() > 0:
            p_pos = clf.predict_proba(e_p)[:, 1]
            auc = roc_auc_score(coll[pm], p_pos)
            auc_str = f"{auc:.4f}"
        else:
            auc_str = "N/A"
        print(f"  {PART_NAMES[p]:12s} {n:>9d}   {r2_pos:+.4f}   {r2_vel:+.4f}    {auc_str}")


def main():
    eval_path = "/home/qlib/.stable_worldmodel/phyworld_collision_eval.h5"
    emb_lewm = "/home/qlib/agent_memory/wm/artifacts/embeddings/lewm_pusht_only_collision_eval_emb_52k_noproj.npy"
    emb_dit = "/home/qlib/agent_memory/wm/artifacts/embeddings/dit_xl_zeroshot_collision_eval_emb_52k.npy"

    print(f"[load] eval = {eval_path}")
    with h5py.File(eval_path, "r") as f:
        prop = f["proprio"][:]
        state = f["state"][:]
        coll = f["collision_event"][:]
        ep_idx = f["episode_idx"][:]
        step_idx = f["step_idx"][:]
        part = f["partition"][:]
    print(f"  total frames: {len(prop)}")
    from collections import Counter
    print(f"  partitions: {dict(Counter(part.tolist()))}")

    print(f"\n[load] LeWM pusht-only emb (192-D, frozen) ...")
    e_lewm = np.load(emb_lewm)
    print(f"  shape: {e_lewm.shape}")

    print(f"\n[load] DiT-XL zero-shot emb (1152-D, frozen) ...")
    e_dit = np.load(emb_dit)
    print(f"  shape: {e_dit.shape}")

    for K in (1, 4):
        print(f"\n{'=' * 60}\n  K={K} probe (mixed-fit Ridge on 80% ID+OOD)\n{'=' * 60}")
        fit_and_eval(e_lewm, "LeWM pusht-only (frozen, 5.5M)",
                     prop, state, coll, ep_idx, part, K=K, step_idx=step_idx)
        fit_and_eval(e_dit, "DiT-XL zero-shot (frozen, 749.8M)",
                     prop, state, coll, ep_idx, part, K=K, step_idx=step_idx)

    print("\nDone.")


if __name__ == "__main__":
    main()
