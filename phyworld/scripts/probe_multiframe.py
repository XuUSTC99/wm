"""Multi-frame probe: concatenate K consecutive frames' embeddings before Ridge.

If single-frame probe shows vel R² ≈ 0.47 (collision/uniform_motion observation)
and multi-frame K=4 jumps to e.g. 0.85, then velocity info IS in the embeddings,
just spread across frames — confirming our "single-frame protocol" hypothesis.

Usage
-----
    python scripts/probe_multiframe.py \
        --ckpt ~/.stable_worldmodel/collision_run/lewm_collision_epoch_8_object.ckpt \
        --data ~/.stable_worldmodel/phyworld_collision.h5 \
        --K 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

LEWM_ROOT = Path(__file__).resolve().parent.parent.parent / "le-wm"
sys.path.insert(0, str(LEWM_ROOT))

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def extract_embeddings(model, pix_uint8, *, batch_size=128, device="cuda",
                       use_projector=True, img_size=224):
    if hasattr(model, "embeddings"):
        encoder, projector = model, None
    elif hasattr(model, "encoder"):
        encoder = model.encoder
        projector = getattr(model, "projector", None) if use_projector else None
    else:
        encoder, projector = model, None
    encoder.eval().to(device)
    if projector is not None:
        projector.eval().to(device)
    mean = _MEAN.to(device)
    std = _STD.to(device)
    N = pix_uint8.shape[0]
    out = []
    for i in range(0, N, batch_size):
        x = pix_uint8[i:i + batch_size].to(device, non_blocking=True).float() / 255.0
        x = (x - mean) / std
        if x.shape[-1] != img_size:
            x = torch.nn.functional.interpolate(x, size=img_size, mode="bilinear",
                                                align_corners=False)
        h = encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        if projector is not None:
            h = projector(h)
        out.append(h.float().cpu().numpy())
        if (i // batch_size) % 100 == 0:
            print(f"  encoded {i + x.shape[0]}/{N}", flush=True)
    return np.concatenate(out, axis=0)


def build_multiframe_feats(emb_per_frame, episode_idx, step_idx, K):
    """For each frame t with enough history (step >= K-1), feature = concat(emb[t-K+1..t])."""
    N, D = emb_per_frame.shape
    feats = np.zeros((N, K * D), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)
    by_ep = {}
    for i, ep in enumerate(episode_idx):
        by_ep.setdefault(int(ep), []).append((int(step_idx[i]), i))
    for ep, lst in by_ep.items():
        lst.sort()
        ordered = [idx for _, idx in lst]
        for k, idx in enumerate(ordered):
            if k >= K - 1:
                ctx = ordered[k - K + 1:k + 1]
                feats[idx] = np.concatenate([emb_per_frame[j] for j in ctx])
                valid[idx] = True
    return feats, valid


def split_episodes(n_episodes, train_frac=0.8, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    n_train = int(round(n_episodes * train_frac))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def fit_regression(emb_tr, y_tr, emb_te, y_te, alpha=1.0):
    m = Ridge(alpha=alpha)
    m.fit(emb_tr, y_tr)
    yp = m.predict(emb_te)
    r2 = r2_score(y_te, yp, multioutput="uniform_average")
    r2_per = r2_score(y_te, yp, multioutput="raw_values") if y_te.ndim > 1 and y_te.shape[1] > 1 else None
    return r2, r2_per


def evaluate(label, feats, valid, mask_tr, mask_te, targets):
    print(f"\n=== {label} ===  feat dim={feats.shape[1]}, valid frames={valid.sum()}", flush=True)
    tr = mask_tr & valid
    te = mask_te & valid
    mu, sigma = feats[tr].mean(0), feats[tr].std(0) + 1e-6
    f_tr = (feats[tr] - mu) / sigma
    f_te = (feats[te] - mu) / sigma
    for name, y in targets.items():
        y_tr, y_te = y[tr], y[te]
        r2, r2p = fit_regression(f_tr, y_tr, f_te, y_te)
        line = f"  {name:14s}  R^2={r2:+.4f}"
        if r2p is not None:
            line += f"  per-dim={np.round(r2p, 3).tolist()}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.expanduser("~/.stable_worldmodel/phyworld_collision.h5"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--K", type=int, default=4, help="number of frames to concatenate")
    ap.add_argument("--no-projector", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"loading data {args.data} ...", flush=True)
    with h5py.File(args.data, "r") as f:
        pix = f["pixels"][:]
        proprio = f["proprio"][:]
        state = f["state"][:] if "state" in f else None
        action = f["action"][:] if "action" in f else None
        episode_idx = f["episode_idx"][:]
        step_idx = f["step_idx"][:]
        ep_len = f["ep_len"][:]
    pix_uint8 = torch.from_numpy(pix).permute(0, 3, 1, 2).contiguous()
    n_ep = len(ep_len)
    print(f"  {n_ep} episodes, {pix.shape[0]} frames, proprio={proprio.shape}, state={None if state is None else state.shape}", flush=True)

    train_eps, test_eps = split_episodes(n_ep, 0.8, args.seed)
    mask_tr = np.isin(episode_idx, train_eps)
    mask_te = np.isin(episode_idx, test_eps)

    # decide pos/vel target columns based on data shape
    if proprio.shape[1] >= 4:  # collision: 2 balls (x1, y1, x2, y2)
        pos_x = proprio[:, [0, 2]]
        vel_x = state[:, [0, 2]]  # (vx1, vx2)
    else:                       # uniform_motion: 1 ball, no state field — action is velocity
        pos_x = proprio[:, [0]]
        vel_x = action[:, [0]]
    targets = {
        "pos_x": pos_x,
        "vel_x": vel_x,
    }

    print(f"\nloading ckpt {args.ckpt} ...", flush=True)
    trained = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"  loaded {type(trained).__name__}")
    emb = extract_embeddings(trained, pix_uint8, batch_size=args.batch_size,
                             device=args.device, use_projector=not args.no_projector)
    del trained
    torch.cuda.empty_cache()

    # K=1: same as single-frame baseline (sanity)
    feats1, valid1 = build_multiframe_feats(emb, episode_idx, step_idx, K=1)
    evaluate(f"single-frame K=1 (proj={not args.no_projector})", feats1, valid1, mask_tr, mask_te, targets)

    # K=args.K: multi-frame
    featsK, validK = build_multiframe_feats(emb, episode_idx, step_idx, K=args.K)
    evaluate(f"multi-frame K={args.K} (proj={not args.no_projector})", featsK, validK, mask_tr, mask_te, targets)


if __name__ == "__main__":
    main()
