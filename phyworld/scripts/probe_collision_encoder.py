"""Probe le-wm encoder on phyworld collision: does it encode physics?

Targets (more interesting than uniform_motion since 2D / 2-ball / collision):
    pos_x1, pos_x2     -- per-ball horizontal position (linear localisation)
    vel_x1, vel_x2     -- per-ball velocity (REQUIRES temporal info)
    mass_ratio m1/m2   -- per-traj scalar; encoder must "see" ball sizes
    collision_event    -- binary: did the impulse happen at this frame?

Compares three feature extractors:
    trained encoder (le-wm checkpoint) | random ViT-tiny | pixel statistics

Splits trajectories 80/20 (NOT frames) to avoid leakage.

Usage
-----
    python scripts/probe_collision_encoder.py \\
        --ckpt ~/.stable_worldmodel/collision_run/lewm_collision_epoch_15_object.ckpt \\
        --data ~/.stable_worldmodel/phyworld_collision.h5
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
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score

LEWM_ROOT = Path(__file__).resolve().parent.parent.parent / "le-wm"
sys.path.insert(0, str(LEWM_ROOT))


def build_random_encoder(img_size=224, patch_size=14, scale="tiny"):
    import stable_pretraining as spt
    return spt.backbone.utils.vit_hf(scale, patch_size=patch_size, image_size=img_size,
                                      pretrained=False, use_mask_token=False)


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def extract_embeddings(model, pix_uint8, *, batch_size=128, device="cuda",
                       use_projector=True, img_size=224):
    """pix_uint8: (N, 3, H, W) uint8 tensor on CPU. Normalize per-batch on GPU."""
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

    mean = _IMAGENET_MEAN.to(device)
    std = _IMAGENET_STD.to(device)

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


def load_data(h5_path, img_size=224):
    """Returns pixels as uint8 (N, 3, H, W) — much smaller than float32."""
    with h5py.File(h5_path, "r") as f:
        pix = f["pixels"][:]                # (N, H, W, 3) uint8
        proprio = f["proprio"][:]
        state = f["state"][:]
        mass = f["mass"][:]
        coll = f["collision_event"][:]
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        episode_idx = f["episode_idx"][:]

    # uint8 NCHW: 160k * 224*224*3 ≈ 24 GB (vs 96 GB if float32)
    x = torch.from_numpy(pix).permute(0, 3, 1, 2).contiguous()  # uint8
    return x, proprio, state, mass, coll, ep_len, ep_offset, episode_idx


def split_episodes(n_episodes, train_frac=0.8, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    n_train = int(round(n_episodes * train_frac))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def fit_regression_probe(emb_tr, y_tr, emb_te, y_te, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(emb_tr, y_tr)
    yp = model.predict(emb_te)
    r2 = r2_score(y_te, yp, multioutput="uniform_average")
    r2_per = r2_score(y_te, yp, multioutput="raw_values") if y_te.ndim > 1 and y_te.shape[1] > 1 else None
    rmse = float(np.sqrt(((yp - y_te) ** 2).mean()))
    return dict(r2=float(r2), r2_per_dim=r2_per, rmse=rmse)


def fit_classification_probe(emb_tr, y_tr, emb_te, y_te):
    """Binary classification probe with class-weighted Logistic Regression."""
    if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
        return dict(skipped="single-class train set")
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    clf.fit(emb_tr, y_tr)
    p = clf.predict_proba(emb_te)[:, 1]
    pred = (p > 0.5).astype(np.uint8)
    pos_frac = float(y_te.mean())
    return dict(
        accuracy=float(accuracy_score(y_te, pred)),
        roc_auc=float(roc_auc_score(y_te, p)) if y_te.std() > 0 else float("nan"),
        positive_rate=pos_frac,
        baseline_acc=max(pos_frac, 1 - pos_frac),
    )


def evaluate_one_extractor(label, emb_full, mask_tr, mask_te, targets):
    print(f"\n=== {label} ===  emb dim={emb_full.shape[1]}", flush=True)
    emb_tr_raw = emb_full[mask_tr]
    emb_te_raw = emb_full[mask_te]
    mu, sigma = emb_tr_raw.mean(0), emb_tr_raw.std(0) + 1e-6
    emb_tr = (emb_tr_raw - mu) / sigma
    emb_te = (emb_te_raw - mu) / sigma

    for name, (kind, y_tr, y_te) in targets.items():
        if kind == "regression":
            res = fit_regression_probe(emb_tr, y_tr, emb_te, y_te)
            line = f"  {name:18s} regr  R^2={res['r2']:+.4f}  RMSE={res['rmse']:.4f}"
            if res["r2_per_dim"] is not None:
                line += f"  per-dim={np.round(res['r2_per_dim'], 3).tolist()}"
            print(line)
        elif kind == "classification":
            res = fit_classification_probe(emb_tr, y_tr, emb_te, y_te)
            if "skipped" in res:
                print(f"  {name:18s} clf   skipped: {res['skipped']}")
            else:
                print(f"  {name:18s} clf   acc={res['accuracy']:.4f}  AUC={res['roc_auc']:.4f}  "
                      f"(pos_rate={res['positive_rate']:.3f}, baseline_acc={res['baseline_acc']:.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.expanduser("~/.stable_worldmodel/phyworld_collision.h5"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--no-projector", action="store_true")
    ap.add_argument("--skip-random", action="store_true")
    ap.add_argument("--skip-pixel", action="store_true")
    args = ap.parse_args()

    print(f"loading data {args.data} ...", flush=True)
    t0 = time.time()
    pix, pos, vel, mass, coll, ep_len, ep_offset, ep_idx = load_data(Path(args.data))
    n_ep = len(ep_len)
    print(f"  {n_ep} episodes, {pix.shape[0]} frames, {pix.shape[2]}x{pix.shape[3]}  ({time.time()-t0:.1f}s)")

    # split by trajectory
    train_eps, test_eps = split_episodes(n_ep, args.train_frac, args.seed)
    mask_tr = np.isin(ep_idx, train_eps)
    mask_te = np.isin(ep_idx, test_eps)
    print(f"  split: {len(train_eps)} train eps / {len(test_eps)} test eps  "
          f"({mask_tr.sum()} / {mask_te.sum()} frames)")

    # build targets
    pos_x = pos[:, [0, 2]]                         # (N, 2)  — both balls' x
    vel_x = vel[:, [0, 2]]                         # (N, 2)  — both balls' vx
    mass_ratio = (mass[:, 0] / (mass[:, 1] + 1e-6)).astype(np.float32).reshape(-1, 1)

    targets = {
        "pos_x  (2D)": ("regression", pos_x[mask_tr], pos_x[mask_te]),
        "vel_x  (2D)": ("regression", vel_x[mask_tr], vel_x[mask_te]),
        "mass_ratio": ("regression", mass_ratio[mask_tr], mass_ratio[mask_te]),
        "collision_event": ("classification", coll[mask_tr], coll[mask_te]),
    }

    # 1) trained encoder
    print(f"\nloading trained le-wm from {args.ckpt} ...", flush=True)
    trained = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"  loaded {type(trained).__name__}")
    emb_trained = extract_embeddings(trained, pix, batch_size=args.batch_size, device=args.device,
                                      use_projector=not args.no_projector)
    evaluate_one_extractor("trained encoder", emb_trained, mask_tr, mask_te, targets)
    del trained
    torch.cuda.empty_cache()

    # 2) random encoder baseline
    if not args.skip_random:
        print("\nbuilding random ViT-tiny ...", flush=True)
        rand = build_random_encoder()
        emb_rand = extract_embeddings(rand, pix, batch_size=args.batch_size, device=args.device, use_projector=False)
        evaluate_one_extractor("random encoder", emb_rand, mask_tr, mask_te, targets)
        del rand; torch.cuda.empty_cache()

    # 3) pixel-stats baseline (compute streaming to avoid 96 GB float blow-up)
    if not args.skip_pixel:
        N = pix.shape[0]
        feats = np.empty((N, 9), dtype=np.float32)  # 3 (mean) + 3 (std) + 3 (mean^2)
        for i in range(0, N, 1024):
            x = pix[i:i + 1024].float() / 255.0  # (B,3,H,W)
            flat = x.flatten(2)
            m = flat.mean(-1)
            s = flat.std(-1)
            feats[i:i + x.shape[0]] = torch.cat([m, s, m ** 2], dim=1).numpy()
        evaluate_one_extractor("pixel-stats baseline", feats, mask_tr, mask_te, targets)


if __name__ == "__main__":
    main()
