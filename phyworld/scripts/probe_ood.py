"""OOD probe on phyworld collision.

Protocol (mirrors the phyworld paper's ID/OOD generalization test, but adapted
for representation probing rather than video generation):

  1. Fit Ridge / class-balanced LogReg on **ID training frames** (our 5000-traj
     training set, all in-distribution by construction).
  2. Apply the same probe to **collision_eval.hdf5** frames, broken down by
     partition: ID / r-OOD / v-OOD / both-OOD (4 disjoint groups).
  3. Compare per-partition R² / AUC across:
       trained encoder (from-scratch | paper-init)
       random ViT-tiny baseline
       pixel-stats (9-D) baseline

  Strong negative result would be: probe R² holds on ID but collapses on
  v-OOD / r-OOD / both-OOD ⇒ encoder learned ID-specific shortcuts, not
  a transferable physical representation.

Usage
-----
    python scripts/probe_ood.py \
        --ckpt ~/.stable_worldmodel/collision_paperinit/lewm_collision_paperinit_epoch_8_object.ckpt \
        --train-data ~/.stable_worldmodel/phyworld_collision.h5 \
        --eval-data  ~/.stable_worldmodel/phyworld_collision_eval.h5 \
        --no-projector
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
from sklearn.metrics import r2_score, roc_auc_score

LEWM_ROOT = Path(__file__).resolve().parent.parent.parent / "le-wm"
sys.path.insert(0, str(LEWM_ROOT))

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

PART_NAMES = {0: "ID", 1: "r-OOD", 2: "v-OOD", 3: "both-OOD"}


def build_random_encoder(img_size=224, patch_size=14, scale="tiny"):
    import stable_pretraining as spt
    return spt.backbone.utils.vit_hf(scale, patch_size=patch_size, image_size=img_size,
                                      pretrained=False, use_mask_token=False)


@torch.no_grad()
def extract_embeddings(model, pix_uint8, *, batch_size=96, device="cuda",
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
    mean = _MEAN.to(device); std = _STD.to(device)
    N = pix_uint8.shape[0]
    out = []
    for i in range(0, N, batch_size):
        x = pix_uint8[i:i + batch_size].to(device, non_blocking=True).float() / 255.0
        x = (x - mean) / std
        if x.shape[-1] != img_size:
            x = torch.nn.functional.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
        h = encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        if projector is not None:
            h = projector(h)
        out.append(h.float().cpu().numpy())
        if (i // batch_size) % 100 == 0:
            print(f"    encoded {i + x.shape[0]}/{N}", flush=True)
    return np.concatenate(out, axis=0)


def load_dataset(path, want_partition=False):
    with h5py.File(path, "r") as f:
        pix = f["pixels"][:]
        prop = f["proprio"][:]
        state = f["state"][:]
        coll = f["collision_event"][:]
        ep_idx = f["episode_idx"][:]
        part = f["partition"][:] if want_partition and "partition" in f else None
    pix_u8 = torch.from_numpy(pix).permute(0, 3, 1, 2).contiguous()
    return pix_u8, prop, state, coll, ep_idx, part


def make_pixel_stats(pix_uint8):
    """9-D pixel statistics feature, streaming to avoid float blowup."""
    N = pix_uint8.shape[0]
    feats = np.empty((N, 9), dtype=np.float32)
    for i in range(0, N, 1024):
        x = pix_uint8[i:i + 1024].float() / 255.0
        flat = x.flatten(2)
        m = flat.mean(-1); s = flat.std(-1)
        feats[i:i + x.shape[0]] = torch.cat([m, s, m ** 2], dim=1).numpy()
    return feats


def fit_and_eval_one(label, emb_tr_full, prop_tr, state_tr, coll_tr,
                     emb_te_full, prop_te, state_te, coll_te, part_te,
                     split_ep_te):
    """Fit probe on full train, evaluate on each eval partition."""
    print(f"\n=== {label} ===  emb dim={emb_tr_full.shape[1]}", flush=True)

    # Normalize using train stats
    mu, sigma = emb_tr_full.mean(0), emb_tr_full.std(0) + 1e-6
    emb_tr = (emb_tr_full - mu) / sigma
    emb_te = (emb_te_full - mu) / sigma

    # Regression targets
    pos_tr = prop_tr[:, [0, 2]]; vel_tr = state_tr[:, [0, 2]]
    pos_te = prop_te[:, [0, 2]]; vel_te = state_te[:, [0, 2]]

    ridge_pos = Ridge(alpha=1.0); ridge_pos.fit(emb_tr, pos_tr)
    ridge_vel = Ridge(alpha=1.0); ridge_vel.fit(emb_tr, vel_tr)

    # Classification
    if coll_tr.sum() > 0 and coll_tr.sum() < len(coll_tr):
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
        clf.fit(emb_tr, coll_tr)
    else:
        clf = None

    # Per-partition evaluation
    print(f"  {'partition':12s} {'n_frames':>9s}  pos_x R²  vel_x R²  collision AUC")
    for p in sorted(np.unique(part_te)):
        mask = (part_te == p)
        if mask.sum() < 100:
            continue
        n = mask.sum()
        r2_pos = r2_score(pos_te[mask], ridge_pos.predict(emb_te[mask]))
        r2_vel = r2_score(vel_te[mask], ridge_vel.predict(emb_te[mask]))
        if clf is not None and coll_te[mask].sum() > 1 and coll_te[mask].std() > 0:
            p_pos = clf.predict_proba(emb_te[mask])[:, 1]
            auc = roc_auc_score(coll_te[mask], p_pos)
            auc_str = f"{auc:.4f}"
        else:
            auc_str = "N/A"
        print(f"  {PART_NAMES[p]:12s} {n:>9d}   {r2_pos:+.4f}   {r2_vel:+.4f}    {auc_str}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-data", default=os.path.expanduser("~/.stable_worldmodel/phyworld_collision.h5"))
    ap.add_argument("--eval-data",  default=os.path.expanduser("~/.stable_worldmodel/phyworld_collision_eval.h5"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--no-projector", action="store_true")
    ap.add_argument("--skip-random", action="store_true")
    ap.add_argument("--skip-pixel", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[load] train={args.train_data}", flush=True)
    pix_tr, prop_tr, state_tr, coll_tr, ep_tr, _ = load_dataset(args.train_data)
    print(f"  {pix_tr.shape[0]} train frames", flush=True)
    print(f"[load] eval ={args.eval_data}", flush=True)
    pix_te, prop_te, state_te, coll_te, ep_te, part_te = load_dataset(args.eval_data, want_partition=True)
    print(f"  {pix_te.shape[0]} eval frames", flush=True)
    print(f"  partitions: {dict(zip(*np.unique(part_te, return_counts=True)))}", flush=True)

    # 1) trained encoder
    print(f"\n[encoder] loading {args.ckpt}", flush=True)
    trained = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"  loaded {type(trained).__name__}")
    print("  extracting on train frames ...", flush=True)
    emb_tr = extract_embeddings(trained, pix_tr, batch_size=args.batch_size,
                                device=args.device, use_projector=not args.no_projector)
    print("  extracting on eval frames ...", flush=True)
    emb_te = extract_embeddings(trained, pix_te, batch_size=args.batch_size,
                                device=args.device, use_projector=not args.no_projector)
    fit_and_eval_one("trained encoder", emb_tr, prop_tr, state_tr, coll_tr,
                     emb_te, prop_te, state_te, coll_te, part_te, None)
    del trained, emb_tr, emb_te
    torch.cuda.empty_cache()

    # 2) random encoder
    if not args.skip_random:
        print(f"\n[encoder] random ViT-tiny baseline", flush=True)
        rand = build_random_encoder()
        emb_tr = extract_embeddings(rand, pix_tr, batch_size=args.batch_size,
                                    device=args.device, use_projector=False)
        emb_te = extract_embeddings(rand, pix_te, batch_size=args.batch_size,
                                    device=args.device, use_projector=False)
        fit_and_eval_one("random encoder", emb_tr, prop_tr, state_tr, coll_tr,
                         emb_te, prop_te, state_te, coll_te, part_te, None)
        del rand, emb_tr, emb_te
        torch.cuda.empty_cache()

    # 3) pixel-stats baseline
    if not args.skip_pixel:
        print(f"\n[encoder] pixel-stats baseline (9-D)", flush=True)
        feat_tr = make_pixel_stats(pix_tr)
        feat_te = make_pixel_stats(pix_te)
        fit_and_eval_one("pixel-stats", feat_tr, prop_tr, state_tr, coll_tr,
                         feat_te, prop_te, state_te, coll_te, part_te, None)

    print(f"\nTotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
