"""OOD probe on phyworld uniform_motion.

**Caveat**: unlike collision (where train was 30K ID-only and eval was a separate
mixed file), our uniform_motion encoder was trained on the **entire**
`uniform_motion_eval.hdf5` (1152 traj), which already contained OOD trajectories.
So the encoder is NOT OOD-naive. However, the **probe Ridge is still fit only on
ID-partition frames**, and tested on each partition — so the test measures
whether the encoder's linear pos/vel readout transfers from ID to OOD radii/
velocities.

Partitions (phyworld paper):
    ID  : r ∈ [0.7, 1.5], v ∈ [1, 4]
    OOD : r ∈ [0.3, 0.6] ∪ [1.5, 2.0], v ∈ [0, 0.8] ∪ [4.5, 6.0]

Usage:
    python scripts/probe_ood_uniform.py \\
        --ckpt ~/.stable_worldmodel/uniform_paperinit/lewm_uniform_paperinit_epoch_20_object.ckpt \\
        --lewm-data   ~/.stable_worldmodel/phyworld_uniform_motion.h5 \\
        --source-hdf5 phyworld/data/uniform_motion_eval.hdf5
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

PART_NAMES = {0: "ID", 1: "r-OOD", 2: "v-OOD", 3: "both-OOD"}


def build_random_encoder(img_size=224, patch_size=14, scale="tiny"):
    import stable_pretraining as spt
    return spt.backbone.utils.vit_hf(scale, patch_size=patch_size, image_size=img_size,
                                      pretrained=False, use_mask_token=False)


def partition_label(r: float, v: float, r_id=(0.7, 1.5), v_id=(1.0, 4.0)) -> int:
    r_ok = r_id[0] <= r <= r_id[1]
    v_ok = v_id[0] <= abs(v) <= v_id[1]
    if r_ok and v_ok: return 0
    if not r_ok and v_ok: return 1
    if r_ok and not v_ok: return 2
    return 3


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


def make_pixel_stats(pix_uint8):
    N = pix_uint8.shape[0]
    feats = np.empty((N, 9), dtype=np.float32)
    for i in range(0, N, 1024):
        x = pix_uint8[i:i + 1024].float() / 255.0
        flat = x.flatten(2)
        m = flat.mean(-1); s = flat.std(-1)
        feats[i:i + x.shape[0]] = torch.cat([m, s, m ** 2], dim=1).numpy()
    return feats


def load_uniform_with_partitions(lewm_path, source_path):
    """Load lewm h5 + match partitions from source hdf5 by traj order."""
    with h5py.File(lewm_path, "r") as f:
        pix = f["pixels"][:]
        prop = f["proprio"][:]
        action = f["action"][:]
        ep_idx = f["episode_idx"][:]
        ep_len = f["ep_len"][:]
    pix_u8 = torch.from_numpy(pix).permute(0, 3, 1, 2).contiguous()

    with h5py.File(source_path, "r") as f:
        init = np.concatenate([f["init_streams"][k][...] for k in sorted(f["init_streams"])], 0)
    n_traj = len(init)
    assert n_traj == len(ep_len), f"traj count mismatch: lewm has {len(ep_len)}, source has {n_traj}"

    parts_per_traj = np.array(
        [partition_label(float(init[i, 0]), float(init[i, 1])) for i in range(n_traj)],
        dtype=np.uint8,
    )
    parts_per_frame = parts_per_traj[ep_idx]
    print(f"  partitions (trajs): "
          f"{ {PART_NAMES[k]: int((parts_per_traj==k).sum()) for k in range(4)} }")
    return pix_u8, prop, action, ep_idx, parts_per_frame


def fit_and_eval(label, emb_full, prop_x, vel_x, parts):
    """Fit Ridge on ID frames, evaluate on each partition."""
    print(f"\n=== {label} ===  emb dim={emb_full.shape[1]}", flush=True)

    id_mask = (parts == 0)
    if id_mask.sum() < 200:
        print(f"  too few ID frames ({id_mask.sum()}), skipping")
        return

    mu, sigma = emb_full[id_mask].mean(0), emb_full[id_mask].std(0) + 1e-6
    emb_norm = (emb_full - mu) / sigma

    # Fit on ID-only
    ridge_pos = Ridge(alpha=1.0); ridge_pos.fit(emb_norm[id_mask], prop_x[id_mask])
    ridge_vel = Ridge(alpha=1.0); ridge_vel.fit(emb_norm[id_mask], vel_x[id_mask])

    print(f"  {'partition':12s} {'n_frames':>9s}   pos_x R²    vx R²")
    for p in sorted(np.unique(parts)):
        mask = (parts == p)
        if mask.sum() < 50:
            continue
        n = mask.sum()
        if p == 0:
            # use leave-out via train/test split inside ID for fair fit-test scenario
            # but here we evaluate on the same ID fit set, giving a high-bound -- mark
            r2_pos = r2_score(prop_x[mask], ridge_pos.predict(emb_norm[mask]))
            r2_vel = r2_score(vel_x[mask], ridge_vel.predict(emb_norm[mask]))
            print(f"  {PART_NAMES[p]:12s} {n:>9d}   {r2_pos:+.4f}*   {r2_vel:+.4f}*  (* = fit==eval set)")
        else:
            r2_pos = r2_score(prop_x[mask], ridge_pos.predict(emb_norm[mask]))
            r2_vel = r2_score(vel_x[mask], ridge_vel.predict(emb_norm[mask]))
            print(f"  {PART_NAMES[p]:12s} {n:>9d}   {r2_pos:+.4f}    {r2_vel:+.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lewm-data",   default=os.path.expanduser("~/.stable_worldmodel/phyworld_uniform_motion.h5"))
    ap.add_argument("--source-hdf5", default="/home/qlib/agent_memory/wm/phyworld/data/uniform_motion_eval.hdf5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--no-projector", action="store_true")
    ap.add_argument("--skip-random", action="store_true")
    ap.add_argument("--skip-pixel", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[load] {args.lewm_data} + partitions from {args.source_hdf5}", flush=True)
    pix_u8, prop, action, ep_idx, parts = load_uniform_with_partitions(args.lewm_data, args.source_hdf5)
    print(f"  {pix_u8.shape[0]} frames total", flush=True)

    prop_x = prop[:, [0]]
    vel_x = action[:, [0]]

    # 1) trained encoder
    print(f"\n[encoder] loading {args.ckpt}", flush=True)
    trained = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"  loaded {type(trained).__name__}")
    emb = extract_embeddings(trained, pix_u8, batch_size=args.batch_size,
                             device=args.device, use_projector=not args.no_projector)
    fit_and_eval("trained encoder", emb, prop_x, vel_x, parts)
    del trained, emb; torch.cuda.empty_cache()

    # 2) random
    if not args.skip_random:
        print(f"\n[encoder] random ViT-tiny baseline", flush=True)
        rand = build_random_encoder()
        emb = extract_embeddings(rand, pix_u8, batch_size=args.batch_size,
                                 device=args.device, use_projector=False)
        fit_and_eval("random encoder", emb, prop_x, vel_x, parts)
        del rand, emb; torch.cuda.empty_cache()

    # 3) pixel-stats
    if not args.skip_pixel:
        print(f"\n[encoder] pixel-stats baseline (9-D)", flush=True)
        feat = make_pixel_stats(pix_u8)
        fit_and_eval("pixel-stats", feat, prop_x, vel_x, parts)

    print(f"\nTotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
