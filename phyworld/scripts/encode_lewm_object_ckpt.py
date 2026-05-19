"""Load a LeWM *_object.ckpt (full JEPA pickle) → encode phyworld_collision
frames → save embeddings .npy. Then probe_all_targets.py can be run on it.

Used for probing LeWM trajectory checkpoints (e.g. epoch 4/8/12/16 from the
paper-init 16-epoch run).
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

LEWM_ROOT = Path(__file__).resolve().parent.parent.parent / "le-wm"
sys.path.insert(0, str(LEWM_ROOT))


@torch.no_grad()
def extract(model, pix_chw, *, use_projector, batch_size=128, device="cuda"):
    """Match the extract logic from probe_lewm_pusht_only.py to keep probes apples-to-apples."""
    # Handle either JEPA wrapper or raw ViT
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

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    N = pix_chw.shape[0]
    out = []
    for i in range(0, N, batch_size):
        x = pix_chw[i:i + batch_size].to(device, non_blocking=True).float() / 255.0
        x = (x - mean) / std
        h = encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        if projector is not None:
            h = projector(h)
        out.append(h.float().cpu().numpy())
        if (i // batch_size) % 50 == 0:
            print(f"  encoded {i + x.shape[0]}/{N}", flush=True)
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to *_object.ckpt (full JEPA pickle)")
    ap.add_argument("--data", default=os.path.expanduser("~/.stable_worldmodel/phyworld_collision.h5"))
    ap.add_argument("--n-frames", type=int, default=32000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-projector", action="store_true")
    ap.add_argument("--out", required=True, help="output .npy path")
    args = ap.parse_args()

    print(f"[load] ckpt {args.ckpt}", flush=True)
    model = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  loaded {type(model).__name__}, {n_params:.2f} M params", flush=True)

    print(f"[load] data {args.data}", flush=True)
    with h5py.File(args.data, "r") as f:
        N = min(args.n_frames, f["pixels"].shape[0])
        pix = f["pixels"][:N]
    pix_chw = torch.from_numpy(pix).permute(0, 3, 1, 2).contiguous()
    print(f"  pix shape: {pix_chw.shape}", flush=True)

    t0 = time.time()
    emb = extract(model, pix_chw, use_projector=not args.no_projector,
                  batch_size=args.batch_size, device=args.device)
    print(f"  emb shape: {emb.shape}, encoding took {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, emb)
    print(f"  saved → {args.out}")


if __name__ == "__main__":
    main()
