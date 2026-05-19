"""Encode phyworld_collision frames through baseline encoders for the 8-way
comparison table. Three modes:

  · random_vit_tiny   — fresh ViT-tiny, NO pretraining (controls for "encoder
                        architecture alone, no training")
  · imagenet_vit_tiny — timm vit_tiny_patch16_224.augreg_in21k_ft_in1k
                        (5.5M params, ImageNet-21k → ImageNet-1k pretrained)
  · pixel_stats       — 9-D feature: per-channel mean / std / mean² over
                        spatial dims (no encoder; CPU only)

Saves embeddings to wm/artifacts/embeddings/{mode}_collision_emb_32k.npy.
Then probe_all_targets.py can be run on each.
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


def encode_with_vit(model, pix_chw, *, batch_size=128, device="cuda"):
    """Generic CLS-token encoder over a uint8 (N, 3, 224, 224) tensor.
    Uses ImageNet normalization. Returns (N, D) fp32 numpy.
    """
    model.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    N = pix_chw.shape[0]
    out = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            x = pix_chw[i:i + batch_size].to(device, non_blocking=True).float() / 255.0
            x = (x - mean) / std
            # timm models expose forward_features → (B, n_tokens, D); take cls
            feats = model.forward_features(x)
            if feats.ndim == 3:
                emb = feats[:, 0]  # cls token
            else:
                emb = feats  # already pooled
            out.append(emb.float().cpu().numpy())
            if (i // batch_size) % 50 == 0:
                print(f"  encoded {i + x.shape[0]}/{N}", flush=True)
    return np.concatenate(out, axis=0)


def encode_pixel_stats(pix_chw):
    """9-D feature: per-channel mean / std / mean² over spatial dims.
    pix_chw: (N, 3, H, W) uint8 tensor → (N, 9) fp32.
    """
    x = pix_chw.float() / 255.0  # (N, 3, H, W)
    flat = x.flatten(2)  # (N, 3, H*W)
    means = flat.mean(-1)             # (N, 3)
    stds = flat.std(-1)               # (N, 3)
    means_sq = (flat ** 2).mean(-1)   # (N, 3)
    feat = torch.cat([means, stds, means_sq], dim=1).numpy().astype(np.float32)
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random_vit_tiny", "imagenet_vit_tiny", "pixel_stats"], required=True)
    ap.add_argument("--data", default=os.path.expanduser("~/.stable_worldmodel/phyworld_collision.h5"))
    ap.add_argument("--n-frames", type=int, default=32000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    out_dir = os.path.expanduser("~/agent_memory/wm/artifacts/embeddings")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.mode}_collision_emb_32k.npy")

    print(f"[load] {args.data}", flush=True)
    with h5py.File(args.data, "r") as f:
        N = min(args.n_frames, f["pixels"].shape[0])
        pix = f["pixels"][:N]
    pix_chw = torch.from_numpy(pix).permute(0, 3, 1, 2).contiguous()
    print(f"  pix shape: {pix_chw.shape}", flush=True)

    t0 = time.time()
    if args.mode == "pixel_stats":
        emb = encode_pixel_stats(pix_chw)
    elif args.mode == "random_vit_tiny":
        import timm
        # ViT-tiny patch16, random init (pretrained=False), match the LeWM
        # encoder's 192-D hidden size for a fair "same architecture, no weights" control.
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  random ViT-tiny: {n_params:.2f} M params", flush=True)
        emb = encode_with_vit(model, pix_chw, batch_size=args.batch_size, device=args.device)
    elif args.mode == "imagenet_vit_tiny":
        import timm
        # timm's vit_tiny patch16 224 pretrained on ImageNet-21k then fine-tuned on ImageNet-1k.
        # This is the closest comparable to LeWM's 5.5M ViT-tiny but with ImageNet visual pretrain.
        model = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  ImageNet ViT-tiny: {n_params:.2f} M params", flush=True)
        emb = encode_with_vit(model, pix_chw, batch_size=args.batch_size, device=args.device)

    print(f"  emb shape: {emb.shape}, encoding took {time.time()-t0:.1f}s", flush=True)
    np.save(out_path, emb)
    print(f"  saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
