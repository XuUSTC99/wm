"""Encode collision_eval.h5 (52k frames, 4 partitions) through various
encoders for the OOD per-partition probe. Each encoder is supported as a mode:

  · lewm_pusht_only   — load lewm_paper_pusht/weights.pt + build JEPA
  · lewm_object_ckpt  — load a *_object.ckpt (full JEPA pickle)
  · imagenet_vit_tiny — timm vit_tiny_patch16_224.augreg_in21k_ft_in1k
  · random_vit_tiny   — timm vit_tiny_patch16_224, no pretrain
  · pixel_stats       — 9-D pixel mean/std/mean² (CPU only)
  · dit_zero_shot     — DiT-XL-2 forward at t=0, null class, mean-pool patch tokens

Outputs .npy file at the path specified by --out.
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
import torch.nn.functional as F

LEWM_ROOT = Path(__file__).resolve().parent.parent.parent / "le-wm"
sys.path.insert(0, str(LEWM_ROOT))


@torch.no_grad()
def encode_lewm_jepa(model, pix_chw, *, use_projector, batch_size=128, device="cuda"):
    """Encode via LeWM JEPA encoder (cls token, optionally through projector)."""
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
        if (i // batch_size) % 100 == 0:
            print(f"  encoded {i + x.shape[0]}/{N}", flush=True)
    return np.concatenate(out, axis=0)


@torch.no_grad()
def encode_timm_vit(model, pix_chw, *, batch_size=128, device="cuda"):
    """Encode via timm ViT (forward_features → cls token)."""
    model.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    N = pix_chw.shape[0]
    out = []
    for i in range(0, N, batch_size):
        x = pix_chw[i:i + batch_size].to(device, non_blocking=True).float() / 255.0
        x = (x - mean) / std
        feats = model.forward_features(x)
        emb = feats[:, 0] if feats.ndim == 3 else feats
        out.append(emb.float().cpu().numpy())
        if (i // batch_size) % 100 == 0:
            print(f"  encoded {i + x.shape[0]}/{N}", flush=True)
    return np.concatenate(out, axis=0)


def encode_pixel_stats(pix_chw):
    """9-D feature: per-channel mean / std / mean²."""
    x = pix_chw.float() / 255.0
    flat = x.flatten(2)
    feat = torch.cat([flat.mean(-1), flat.std(-1), (flat ** 2).mean(-1)], dim=1).numpy().astype(np.float32)
    return feat


@torch.no_grad()
def encode_dit_zero_shot(pix_chw, *, dit_dir, batch_size=8, device="cuda", img_size=256):
    """Encode via DiT-XL transformer last block, mean-pool patch tokens."""
    from diffusers import AutoencoderKL, DiTTransformer2DModel
    vae = AutoencoderKL.from_pretrained(dit_dir, subfolder="vae", torch_dtype=torch.float16).to(device).eval()
    tr = DiTTransformer2DModel.from_pretrained(dit_dir, subfolder="transformer", torch_dtype=torch.float16).to(device).eval()

    feats = {}
    def hook(_m, _in, out):
        feats["last"] = out if isinstance(out, torch.Tensor) else out[0]
    h_handle = tr.transformer_blocks[-1].register_forward_hook(hook)

    N = pix_chw.shape[0]
    out_emb = np.empty((N, 1152), dtype=np.float32)
    for i in range(0, N, batch_size):
        x_u = pix_chw[i:i + batch_size].to(device, non_blocking=True).to(torch.float16)
        x = (x_u / 127.5 - 1.0)
        if x.shape[-1] != img_size:
            x = F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
        latent = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
        B = latent.shape[0]
        t = torch.zeros(B, dtype=torch.long, device=device)
        cls = torch.full((B,), 1000, dtype=torch.long, device=device)
        tr(latent, timestep=t, class_labels=cls, return_dict=False)
        pooled = feats["last"].mean(dim=1).float().cpu().numpy()
        out_emb[i:i + B] = pooled
        if (i // batch_size) % 50 == 0:
            print(f"  encoded {i + B}/{N}", flush=True)
    h_handle.remove()
    return out_emb


def build_lewm_pusht_jepa():
    """Reproduce the JEPA architecture matching lewm_paper_pusht/config.json."""
    import stable_pretraining as spt
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    encoder = spt.backbone.utils.vit_hf("tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False)
    hidden = encoder.config.hidden_size
    predictor = ARPredictor(num_frames=3, input_dim=hidden, hidden_dim=hidden, output_dim=hidden,
                             depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0)
    action_encoder = Embedder(input_dim=10, emb_dim=hidden)
    projector = MLP(input_dim=hidden, output_dim=hidden, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden, output_dim=hidden, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    return JEPA(encoder=encoder, predictor=predictor, action_encoder=action_encoder,
                projector=projector, pred_proj=pred_proj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=[
        "lewm_pusht_only", "lewm_object_ckpt", "imagenet_vit_tiny",
        "random_vit_tiny", "pixel_stats", "dit_zero_shot"
    ])
    ap.add_argument("--ckpt", default=None, help="Required for lewm_pusht_only (weights.pt) or lewm_object_ckpt (*_object.ckpt)")
    ap.add_argument("--data", default=os.path.expanduser("~/.stable_worldmodel/phyworld_collision_eval.h5"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-projector", action="store_true", help="for lewm modes: probe cls token before projector")
    ap.add_argument("--dit-dir", default="/home/qlib/.stable_worldmodel/dit_xl_2_256")
    args = ap.parse_args()

    print(f"[load] data {args.data}")
    with h5py.File(args.data, "r") as f:
        pix = f["pixels"][:]
    pix_chw = torch.from_numpy(pix).permute(0, 3, 1, 2).contiguous()
    print(f"  pix shape: {pix_chw.shape}")

    t0 = time.time()
    if args.mode == "pixel_stats":
        emb = encode_pixel_stats(pix_chw)
    elif args.mode == "random_vit_tiny":
        import timm
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
        emb = encode_timm_vit(model, pix_chw, batch_size=args.batch_size, device=args.device)
    elif args.mode == "imagenet_vit_tiny":
        import timm
        model = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
        emb = encode_timm_vit(model, pix_chw, batch_size=args.batch_size, device=args.device)
    elif args.mode == "lewm_pusht_only":
        assert args.ckpt, "--ckpt required for lewm_pusht_only"
        model = build_lewm_pusht_jepa()
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  loaded {len(sd)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
        emb = encode_lewm_jepa(model, pix_chw, use_projector=not args.no_projector,
                                batch_size=args.batch_size, device=args.device)
    elif args.mode == "lewm_object_ckpt":
        assert args.ckpt, "--ckpt required for lewm_object_ckpt"
        model = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        emb = encode_lewm_jepa(model, pix_chw, use_projector=not args.no_projector,
                                batch_size=args.batch_size, device=args.device)
    elif args.mode == "dit_zero_shot":
        emb = encode_dit_zero_shot(pix_chw, dit_dir=args.dit_dir,
                                    batch_size=args.batch_size, device=args.device)

    print(f"  emb shape: {emb.shape}, took {time.time()-t0:.0f}s")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, emb)
    print(f"  saved → {args.out}")


if __name__ == "__main__":
    main()
