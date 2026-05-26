"""Zero-shot probe on DiT-XL-2-256 (ImageNet pretrained) for phyworld physics.

Comparison target with LeWM (5.5M, JEPA, paper-init from PushT, trained on phyworld):
  · DiT-XL  (456M, diffusion, ImageNet-pretrained, NOT fine-tuned on phyworld)

For each frame in phyworld_collision.h5:
  1. Resize 224×224 → 256×256 (DiT-XL native)
  2. VAE encode → (4, 32, 32) latent
  3. Pass through DiT-XL transformer at timestep t=0, class y=null
  4. Extract last-block hidden states (B, 256, 1152)  -- 16×16 patches × hidden=1152
  5. Mean-pool patch tokens → 1152-D embedding per frame
  6. Ridge / LogReg probe on the SAME phyworld targets we used for LeWM

Note: DiT-XL is class-conditional (1000 ImageNet classes). For our use we feed
the null class (1000 = unconditional in DiTPipeline). Timestep t=0 means no noise.

Usage:
    python scripts/probe_dit_zeroshot.py \\
        --dit-dir ~/.stable_worldmodel/dit_xl_2_256 \\
        --data ~/.stable_worldmodel/phyworld_collision.h5 \\
        --max-frames 80000
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
from diffusers import AutoencoderKL, DiTTransformer2DModel
from PIL import Image


def split_episodes(n_episodes, train_frac=0.8, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    n_train = int(round(n_episodes * train_frac))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


@torch.no_grad()
def extract_dit_embeddings(vae, transformer, pix_uint8,
                            batch_size=8, device="cuda", img_size=256, dtype=torch.float16):
    """
    pix_uint8: (N, 3, 224, 224) uint8
    return: (N, 1152) float32  -- mean-pooled patch tokens from DiT last block
    """
    vae = vae.to(device).to(dtype).eval()
    transformer = transformer.to(device).to(dtype).eval()
    N = pix_uint8.shape[0]
    embs = np.empty((N, transformer.config.num_attention_heads * transformer.config.attention_head_dim),
                    dtype=np.float32)
    # imagenet-ish normalization for VAE: scale to [-1, 1]
    for i in range(0, N, batch_size):
        x_u = pix_uint8[i:i + batch_size].to(device, non_blocking=True).to(dtype)
        x = (x_u / 127.5 - 1.0)
        if x.shape[-1] != img_size:
            x = torch.nn.functional.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
        # VAE encode
        latent = vae.encode(x).latent_dist.mean * vae.config.scaling_factor  # (B, 4, 32, 32)
        # DiT forward at t=0 with null class (=1000 for DiT-XL 1000-class)
        B = latent.shape[0]
        timestep = torch.zeros(B, dtype=torch.long, device=device)
        class_labels = torch.full((B,), 1000, dtype=torch.long, device=device)  # null
        # Hook to grab intermediate output
        feats = {}
        def hook(_m, _in, out):
            feats["last"] = out if isinstance(out, torch.Tensor) else out[0]
        # Grab output of the last transformer block (before final norm)
        h_handle = transformer.transformer_blocks[-1].register_forward_hook(hook)
        try:
            transformer(latent, timestep=timestep, class_labels=class_labels, return_dict=False)
        finally:
            h_handle.remove()
        # feats["last"] shape: (B, n_tokens, hidden); for DiT-XL/2 256×256 it's 256 tokens × 1152
        token_emb = feats["last"]
        pooled = token_emb.mean(dim=1)  # (B, hidden)
        embs[i:i + B] = pooled.float().cpu().numpy()
        if (i // batch_size) % 50 == 0:
            print(f"    DiT encoded {i + B}/{N}", flush=True)
    return embs


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


def fit_eval_collision(emb, prop, state, coll, ep_idx, step_idx, n_ep, train_frac=0.8, seed=0, K=1, valid_mask=None):
    train_eps, test_eps = split_episodes(n_ep, train_frac, seed)
    base_tr = np.isin(ep_idx, train_eps)
    base_te = np.isin(ep_idx, test_eps)
    if valid_mask is None:
        valid_mask = np.ones(len(ep_idx), dtype=bool)
    mask_tr = base_tr & valid_mask
    mask_te = base_te & valid_mask
    mu, sigma = emb[mask_tr].mean(0), emb[mask_tr].std(0) + 1e-6
    e_tr = (emb[mask_tr] - mu) / sigma
    e_te = (emb[mask_te] - mu) / sigma

    pos_x = prop[:, [0, 2]]
    vel_x = state[:, [0, 2]]
    print(f"  [K={K}] train={mask_tr.sum()}, test={mask_te.sum()}, feat_dim={emb.shape[1]}")
    print(f"  {'target':18s}  {'R² / AUC'}")
    m = Ridge(alpha=1.0); m.fit(e_tr, pos_x[mask_tr])
    print(f"  {'pos_x R²':18s}  {r2_score(pos_x[mask_te], m.predict(e_te)):+.4f}")
    m = Ridge(alpha=1.0); m.fit(e_tr, vel_x[mask_tr])
    print(f"  {'vel_x R²':18s}  {r2_score(vel_x[mask_te], m.predict(e_te)):+.4f}")
    if coll[mask_tr].sum() > 1 and coll[mask_tr].sum() < mask_tr.sum():
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
        clf.fit(e_tr, coll[mask_tr])
        p = clf.predict_proba(e_te)[:, 1]
        print(f"  {'collision AUC':18s}  {roc_auc_score(coll[mask_te], p):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dit-dir", default="/home/qlib/.stable_worldmodel/dit_xl_2_256")
    ap.add_argument("--data", default=os.path.expanduser("~/.stable_worldmodel/phyworld_collision.h5"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=80000,
                    help="Cap frames (DiT is slow; default 80k = 2500 traj × 32 frames).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[load] DiT-XL from {args.dit_dir}", flush=True)
    vae = AutoencoderKL.from_pretrained(args.dit_dir, subfolder="vae", torch_dtype=torch.float16)
    transformer = DiTTransformer2DModel.from_pretrained(args.dit_dir, subfolder="transformer", torch_dtype=torch.float16)
    n_params = sum(p.numel() for p in transformer.parameters()) / 1e6
    print(f"  DiT-XL transformer: {n_params:.1f} M params, hidden={transformer.config.num_attention_heads * transformer.config.attention_head_dim}")

    print(f"[load] data {args.data}", flush=True)
    with h5py.File(args.data, "r") as f:
        N_total = f["pixels"].shape[0]
        N = min(args.max_frames, N_total)
        # Take first N frames (contiguous trajs)
        pix = f["pixels"][:N]
        prop = f["proprio"][:N]
        state = f["state"][:N]
        coll = f["collision_event"][:N]
        ep_idx = f["episode_idx"][:N]
        step_idx = f["step_idx"][:N]
        ep_len = f["ep_len"][:]
    pix_u8 = torch.from_numpy(pix).permute(0, 3, 1, 2).contiguous()
    n_ep = len(np.unique(ep_idx))
    print(f"  using {N} frames over {n_ep} trajs", flush=True)

    emb_cache = os.path.expanduser("~/agent_memory/wm/artifacts/embeddings/dit_xl_collision_emb_32k.npy")
    os.makedirs(os.path.dirname(emb_cache), exist_ok=True)
    if os.path.exists(emb_cache):
        print(f"[cache] loading {emb_cache}", flush=True)
        emb = np.load(emb_cache)
    else:
        print(f"\n[encode] DiT-XL zero-shot ...", flush=True)
        emb = extract_dit_embeddings(vae, transformer, pix_u8,
                                      batch_size=args.batch_size, device=args.device)
        np.save(emb_cache, emb)
        print(f"  cached to {emb_cache}", flush=True)
    print(f"  emb shape: {emb.shape}", flush=True)

    print(f"\n=== DiT-XL-2-256 zero-shot (ImageNet pretrained, NOT fine-tuned on phyworld) ===")
    print(f"\n[K=1 single-frame]")
    fit_eval_collision(emb, prop, state, coll, ep_idx, step_idx, n_ep, seed=args.seed, K=1)

    print(f"\n[K=4 multi-frame]")
    feats_k4, valid_k4 = build_multiframe_feats(emb, ep_idx, step_idx, K=4)
    fit_eval_collision(feats_k4, prop, state, coll, ep_idx, step_idx, n_ep,
                       seed=args.seed, K=4, valid_mask=valid_k4)

    print(f"\nTotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
