"""DiT-XL-2-256 LoRA fine-tune on phyworld_collision frames, then re-probe.

Pipeline:
  1. Load DiT-XL transformer + SD-VAE (frozen for both).
  2. Wrap last 4 transformer blocks with LoRA (rank=16) on attention Q/K/V/out.
  3. Train LoRA params on phyworld_collision frames with standard DDPM ε-pred
     loss; class_labels=null (1000), random timestep per sample.
  4. Save LoRA weights.
  5. Re-extract embeddings (same hook as probe_dit_zeroshot.py: last block,
     mean-pool, 1152-D) **with LoRA active**.
  6. Run the same K=1 / K=4 probe protocol as probe_dit_zeroshot.py.

Comparison target: probe_dit_zeroshot.py results (DiT-XL zero-shot, NO FT).
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
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from diffusers import AutoencoderKL, DiTTransformer2DModel, DDPMScheduler
from peft import LoraConfig, get_peft_model


# ----------------------------- data -----------------------------

class CollisionFrameDataset(Dataset):
    """Loads phyworld_collision frames; yields uint8 (3, 224, 224)."""
    def __init__(self, h5_path: str, n_frames: int):
        self.h5_path = h5_path
        self.n = n_frames
        self._f = None  # lazy open in worker

    def _ensure_open(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self._ensure_open()
        pix = self._f["pixels"][idx]   # (224, 224, 3) uint8
        pix = np.transpose(pix, (2, 0, 1))
        return torch.from_numpy(pix)  # (3, 224, 224) uint8


# ----------------------------- helpers -----------------------------

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


def fit_eval_collision(emb, prop, state, coll, ep_idx, n_ep, *, train_frac=0.8, seed=0, K=1, valid_mask=None):
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
    print(f"  target              R² / AUC")
    m = Ridge(alpha=1.0); m.fit(e_tr, pos_x[mask_tr])
    print(f"  pos_x R²            {r2_score(pos_x[mask_te], m.predict(e_te)):+.4f}")
    m = Ridge(alpha=1.0); m.fit(e_tr, vel_x[mask_tr])
    print(f"  vel_x R²            {r2_score(vel_x[mask_te], m.predict(e_te)):+.4f}")
    if coll[mask_tr].sum() > 1 and coll[mask_tr].sum() < mask_tr.sum():
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
        clf.fit(e_tr, coll[mask_tr])
        p = clf.predict_proba(e_te)[:, 1]
        print(f"  collision AUC       {roc_auc_score(coll[mask_te], p):.4f}")


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit-dir", default="/home/qlib/.stable_worldmodel/dit_xl_2_256")
    ap.add_argument("--data", default=os.path.expanduser("~/.stable_worldmodel/phyworld_collision.h5"))
    ap.add_argument("--n-frames", type=int, default=32000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--num-last-blocks", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default=os.path.expanduser("~/agent_memory/wm/artifacts/embeddings/dit_xl_lora_collision"))
    ap.add_argument("--save-every-epochs", type=int, default=0,
                    help="If >0, save a LoRA checkpoint every N epochs and probe each at the end (trajectory mode).")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device

    os.makedirs(args.save_dir, exist_ok=True)

    # ----- load models -----
    print(f"[load] DiT-XL from {args.dit_dir}", flush=True)
    # VAE frozen in fp16 (encoding only)
    vae = AutoencoderKL.from_pretrained(args.dit_dir, subfolder="vae", torch_dtype=torch.float16).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Transformer: keep frozen in fp16 base weights, LoRA in fp32 for stable training
    transformer = DiTTransformer2DModel.from_pretrained(args.dit_dir, subfolder="transformer", torch_dtype=torch.float16).to(device)
    for p in transformer.parameters():
        p.requires_grad_(False)

    n_blocks = len(transformer.transformer_blocks)
    print(f"  transformer has {n_blocks} blocks; applying LoRA to last {args.num_last_blocks}")

    # Build target module list: last K blocks × attention {q,k,v,o}
    target_modules = []
    for blk_i in range(n_blocks - args.num_last_blocks, n_blocks):
        for name in ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"]:
            target_modules.append(f"transformer_blocks.{blk_i}.{name}")

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_cfg)
    n_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in transformer.parameters())
    print(f"  LoRA params: {n_trainable/1e6:.3f} M trainable / {n_total/1e6:.1f} M total ({100*n_trainable/n_total:.3f}%)")

    # Scheduler (matches DiT release config: 1000 steps, linear beta)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                              beta_start=0.0001, beta_end=0.02)
    scheduler.set_timesteps(1000)

    # ----- data -----
    with h5py.File(args.data, "r") as f:
        total_frames = f["pixels"].shape[0]
        n_frames = min(args.n_frames, total_frames)
    dataset = CollisionFrameDataset(args.data, n_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    print(f"  using {n_frames} training frames, {len(loader)} steps / epoch", flush=True)

    # ----- training loop -----
    optim = torch.optim.AdamW([p for p in transformer.parameters() if p.requires_grad],
                               lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    transformer.train()
    t_train = time.time()
    global_step = 0
    log_every = 50
    saved_ckpt_dirs = []  # list of (epoch_int, dir_path) for trajectory mode
    for epoch in range(args.epochs):
        for step, pix_u8 in enumerate(loader):
            pix_u8 = pix_u8.to(device, non_blocking=True)
            with torch.no_grad():
                x = pix_u8.to(torch.float16) / 127.5 - 1.0
                if x.shape[-1] != args.img_size:
                    x = F.interpolate(x, size=args.img_size, mode="bilinear", align_corners=False)
                latent = vae.encode(x).latent_dist.mean * vae.config.scaling_factor  # (B,4,32,32) fp16

            # add noise
            bsz = latent.shape[0]
            t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(latent)
            noisy = scheduler.add_noise(latent, noise, t)

            class_labels = torch.full((bsz,), 1000, dtype=torch.long, device=device)
            pred = transformer(noisy, timestep=t, class_labels=class_labels, return_dict=False)[0]
            # DiT predicts (B, 8, 32, 32) -- first 4 are noise pred, last 4 are variance (learned variance)
            pred_noise = pred[:, :4]
            loss = F.mse_loss(pred_noise.float(), noise.float())

            optim.zero_grad()
            loss.backward()
            optim.step()
            global_step += 1

            if global_step % log_every == 0 or step == 0:
                elapsed = time.time() - t_train
                total_steps = args.epochs * len(loader)
                eta = elapsed / global_step * (total_steps - global_step) if global_step > 0 else 0
                print(f"  ep {epoch} step {step}/{len(loader)} (global {global_step}/{total_steps}) "
                      f"loss={loss.item():.4f} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

        # Trajectory mode: save LoRA every N epochs
        if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
            ck_dir = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
            os.makedirs(ck_dir, exist_ok=True)
            transformer.save_pretrained(ck_dir)
            saved_ckpt_dirs.append((epoch + 1, ck_dir))
            print(f"  [ckpt] saved LoRA at end of epoch {epoch + 1} → {ck_dir}", flush=True)

    print(f"\n[train done] total {time.time()-t_train:.0f}s, {global_step} steps")

    # Final save (always)
    transformer.save_pretrained(args.save_dir)
    print(f"  LoRA weights saved to {args.save_dir}")
    # If trajectory mode and the final epoch was already captured, don't duplicate
    if args.save_every_epochs > 0 and (saved_ckpt_dirs and saved_ckpt_dirs[-1][0] != args.epochs):
        ck_dir = os.path.join(args.save_dir, f"epoch_{args.epochs}")
        os.makedirs(ck_dir, exist_ok=True)
        transformer.save_pretrained(ck_dir)
        saved_ckpt_dirs.append((args.epochs, ck_dir))

    # ----- helper: encode all frames with current transformer state -----
    def encode_now(label):
        transformer.eval()
        feats_local = {}
        def hook(_m, _in, out):
            feats_local["last"] = out if isinstance(out, torch.Tensor) else out[0]
        base_blocks = transformer.base_model.model.transformer_blocks
        h_handle = base_blocks[-1].register_forward_hook(hook)
        embs_local = np.empty((n_frames, 1152), dtype=np.float32)
        t_enc = time.time()
        with torch.no_grad():
            with h5py.File(args.data, "r") as f:
                for i in range(0, n_frames, args.batch_size):
                    pix_u8 = f["pixels"][i:i + args.batch_size]
                    pix_chw = torch.from_numpy(np.transpose(pix_u8, (0, 3, 1, 2)))
                    x = pix_chw.to(device, non_blocking=True).to(torch.float16) / 127.5 - 1.0
                    if x.shape[-1] != args.img_size:
                        x = F.interpolate(x, size=args.img_size, mode="bilinear", align_corners=False)
                    latent = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
                    B = latent.shape[0]
                    tt = torch.zeros(B, dtype=torch.long, device=device)
                    cls = torch.full((B,), 1000, dtype=torch.long, device=device)
                    transformer(latent, timestep=tt, class_labels=cls, return_dict=False)
                    pooled = feats_local["last"].mean(dim=1).float().cpu().numpy()
                    embs_local[i:i + B] = pooled
                    if (i // args.batch_size) % 50 == 0:
                        print(f"    [{label}] encoded {i + B}/{n_frames}", flush=True)
        h_handle.remove()
        print(f"  [{label}] encoding done in {time.time()-t_enc:.0f}s", flush=True)
        return embs_local

    # ----- load probe targets -----
    with h5py.File(args.data, "r") as f:
        prop = f["proprio"][:n_frames]
        state = f["state"][:n_frames]
        coll = f["collision_event"][:n_frames]
        ep_idx = f["episode_idx"][:n_frames]
        step_idx = f["step_idx"][:n_frames]
    n_ep = len(np.unique(ep_idx))

    def probe_emb(embs_in, label):
        print(f"\n=== {label} ===")
        print(f"\n[K=1 single-frame]")
        fit_eval_collision(embs_in, prop, state, coll, ep_idx, n_ep, seed=args.seed, K=1)
        print(f"\n[K=4 multi-frame]")
        feats_k4, valid_k4 = build_multiframe_feats(embs_in, ep_idx, step_idx, K=4)
        fit_eval_collision(feats_k4, prop, state, coll, ep_idx, n_ep, seed=args.seed, K=4, valid_mask=valid_k4)

    # ----- final / single-state probe -----
    if not saved_ckpt_dirs:
        # Non-trajectory mode: probe current (final) state
        embs = encode_now("final")
        emb_cache = os.path.join(os.path.dirname(args.save_dir), "dit_xl_lora_collision_emb_32k.npy")
        np.save(emb_cache, embs)
        print(f"  cached embeddings to {emb_cache}")
        probe_emb(embs, f"DiT-XL + LoRA fine-tuned ({args.epochs} epoch)")
    else:
        # Trajectory mode: probe each saved checkpoint
        from peft import PeftModel
        print(f"\n[trajectory] probing {len(saved_ckpt_dirs)} intermediate checkpoints ...")
        for ep_num, ck_dir in saved_ckpt_dirs:
            print(f"\n--- loading LoRA from epoch {ep_num} ---")
            # Load adapter weights from this checkpoint into current transformer
            # (the LoRA structure is already set up; we just need to swap weights)
            from safetensors.torch import load_file
            adapter_path = os.path.join(ck_dir, "adapter_model.safetensors")
            if not os.path.exists(adapter_path):
                adapter_path = os.path.join(ck_dir, "adapter_model.bin")
            sd = load_file(adapter_path) if adapter_path.endswith(".safetensors") else torch.load(adapter_path, map_location=device)
            # PEFT-saved keys often look like "base_model.model....lora_A.weight" -- match against current model
            missing, unexpected = transformer.load_state_dict(sd, strict=False)
            print(f"  loaded {len(sd)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
            embs_ep = encode_now(f"epoch_{ep_num}")
            emb_cache_ep = os.path.join(os.path.dirname(args.save_dir),
                                         f"dit_xl_lora_collision_emb_32k_epoch{ep_num}.npy")
            np.save(emb_cache_ep, embs_ep)
            print(f"  cached embeddings to {emb_cache_ep}")
            probe_emb(embs_ep, f"DiT-XL + LoRA epoch {ep_num}")

    print(f"\nTotal wall time {time.time()-t_train:.0f}s")


if __name__ == "__main__":
    main()
