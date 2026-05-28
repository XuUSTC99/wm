"""DiT-XL-2-256 LoRA fine-tune on phyworld {collision, uniform_motion, parabola}.

Differences vs dit_lora_finetune_probe.py (5/13):
  - Multi-domain via --domain
  - Gradient clipping (max_norm=1.0) to fix the NaN-at-epoch-8 collapse
  - No inline R² probe; just saves emb at standardized path so
    probe_mlp_mse_pearson.py can pick it up
  - Encodes the *probe* h5 (full eval data, all OOD partitions), not the FT
    train h5 — required for OOD-partition probes
"""
from __future__ import annotations
import argparse, os, sys, time
import h5py, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL, DiTTransformer2DModel, DDPMScheduler
from peft import LoraConfig, get_peft_model


# Per-domain data paths — ID-only training (PhyWorld official 30K subset, 1000 trajs)
DOMAINS = {
    "collision": {
        "ft_data":    "/home/qlib/.stable_worldmodel/phyworld_collision_id1k.h5",
        "probe_data": "/home/qlib/.stable_worldmodel/phyworld_collision_eval.h5",
        "ft_frames":  32000,
        "emb_out":    "/home/qlib/agent_memory/wm/artifacts/embeddings/dit_xl_lora_id1k_collision_eval_emb_52k.npy",
        "lora_dir":   "/home/qlib/agent_memory/wm/artifacts/embeddings/dit_xl_lora_id1k_collision",
    },
    "uniform_motion": {
        "ft_data":    "/home/qlib/.stable_worldmodel/phyworld_uniform_motion_id1k.h5",
        "probe_data": "/home/qlib/.stable_worldmodel/phyworld_uniform_motion.h5",
        "ft_frames":  32000,
        "emb_out":    "/home/qlib/agent_memory/wm/artifacts/embeddings/dit_xl_lora_id1k_uniform_motion_emb_37k.npy",
        "lora_dir":   "/home/qlib/agent_memory/wm/artifacts/embeddings/dit_xl_lora_id1k_uniform_motion",
    },
    "parabola": {
        "ft_data":    "/home/qlib/.stable_worldmodel/phyworld_parabola_id1k.h5",
        "probe_data": "/home/qlib/.stable_worldmodel/phyworld_parabola.h5",
        "ft_frames":  32000,
        "emb_out":    "/home/qlib/agent_memory/wm/artifacts/embeddings/dit_xl_lora_id1k_parabola_emb.npy",
        "lora_dir":   "/home/qlib/agent_memory/wm/artifacts/embeddings/dit_xl_lora_id1k_parabola",
    },
}


class FrameDataset(Dataset):
    def __init__(self, h5_path, n_frames):
        self.h5_path = h5_path; self.n = n_frames; self._f = None
    def __len__(self): return self.n
    def __getitem__(self, idx):
        if self._f is None: self._f = h5py.File(self.h5_path, "r")
        pix = self._f["pixels"][idx]
        return torch.from_numpy(np.transpose(pix, (2, 0, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), required=True)
    ap.add_argument("--dit-dir", default="/home/qlib/.stable_worldmodel/dit_xl_2_256")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="Max gradient norm. Fix for NaN-at-ep8 from 5/13.")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--num-last-blocks", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = DOMAINS[args.domain]
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda"
    os.makedirs(cfg["lora_dir"], exist_ok=True)

    print(f"[domain] {args.domain}")
    print(f"  ft data:    {cfg['ft_data']} ({cfg['ft_frames']} frames)")
    print(f"  probe data: {cfg['probe_data']}")
    print(f"  emb out:    {cfg['emb_out']}", flush=True)

    # ---- models ----
    print(f"[load] DiT-XL from {args.dit_dir}", flush=True)
    vae = AutoencoderKL.from_pretrained(args.dit_dir, subfolder="vae",
                                         torch_dtype=torch.float16).to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    transformer = DiTTransformer2DModel.from_pretrained(args.dit_dir, subfolder="transformer",
                                                         torch_dtype=torch.float16).to(device)
    for p in transformer.parameters(): p.requires_grad_(False)

    n_blocks = len(transformer.transformer_blocks)
    target_modules = []
    for blk_i in range(n_blocks - args.num_last_blocks, n_blocks):
        for nm in ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"]:
            target_modules.append(f"transformer_blocks.{blk_i}.{nm}")
    lora_cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                          target_modules=target_modules, lora_dropout=0.0, bias="none")
    transformer = get_peft_model(transformer, lora_cfg)
    n_train = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"  LoRA: {n_train/1e6:.3f}M trainable", flush=True)

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                              beta_start=0.0001, beta_end=0.02)

    # ---- data ----
    with h5py.File(cfg["ft_data"], "r") as f:
        total = f["pixels"].shape[0]
    n_frames = min(cfg["ft_frames"], total)
    dataset = FrameDataset(cfg["ft_data"], n_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    print(f"  FT frames: {n_frames}, steps/epoch: {len(loader)}", flush=True)

    # ---- training ----
    optim = torch.optim.AdamW([p for p in transformer.parameters() if p.requires_grad],
                               lr=args.lr, betas=(0.9, 0.999))
    transformer.train()
    t0 = time.time(); gstep = 0
    last_lora_norm = 0.0
    for ep in range(args.epochs):
        for step, pix in enumerate(loader):
            pix = pix.to(device, non_blocking=True)
            with torch.no_grad():
                x = pix.to(torch.float16) / 127.5 - 1.0
                if x.shape[-1] != args.img_size:
                    x = F.interpolate(x, size=args.img_size, mode="bilinear", align_corners=False)
                latent = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
            bsz = latent.shape[0]
            t = torch.randint(0, 1000, (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(latent)
            noisy = scheduler.add_noise(latent, noise, t)
            cls = torch.full((bsz,), 1000, dtype=torch.long, device=device)
            pred = transformer(noisy, timestep=t, class_labels=cls, return_dict=False)[0]
            loss = F.mse_loss(pred[:, :4].float(), noise.float())
            optim.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in transformer.parameters() if p.requires_grad], args.grad_clip)
            optim.step()
            gstep += 1
            if gstep % 50 == 0 or step == 0:
                with torch.no_grad():
                    lora_w_norms = [p.detach().float().norm().item()
                                     for n, p in transformer.named_parameters() if "lora_" in n]
                    last_lora_norm = float(np.mean(lora_w_norms))
                elapsed = time.time() - t0
                total_steps = args.epochs * len(loader)
                eta = elapsed / gstep * (total_steps - gstep) if gstep > 0 else 0
                print(f"  ep{ep} step{step}/{len(loader)} (g{gstep}/{total_steps}) "
                      f"loss={loss.item():.4f} gnorm={gnorm:.3f} lora_norm={last_lora_norm:.4f} "
                      f"el={elapsed:.0f}s eta={eta:.0f}s", flush=True)
                if not np.isfinite(loss.item()):
                    print("  [FATAL] loss is NaN, aborting", flush=True)
                    sys.exit(1)
    print(f"[train done] {time.time()-t0:.0f}s, {gstep} steps", flush=True)
    transformer.save_pretrained(cfg["lora_dir"])
    print(f"  LoRA weights → {cfg['lora_dir']}", flush=True)

    # ---- encode probe h5 ----
    print(f"[encode] {cfg['probe_data']}", flush=True)
    transformer.eval()
    feats_local = {}
    def hook(_m, _in, out):
        feats_local["last"] = out if isinstance(out, torch.Tensor) else out[0]
    base = transformer.base_model.model.transformer_blocks
    h = base[-1].register_forward_hook(hook)

    with h5py.File(cfg["probe_data"], "r") as f:
        N = f["pixels"].shape[0]
        embs = np.empty((N, 1152), dtype=np.float32)
        t_enc = time.time()
        with torch.no_grad():
            for i in range(0, N, args.batch_size):
                pix = f["pixels"][i:i + args.batch_size]
                xchw = torch.from_numpy(np.transpose(pix, (0, 3, 1, 2))).to(device).to(torch.float16) / 127.5 - 1.0
                if xchw.shape[-1] != args.img_size:
                    xchw = F.interpolate(xchw, size=args.img_size, mode="bilinear", align_corners=False)
                latent = vae.encode(xchw).latent_dist.mean * vae.config.scaling_factor
                B = latent.shape[0]
                tt = torch.zeros(B, dtype=torch.long, device=device)
                cls = torch.full((B,), 1000, dtype=torch.long, device=device)
                transformer(latent, timestep=tt, class_labels=cls, return_dict=False)
                embs[i:i + B] = feats_local["last"].mean(dim=1).float().cpu().numpy()
                if (i // args.batch_size) % 100 == 0:
                    print(f"  encoded {i + B}/{N}", flush=True)
    h.remove()
    np.save(cfg["emb_out"], embs)
    print(f"[done] {cfg['emb_out']}  shape={embs.shape}  encode={time.time()-t_enc:.0f}s", flush=True)


if __name__ == "__main__":
    main()
