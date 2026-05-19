"""Re-probe DiT LoRA trajectory checkpoints (epoch_2/4/6/8) — fix for the
NaN bug in dit_lora_finetune_probe.py.

Bug: dit_lora_finetune_probe.py used torch.load + load_state_dict on a
PEFT-wrapped model, which silently fails on key mismatch and leaves LoRA in
random-init state; combined with fp16 inference, this produced all-NaN
embeddings on the first checkpoint and crashed the probing loop.

Fix:
  1. Use PeftModel.from_pretrained(base, adapter_dir) for canonical LoRA load
  2. Optionally merge LoRA into base weights and run inference in fp32 for
     numerical stability (LoRA adapters were trained in fp32 on fp16 base,
     so merge_and_unload + fp32 forward avoids precision issues)
  3. Save embeddings, NaN-check, run probe_all_targets via subprocess
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DiTTransformer2DModel
from peft import PeftModel


@torch.no_grad()
def encode(transformer, vae, args, device, n_frames, dtype):
    transformer.eval()
    feats = {}
    def hook(_m, _in, out):
        feats["last"] = out if isinstance(out, torch.Tensor) else out[0]
    # After merge_and_unload, model is no longer PEFT-wrapped — direct access
    base_blocks = transformer.transformer_blocks
    h_handle = base_blocks[-1].register_forward_hook(hook)
    embs = np.empty((n_frames, 1152), dtype=np.float32)

    t_enc = time.time()
    with h5py.File(args.data, "r") as f:
        for i in range(0, n_frames, args.batch_size):
            pix_u8 = f["pixels"][i:i + args.batch_size]
            pix_chw = torch.from_numpy(np.transpose(pix_u8, (0, 3, 1, 2)))
            x = pix_chw.to(device, non_blocking=True).to(dtype) / 127.5 - 1.0
            if x.shape[-1] != args.img_size:
                x = F.interpolate(x, size=args.img_size, mode="bilinear", align_corners=False)
            latent = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
            B = latent.shape[0]
            tt = torch.zeros(B, dtype=torch.long, device=device)
            cls = torch.full((B,), 1000, dtype=torch.long, device=device)
            transformer(latent, timestep=tt, class_labels=cls, return_dict=False)
            pooled = feats["last"].mean(dim=1).float().cpu().numpy()
            embs[i:i + B] = pooled
            if (i // args.batch_size) % 100 == 0:
                print(f"    encoded {i + B}/{n_frames}", flush=True)
    h_handle.remove()
    print(f"  encoding done in {time.time()-t_enc:.0f}s", flush=True)
    return embs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit-dir", default="/home/qlib/.stable_worldmodel/dit_xl_2_256")
    ap.add_argument("--save-dir", default="/home/qlib/agent_memory/wm/artifacts/embeddings/dit_xl_lora_collision_traj")
    ap.add_argument("--data", default=os.path.expanduser("~/.stable_worldmodel/phyworld_collision.h5"))
    ap.add_argument("--n-frames", type=int, default=32000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp32",
                    help="fp32 is safer for LoRA-modified DiT; fp16 was the original (NaN-prone) setting.")
    args = ap.parse_args()

    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    device = args.device

    out_emb_dir = "/home/qlib/agent_memory/wm/artifacts/embeddings"
    out_logs_dir = "/home/qlib/agent_memory/wm/artifacts/logs"

    print(f"[load] base DiT + VAE in {args.precision}", flush=True)
    vae = AutoencoderKL.from_pretrained(args.dit_dir, subfolder="vae", torch_dtype=dtype).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    for ep in args.epochs:
        ck_dir = os.path.join(args.save_dir, f"epoch_{ep}")
        if not os.path.isdir(ck_dir):
            print(f"[skip] no ckpt at {ck_dir}")
            continue

        print(f"\n=== epoch {ep}: re-encode + probe ===", flush=True)
        # Fresh base transformer each time to avoid state leakage
        base = DiTTransformer2DModel.from_pretrained(args.dit_dir, subfolder="transformer", torch_dtype=dtype).to(device)
        # Canonical PEFT load
        peft_model = PeftModel.from_pretrained(base, ck_dir)
        # Merge LoRA into base weights → unwrap to plain DiT (much faster + no PEFT plumbing during inference)
        merged = peft_model.merge_and_unload()
        merged.eval()
        merged.to(device).to(dtype)

        # Sanity: check first layer's q weight differs from raw base
        with torch.no_grad():
            sample_w = merged.transformer_blocks[-1].attn1.to_q.weight
            print(f"  q-weight checksum: mean={sample_w.mean().item():.5f}, std={sample_w.std().item():.5f}", flush=True)

        embs = encode(merged, vae, args, device, args.n_frames, dtype)

        # NaN/Inf check
        n_nan = np.isnan(embs).any(axis=1).sum()
        n_inf = np.isinf(embs).any(axis=1).sum()
        print(f"  emb NaN rows: {n_nan}, Inf rows: {n_inf}", flush=True)
        if n_nan > 0 or n_inf > 0:
            print(f"  WARNING: emb still has NaN/Inf; saving anyway for debug", flush=True)

        emb_path = os.path.join(out_emb_dir, f"dit_xl_lora_collision_emb_32k_epoch{ep}.npy")
        np.save(emb_path, embs)
        print(f"  saved → {emb_path}", flush=True)

        # Run probe via subprocess (so its CPU work doesn't compete with next encoding)
        log_path = os.path.join(out_logs_dir, f"probe_all_targets_dit_lora_epoch{ep}.log")
        print(f"  probing → {log_path}", flush=True)
        # Run with subprocess so any internal warning doesn't kill main loop
        probe_script = "/home/qlib/agent_memory/wm/phyworld/scripts/probe_all_targets.py"
        cmd = [sys.executable, probe_script, "--emb", emb_path]
        with open(log_path, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        print(f"  probe done.", flush=True)

        # Free GPU memory before next epoch
        del base, peft_model, merged
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
