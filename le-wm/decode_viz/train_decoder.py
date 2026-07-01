"""Train a pixel decoder on a FROZEN LeWM encoder and visualize reconstructions.

Pipeline (matches the session's "Plan A" — repo has no decoder, paper App.D only):
    real frame --[frozen encoder]--> latent (192-d CLS) --[decoder]--> reconstructed frame

Because the encoder is frozen, every frame's embedding is constant, so we
precompute all embeddings once and then train ONLY the decoder. The
reconstruction is a direct visual read-out of what the embedding encodes:
if the ball reappears at the right place, the latent carries position.

Default encoder = un-finetuned pusht weights.pt (`lewm_paper_pusht/weights.pt`).

Usage (from repo root, with le-wm/.venv):
  le-wm/.venv/bin/python le-wm/decode_viz/train_decoder.py \
      --domain uniform_motion --epochs 40 --device cuda \
      --out /data1/likun-share/junjxu/runs/decoder_viz/uniform_pusht
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from decoder import LatentDecoder, build_frozen_encoder, IMAGENET_MEAN, IMAGENET_STD

DATA_DIR = os.environ.get(
    "PHYWORLD_DATA_DIR",
    "/data1/likun-share/junjxu/.stable_worldmodel/datasets",
)
DEFAULT_CKPT = os.environ.get(
    "LEWM_PUSHT_CKPT",
    "/data1/likun-share/junjxu/.stable_worldmodel/lewm_paper_pusht/weights.pt",
)


def load_pixels(domain, max_frames=None):
    """Return pixels as a (N, 3, 224, 224) uint8 torch tensor (on CPU)."""
    path = f"{DATA_DIR}/phyworld_{domain}_id1k.h5"
    with h5py.File(path, "r") as f:
        N = f["pixels"].shape[0]
        if max_frames:
            N = min(N, max_frames)
        pix = f["pixels"][:N]  # (N, H, W, 3) uint8
    pix = torch.from_numpy(pix).permute(0, 3, 1, 2).contiguous()  # (N,3,H,W)
    return pix


@torch.no_grad()
def precompute_embeddings(encode_fn, pixels, batch_size, device):
    embs = []
    N = pixels.shape[0]
    for i in range(0, N, batch_size):
        embs.append(encode_fn(pixels[i:i + batch_size]).cpu())
        if (i // batch_size) % 20 == 0:
            print(f"  embed {i + min(batch_size, N - i)}/{N}", flush=True)
    return torch.cat(embs, 0)  # (N, D)


def save_grid(pixels_uint8, recon01, idxs, path, n=8):
    """Top row = ground-truth frames, bottom row = decoder reconstructions."""
    n = min(n, len(idxs))
    gt = pixels_uint8[idxs[:n]].float() / 255.0          # (n,3,H,W)
    rc = recon01[:n].clamp(0, 1).cpu()                   # (n,3,H,W)
    rows = torch.cat([gt, rc], dim=0)                    # (2n,3,H,W)
    grid = rows.permute(0, 2, 3, 1).numpy()              # (2n,H,W,3)
    H, W = grid.shape[1:3]
    canvas = np.ones((2 * H + 6, n * W + (n - 1) * 4, 3), dtype=np.float32)
    for r in range(2):
        for c in range(n):
            img = grid[r * n + c]
            y = r * (H + 6)
            x = c * (W + 4)
            canvas[y:y + H, x:x + W] = img
    Image.fromarray((canvas * 255).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="uniform_motion",
                    choices=["uniform_motion", "parabola", "collision"])
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="encoder ckpt (default = un-finetuned pusht weights.pt)")
    ap.add_argument("--emb-source", default="cls", choices=["cls", "proj"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output dir for grids + decoder weights")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"[1/4] build frozen encoder from {args.ckpt} (emb_source={args.emb_source})", flush=True)
    encode_fn, emb_dim = build_frozen_encoder(args.ckpt, device=device, emb_source=args.emb_source)
    print(f"      emb_dim = {emb_dim}", flush=True)

    print(f"[2/4] load pixels for {args.domain}", flush=True)
    pixels = load_pixels(args.domain, args.max_frames)  # (N,3,H,W) uint8 CPU
    N = pixels.shape[0]
    print(f"      {N} frames, {tuple(pixels.shape[1:])}", flush=True)

    print(f"[3/4] precompute embeddings (frozen encoder)", flush=True)
    t0 = time.time()
    emb = precompute_embeddings(encode_fn, pixels, batch_size=256, device=device)  # (N,D) CPU
    print(f"      done in {time.time()-t0:.1f}s, emb {tuple(emb.shape)}", flush=True)

    # train / val split
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    n_val = int(N * args.val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    # fixed val samples for the comparison grid (spread across the set)
    grid_idx = val_idx[torch.linspace(0, n_val - 1, 8).long()].tolist()

    # decoder + input standardization from TRAIN embeddings
    decoder = LatentDecoder(emb_dim=emb_dim).to(device)
    decoder.set_norm(emb[train_idx].mean(0), emb[train_idx].std(0))
    emb = emb.to(device)  # (N,D) — tiny, keep on GPU
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)  # unused (kept for ref)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    print(f"[4/4] train decoder: {len(train_idx)} train / {n_val} val, "
          f"{args.epochs} epochs", flush=True)

    def run_val():
        decoder.eval()
        tot, nb = 0.0, 0
        with torch.no_grad():
            for i in range(0, n_val, args.batch_size):
                bidx = val_idx[i:i + args.batch_size]
                z = emb[bidx]
                tgt = pixels[bidx].to(device).float() / 255.0
                pred = decoder(z)
                tot += F.mse_loss(pred, tgt).item()
                nb += 1
        return tot / max(1, nb)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        decoder.train()
        ep_perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        run_loss, nb = 0.0, 0
        for i in range(0, len(ep_perm), args.batch_size):
            bidx = ep_perm[i:i + args.batch_size]
            z = emb[bidx]
            tgt = pixels[bidx].to(device).float() / 255.0  # (B,3,H,W) in [0,1]
            pred = decoder(z)
            loss = F.l1_loss(pred, tgt) + F.mse_loss(pred, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item()
            nb += 1
        sched.step()
        vmse = run_val()
        psnr = -10.0 * np.log10(max(vmse, 1e-10))
        print(f"  ep {ep:3d} | train {run_loss/nb:.4f} | val_mse {vmse:.5f} "
              f"| val_psnr {psnr:.2f} dB | lr {sched.get_last_lr()[0]:.2e}", flush=True)

        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            decoder.eval()
            with torch.no_grad():
                rec = decoder(emb[torch.tensor(grid_idx, device=device)])
            save_grid(pixels, rec, grid_idx, out / f"recon_ep{ep:03d}.png")
        if vmse < best:
            best = vmse
            torch.save({"decoder": decoder.state_dict(), "emb_dim": emb_dim,
                        "args": vars(args)}, out / "decoder_best.pt")

    # final grid copy
    decoder.eval()
    with torch.no_grad():
        rec = decoder(emb[torch.tensor(grid_idx, device=device)])
    save_grid(pixels, rec, grid_idx, out / "recon_final.png")
    print(f"[done] best val_mse={best:.5f} ({-10*np.log10(max(best,1e-10)):.2f} dB). "
          f"Grids + weights in {out}", flush=True)


if __name__ == "__main__":
    main()
