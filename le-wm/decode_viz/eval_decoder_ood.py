"""Evaluate a trained pixel decoder on the OOD eval set, per partition.

Loads a frozen encoder + a decoder trained on ID (id1k), then measures how well
the latent still reconstructs frames on the held-out OOD partitions:
    ID  /  r/m-OOD (unseen position)  /  v-OOD (unseen speed)  /  both-OOD

This answers: does finetuning the encoder improve the latent's
information-preservation on OOD frames it never trained on?

Reports per-partition reconstruction PSNR and saves a comparison grid per
partition (top = ground truth, bottom = decoder reconstruction).

Usage:
  le-wm/.venv/bin/python le-wm/decode_viz/eval_decoder_ood.py \
      --domain uniform_motion \
      --ckpt <encoder ckpt> --decoder <decoder_best.pt> \
      --tag finetuned --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from decoder import LatentDecoder, build_frozen_encoder

DATA_DIR = os.environ.get(
    "PHYWORLD_DATA_DIR",
    "/data1/likun-share/junjxu/.stable_worldmodel/datasets",
)
REPO = Path(__file__).resolve().parent.parent.parent

# eval h5 + source hdf5 (for init_streams → partition), per domain
EVAL = {
    "uniform_motion": dict(eval_h5=f"{DATA_DIR}/phyworld_uniform_motion.h5",
                           src=str(REPO / "phyworld/data/uniform_motion_eval.hdf5"), ncol=2),
    "parabola": dict(eval_h5=f"{DATA_DIR}/phyworld_parabola.h5",
                     src=str(REPO / "phyworld/data/parabola_eval.hdf5"), ncol=2),
    "collision": dict(eval_h5=f"{DATA_DIR}/phyworld_collision_eval.h5",
                      src=str(REPO / "phyworld/data/collision_eval.hdf5"), ncol=4),
}
NAMES = {0: "ID", 1: "r/m-OOD", 2: "v-OOD", 3: "both-OOD"}


def pl_2col(r, v):
    r_ok = 0.7 <= abs(r) <= 1.5
    v_ok = 1.0 <= abs(v) <= 4.0
    if r_ok and v_ok: return 0
    if not r_ok and v_ok: return 1
    if r_ok and not v_ok: return 2
    return 3


def pl_4col(m0, m1, v0, v1):
    m_ok = (0.7 <= m0 <= 1.5) and (0.7 <= m1 <= 1.5)
    v_ok = (1.0 <= abs(v0) <= 4.0) and (1.0 <= abs(v1) <= 4.0)
    if m_ok and v_ok: return 0
    if not m_ok and v_ok: return 1
    if m_ok and not v_ok: return 2
    return 3


def frame_partitions(cfg):
    with h5py.File(cfg["src"], "r") as f:
        init = np.concatenate([f["init_streams"][k][...] for k in sorted(f["init_streams"])], 0)
    if cfg["ncol"] == 4:
        ep_parts = np.array([pl_4col(*init[i]) for i in range(len(init))], np.uint8)
    else:
        ep_parts = np.array([pl_2col(float(init[i, 0]), float(init[i, 1])) for i in range(len(init))], np.uint8)
    return ep_parts


def save_grid(gt01, rc01, path, n=8):
    n = min(n, gt01.shape[0])
    gt = gt01[:n]; rc = rc01[:n].clamp(0, 1)
    rows = torch.cat([gt, rc], 0).permute(0, 2, 3, 1).cpu().numpy()
    H, W = rows.shape[1:3]
    canvas = np.ones((2 * H + 6, n * W + (n - 1) * 4, 3), np.float32)
    for r in range(2):
        for c in range(n):
            canvas[r * (H + 6):r * (H + 6) + H, c * (W + 4):c * (W + 4) + W] = rows[r * n + c]
    Image.fromarray((canvas * 255).astype(np.uint8)).save(path)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="uniform_motion", choices=list(EVAL))
    ap.add_argument("--ckpt", required=True, help="encoder ckpt (state_dict or JEPA object)")
    ap.add_argument("--decoder", required=True, help="trained decoder_best.pt")
    ap.add_argument("--emb-source", default="cls", choices=["cls", "proj"])
    ap.add_argument("--max-trajs", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = EVAL[args.domain]
    device = args.device

    encode_fn, emb_dim = build_frozen_encoder(args.ckpt, device=device, emb_source=args.emb_source)
    dec = LatentDecoder(emb_dim=emb_dim).to(device)
    sd = torch.load(args.decoder, map_location="cpu")
    dec.load_state_dict(sd["decoder"])
    dec.eval()

    ep_parts = frame_partitions(cfg)
    with h5py.File(cfg["eval_h5"], "r") as f:
        ep_idx = f["episode_idx"][:]
        # cap by trajectories (deterministic) for speed
        uniq = np.unique(ep_idx)
        rng = np.random.default_rng(0)
        sel = set(rng.permutation(uniq)[:args.max_trajs].tolist()) if args.max_trajs else set(uniq.tolist())
        mask = np.array([e in sel for e in ep_idx])
        rows = np.where(mask)[0]
        pix = f["pixels"][:]  # (M,224,224,3) uint8 — load once
    parts = ep_parts[ep_idx[rows]]                       # partition per selected frame
    pix = torch.from_numpy(pix[rows]).permute(0, 3, 1, 2).contiguous()  # (n,3,H,W) uint8

    # per-partition PSNR + grids
    results = {}
    for p in range(4):
        pidx = np.where(parts == p)[0]
        if len(pidx) == 0:
            results[NAMES[p]] = None
            continue
        # aggregate MSE over this partition (batched)
        tot_se, tot_px = 0.0, 0
        grid_gt, grid_rc = None, None
        for i in range(0, len(pidx), 256):
            b = pidx[i:i + 256]
            x = pix[b].to(device)
            z = encode_fn(x)
            rc = dec(z)                                  # (B,3,H,W) in [0,1]
            gt = x.float() / 255.0
            tot_se += F.mse_loss(rc, gt, reduction="sum").item()
            tot_px += gt.numel()
            if grid_gt is None:
                k = min(8, len(b))
                sel8 = np.linspace(0, len(pidx) - 1, k).astype(int)
                gb = pidx[sel8]
                xg = pix[gb].to(device)
                grid_gt = xg.float() / 255.0
                grid_rc = dec(encode_fn(xg))
        mse = tot_se / tot_px
        psnr = -10.0 * np.log10(max(mse, 1e-10))
        results[NAMES[p]] = dict(n=int(len(pidx)), mse=mse, psnr=psnr)
        tagn = f"{args.tag}_" if args.tag else ""
        save_grid(grid_gt, grid_rc, out / f"{tagn}{args.domain}_p{p}_{NAMES[p].replace('/','')}.png")

    # report
    print(f"\n=== OOD recon PSNR | {args.domain} | tag={args.tag} | enc={Path(args.ckpt).name} ===")
    print(f"{'partition':10s} {'n':>6s} {'PSNR(dB)':>10s} {'MSE':>10s}")
    for p in range(4):
        r = results[NAMES[p]]
        if r is None:
            print(f"{NAMES[p]:10s} {'-':>6s}")
        else:
            print(f"{NAMES[p]:10s} {r['n']:6d} {r['psnr']:10.2f} {r['mse']:10.5f}")
    with open(out / f"psnr_{args.tag or 'eval'}.json", "w") as fp:
        json.dump(dict(domain=args.domain, tag=args.tag, ckpt=args.ckpt,
                       decoder=args.decoder, results=results), fp, indent=2)
    print(f"[saved] grids + psnr_{args.tag or 'eval'}.json in {out}")


if __name__ == "__main__":
    main()
