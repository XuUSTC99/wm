"""Regenerate per-partition reconstruction grids from an already-trained
universal decoder (no retraining). Reproduces the same seed=0 trajectory split
so the grids show held-out test frames, consistent with the reported PSNR.

Usage:
  python regen_grids.py --ckpt <encoder.ckpt> --decoder <udecoder_TAG.pt> \
    --tag posonly --out <dir> --n 8
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from decoder import LatentDecoder, build_frozen_encoder  # noqa: E402
from eval_decoder_ood import EVAL, NAMES, frame_partitions, save_grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="uniform_motion", choices=list(EVAL))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--emb-source", default="cls", choices=["cls", "proj"])
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = EVAL[args.domain]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    encode_fn, emb_dim = build_frozen_encoder(args.ckpt, device=device, emb_source=args.emb_source)
    ep_parts = frame_partitions(cfg)
    with h5py.File(cfg["eval_h5"], "r") as f:
        pixels = f["pixels"][:]
        ep_idx = f["episode_idx"][:]
    N = len(pixels)
    parts_frame = ep_parts[ep_idx]

    # reproduce the seed=0 stratified trajectory split
    rng = np.random.default_rng(args.seed)
    uniq_eps = np.unique(ep_idx)
    ep_part_of = {int(e): int(ep_parts[e]) for e in uniq_eps}
    test_eps = set()
    for p in sorted(NAMES):
        eps_p = [e for e in uniq_eps if ep_part_of[int(e)] == p]
        rng.shuffle(eps_p)
        n_test = int(round(len(eps_p) * args.test_frac))
        test_eps.update(int(e) for e in eps_p[:n_test])
    is_test_frame = np.array([int(e) in test_eps for e in ep_idx])

    pix_t = torch.from_numpy(pixels).permute(0, 3, 1, 2).contiguous()

    dec = LatentDecoder(emb_dim=emb_dim).to(device)
    sd = torch.load(args.decoder, map_location=device)
    dec.load_state_dict(sd["decoder"])
    dec.eval()

    with torch.no_grad():
        for p in sorted(NAMES):
            mask = (parts_frame == p) & is_test_frame
            idx = np.nonzero(mask)[0]
            if len(idx) < args.n:
                continue
            sel = idx[np.linspace(0, len(idx) - 1, args.n).astype(int)]
            chunk = pix_t[torch.from_numpy(sel)].to(device)
            z = encode_fn(chunk).float()
            pred = dec(z).clamp(0, 1).cpu()
            gt = (pix_t[torch.from_numpy(sel)].float() / 255.0).cpu()
            fn = out / f"udec_{args.tag}_{NAMES[p].replace('/', '')}.png"
            save_grid(gt, pred, str(fn))
            print(f"[saved] {fn}  ({len(idx)} held-out frames in {NAMES[p]})", flush=True)


if __name__ == "__main__":
    main()
