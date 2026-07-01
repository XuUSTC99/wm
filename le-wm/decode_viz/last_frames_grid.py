"""Decode the LAST n consecutive frames of a single trajectory.

The existing recon grids (uniform_pusht_ep*.png) show 8 *different* validation
samples spread across the set. This script instead shows one trajectory's last
n frames in temporal order, so you can see whether the decoder still places the
ball correctly at the END of its path (where the ball is at its most extreme
position). Top row = ground truth, bottom row = decoder reconstruction.

Reuses the already-trained decoder (no retraining).
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
from eval_decoder_ood import save_grid  # noqa: E402

DATA = "/data1/likun-share/junjxu/.stable_worldmodel/datasets"
PUSHT = "/data1/likun-share/junjxu/.stable_worldmodel/lewm_paper_pusht/weights.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="uniform_motion")
    ap.add_argument("--ckpt", default=PUSHT)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--emb-source", default="cls", choices=["cls", "proj"])
    ap.add_argument("--episode", type=int, default=-1,
                    help="-1 = auto-pick the trajectory with the largest horizontal travel")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    encode_fn, emb_dim = build_frozen_encoder(args.ckpt, device=device, emb_source=args.emb_source)
    dec_blob = torch.load(args.decoder, map_location="cpu", weights_only=False)
    dec = LatentDecoder(emb_dim=dec_blob["emb_dim"]).to(device)
    dec.load_state_dict(dec_blob["decoder"])  # includes emb_mean/emb_std buffers
    dec.eval()

    with h5py.File(f"{DATA}/phyworld_{args.domain}_id1k.h5", "r") as f:
        ep_idx = f["episode_idx"][:]
        step_idx = f["step_idx"][:]
        proprio = f["proprio"][:]          # (N, 2) ball position
        def ball_visible(frame_uint8):
            r = ((frame_uint8[..., 0].astype(int) > 120)
                 & (frame_uint8[..., 1].astype(int) < 100)
                 & (frame_uint8[..., 2].astype(int) < 100))
            return int(r.sum()) > 50

        # choose episode
        if args.episode < 0:
            # max horizontal travel, but require the ball to stay IN FRAME for all
            # n last frames (else it just flies off-screen and the grid is blank).
            eps = np.unique(ep_idx)
            order = sorted(
                eps,
                key=lambda e: -abs(proprio[np.where(ep_idx == e)[0][-1], 0]
                                   - proprio[np.where(ep_idx == e)[0][0], 0]))
            ep = None
            for e in order:
                rows = np.where(ep_idx == e)[0]
                rows = rows[np.argsort(step_idx[rows])][-args.n:]
                if all(ball_visible(f["pixels"][int(i)]) for i in rows):
                    ep = int(e)
                    break
            print(f"[auto] episode {ep}: largest travel with ball in-frame for all "
                  f"{args.n} last frames")
        else:
            ep = args.episode
        rows = np.where(ep_idx == ep)[0]
        rows = rows[np.argsort(step_idx[rows])]
        sel = rows[-args.n:]                # LAST n frames, temporal order
        print(f"[episode {ep}] steps {step_idx[sel].tolist()}  "
              f"ball x: {proprio[sel[0], 0]:.2f} -> {proprio[sel[-1], 0]:.2f}")
        pix = f["pixels"][sel.tolist()]    # (n, H, W, 3) uint8

    pix_chw = torch.from_numpy(pix).permute(0, 3, 1, 2).contiguous()  # (n,3,H,W) uint8
    with torch.no_grad():
        emb = encode_fn(pix_chw.to(device))
        rec = dec(emb).clamp(0, 1).cpu()
    gt = pix_chw.float().div(255.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_grid(gt, rec, str(out), n=args.n)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
