"""Fit the frozen affine state decoder and latent covariance used by GIPP."""
import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "le-wm"))
SWM = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True, help="LeWM-format HDF5")
    ap.add_argument("--output", required=True)
    ap.add_argument("--velocity-key", default="finite_difference",
                    choices=["finite_difference", "action", "state"])
    ap.add_argument("--position-cols", default="0,1")
    ap.add_argument("--velocity-cols", default="0,1")
    ap.add_argument("--max-frames", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--ridge", type=float, default=1.0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(args.ckpt, map_location="cpu", weights_only=False).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # Decoder fitting is action-free by construction.
    if hasattr(model, "use_action"):
        model.use_action = False

    pc = np.array([int(x) for x in args.position_cols.split(",")])
    vc = np.array([int(x) for x in args.velocity_cols.split(",")])
    with h5py.File(args.dataset, "r") as f:
        n = min(len(f["pixels"]), args.max_frames)
        ids = np.linspace(0, len(f["pixels"]) - 1, n, dtype=np.int64)
        fd_velocity = None
        gravity = None
        if args.velocity_key == "finite_difference":
            all_pos = np.asarray(f["proprio"])
            fd_velocity = np.zeros_like(all_pos)
            same_episode = (np.asarray(f["episode_idx"])[1:] == np.asarray(f["episode_idx"])[:-1])
            valid = np.flatnonzero(same_episode)
            fd_velocity[valid] = (all_pos[1:] - all_pos[:-1])[valid]
            # At episode ends, reuse the last observable displacement.
            ends = np.flatnonzero(~same_episode)
            fd_velocity[ends] = fd_velocity[np.maximum(ends - 1, 0)]
            # Robust one-step acceleration estimate without crossing episodes.
            valid_acc = np.flatnonzero(same_episode[:-1] & same_episode[1:])
            gravity = np.median(fd_velocity[valid_acc + 1] - fd_velocity[valid_acc], axis=0)[vc]
        latents, states = [], []
        for lo in range(0, n, args.batch_size):
            take = ids[lo:lo + args.batch_size]
            # h5py requires increasing indices; linspace satisfies that contract.
            px = torch.from_numpy(f["pixels"][take]).permute(0, 3, 1, 2).float() / 255.0
            px = (px - IMNET_MEAN) / IMNET_STD
            with torch.no_grad():
                emb = model.encode({"pixels": px[:, None].to(dev)})["emb"][:, 0]
            latents.append(emb.cpu().numpy())
            pos = np.asarray(f["proprio"][take])[:, pc]
            if fd_velocity is not None:
                vel = fd_velocity[take][:, vc]
            else:
                vel = np.asarray(f[args.velocity_key][take])[:, vc]
            states.append(np.concatenate([pos, vel], axis=1))

    x = np.concatenate(latents).astype(np.float64)
    y = np.concatenate(states).astype(np.float64)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    ridge = Ridge(alpha=args.ridge).fit((x - mu) / sd, y)
    weight = ridge.coef_ / sd[None, :]
    bias = ridge.intercept_ - weight @ mu
    covariance = np.cov(x, rowvar=False)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(weight=weight.astype(np.float32), bias=bias.astype(np.float32),
                   covariance=covariance.astype(np.float32), latent_mean=mu.astype(np.float32),
                   state_scale=(y.std(0) + 1e-6).astype(np.float32),
                   n_samples=np.int64(len(x)), ridge_alpha=np.float32(args.ridge))
    if gravity is not None:
        payload["gravity"] = gravity.astype(np.float32)
    np.savez(out, **payload)
    pred = x @ weight.T + bias
    mse = ((pred - y) ** 2).mean(0)
    print(f"saved {out} | samples={len(x)} latent={x.shape[1]} state={y.shape[1]}")
    print("decoder MSE:", mse)


if __name__ == "__main__":
    main()
