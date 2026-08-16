"""Fit a frozen affine state decoder and covariance used by GIPP."""

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
    ap.add_argument("--history-size", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument(
        "--ridge-solver", default="auto",
        choices=["auto", "cholesky", "lsqr"],
    )
    args = ap.parse_args()
    if args.history_size < 1:
        raise ValueError("history-size must be positive")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(
        args.ckpt, map_location="cpu", weights_only=False).to(dev).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if hasattr(model, "use_action"):
        model.use_action = False

    position_cols = np.array([
        int(value) for value in args.position_cols.split(",")])
    velocity_cols = np.array([
        int(value) for value in args.velocity_cols.split(",")])
    with h5py.File(args.dataset, "r") as handle:
        total_frames = len(handle["pixels"])
        episode = np.asarray(handle["episode_idx"])
        step = np.asarray(handle["step_idx"])
        candidates = np.arange(
            args.history_size - 1, total_frames, dtype=np.int64)
        if args.history_size > 1:
            first = candidates - (args.history_size - 1)
            valid_window = (
                (episode[candidates] == episode[first])
                & (step[candidates] - step[first]
                   == args.history_size - 1)
            )
            candidates = candidates[valid_window]
        n_samples = min(len(candidates), args.max_frames)
        selection = np.linspace(
            0, len(candidates) - 1, n_samples, dtype=np.int64)
        target_ids = candidates[selection]

        finite_difference_velocity = None
        gravity = None
        if args.velocity_key == "finite_difference":
            all_position = np.asarray(handle["proprio"])
            finite_difference_velocity = np.zeros_like(all_position)
            same_episode = episode[1:] == episode[:-1]
            valid_difference = np.flatnonzero(same_episode)
            finite_difference_velocity[valid_difference] = (
                all_position[1:] - all_position[:-1]
            )[valid_difference]
            ends = np.flatnonzero(~same_episode)
            finite_difference_velocity[ends] = finite_difference_velocity[
                np.maximum(ends - 1, 0)]
            valid_acceleration = np.flatnonzero(
                same_episode[:-1] & same_episode[1:])
            gravity = np.median(
                finite_difference_velocity[valid_acceleration + 1]
                - finite_difference_velocity[valid_acceleration],
                axis=0,
            )[velocity_cols]

        required_ids = np.unique(np.concatenate([
            target_ids - offset for offset in range(args.history_size)
        ]))
        encoded = []
        for lower in range(0, len(required_ids), args.batch_size):
            frame_ids = required_ids[lower:lower + args.batch_size]
            pixels = torch.from_numpy(
                handle["pixels"][frame_ids]
            ).permute(0, 3, 1, 2).float() / 255.0
            pixels = (pixels - IMNET_MEAN) / IMNET_STD
            with torch.no_grad():
                embedding = model.encode({
                    "pixels": pixels[:, None].to(dev)
                })["emb"][:, 0]
            encoded.append(embedding.cpu().numpy())
        encoded = np.concatenate(encoded)
        lookup = np.full(total_frames, -1, dtype=np.int64)
        lookup[required_ids] = np.arange(len(required_ids))
        blocks = [
            encoded[lookup[target_ids - offset]]
            for offset in range(args.history_size - 1, -1, -1)
        ]
        features = np.concatenate(blocks, axis=1).astype(np.float64)
        current = blocks[-1].astype(np.float64)
        position = np.asarray(
            handle["proprio"][target_ids])[:, position_cols]
        if finite_difference_velocity is not None:
            velocity = finite_difference_velocity[
                target_ids][:, velocity_cols]
        else:
            velocity = np.asarray(
                handle[args.velocity_key][target_ids])[:, velocity_cols]
        state = np.concatenate([position, velocity], axis=1).astype(
            np.float64)
    feature_mean = features.mean(0)
    feature_scale = features.std(0) + 1e-6
    ridge = Ridge(
        alpha=args.ridge, solver=args.ridge_solver, tol=1e-5,
    ).fit((features - feature_mean) / feature_scale, state)
    full_weight = ridge.coef_ / feature_scale[None, :]
    bias = ridge.intercept_ - full_weight @ feature_mean
    latent_dim = current.shape[1]
    weight = full_weight[:, -latent_dim:]
    history_weight = full_weight[:, :-latent_dim]
    covariance = np.cov(current, rowvar=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weight": weight.astype(np.float32),
        "bias": bias.astype(np.float32),
        "covariance": covariance.astype(np.float32),
        "latent_mean": current.mean(0).astype(np.float32),
        "state_scale": (state.std(0) + 1e-6).astype(np.float32),
        "n_samples": np.int64(len(features)),
        "ridge_alpha": np.float32(args.ridge),
        "temporal_history_size": np.int64(args.history_size),
    }
    if history_weight.shape[1]:
        payload["history_weight"] = history_weight.astype(np.float32)
    if gravity is not None:
        payload["gravity"] = gravity.astype(np.float32)
    np.savez(output, **payload)

    prediction = features @ full_weight.T + bias
    mse = ((prediction - state) ** 2).mean(0)
    print(
        f"saved {output} | samples={len(features)} "
        f"latent={latent_dim} history={args.history_size} "
        f"state={state.shape[1]}"
    )
    print("decoder MSE:", mse)


if __name__ == "__main__":
    main()
