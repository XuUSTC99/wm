"""Convert phyworld collision_30K hdf5 -> le-wm pusht-style h5 with **force-as-action**.

Why force-as-action
-------------------
The previous uniform-motion experiment used `action[t] = pos[t+1] - pos[t]`
(velocity), which the predictor learned as a trivial linear shift -- so the
encoder never had to embed velocity. For collision data we use:

    action[t] = a[t] = (v[t+1] - v[t]) / dt        # per-ball acceleration (4D)

Acceleration is ~0 during free flight and a one-frame impulse at the collision
moment. Reconstructing the next state from `(emb_t, action_t)` now requires the
emb to carry **velocity** (otherwise the predictor cannot integrate
`pos_{t+1} = pos_t + v_t * dt`). We expect this to force the encoder to learn
velocity, which the probe will measure.

Output schema (matches what le-wm.HDF5Dataset expects)
------------------------------------------------------
    pixels      (N, 224, 224, 3) uint8     # both-ball scene per frame
    action      (N, 4)            float32   # (ax1, ay1, ax2, ay2)  -- THIS IS NEW
    proprio     (N, 4)            float32   # (x1, y1, x2, y2)
    state       (N, 4)            float32   # (vx1, vy1, vx2, vy2)  -- for probing
    mass        (N, 2)            float32   # (m1, m2) broadcast from init -- for probing
    collision_event (N,)          uint8     # binary: did collision happen at frame t? -- for probing
    ep_len, ep_offset, episode_idx, step_idx, ...

Why proprio holds positions and state holds velocities: le-wm's normalizer sees
proprio (loaded by HDF5Dataset), and we want it to see something benign. state
is loaded if requested but not fed to the model -- perfect place to stash
velocities for downstream probing.

Run from the phyworld dir:
    python scripts/convert_collision_to_lewm.py --limit 5000
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import imageio.v3 as iio
from PIL import Image


def decode_mp4(blob: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tf.write(blob); path = tf.name
    try:
        return iio.imread(path)  # (T, H, W, 3) uint8
    finally:
        os.unlink(path)


def resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    out = np.empty((frames.shape[0], size, size, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        out[i] = np.asarray(Image.fromarray(f).resize((size, size), Image.BILINEAR))
    return out


def compute_per_ball_kinematics(positions_2balls: np.ndarray, dt: float = 1.0):
    """positions_2balls: (T, 2_balls, 2_xy) -> velocity, acceleration of same shape.

    We use forward differences and pad the last row(s) with the previous value.
    """
    T = positions_2balls.shape[0]
    vel = np.empty_like(positions_2balls)
    vel[:-1] = (positions_2balls[1:] - positions_2balls[:-1]) / dt
    vel[-1] = vel[-2]  # no future; reuse last-known velocity
    acc = np.empty_like(positions_2balls)
    acc[:-1] = (vel[1:] - vel[:-1]) / dt
    acc[-1] = 0.0  # no future; force=0 (no impulse to apply)
    return vel, acc


def detect_collision(acc: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Per-frame binary indicator: was there a collision impulse at time t?

    acc shape: (T, 2_balls, 2_xy). A collision lights up |a| above threshold.
    """
    return (np.linalg.norm(acc, axis=-1).max(axis=-1) > threshold).astype(np.uint8)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_src = str(repo_root / "data" / "collision_30K.hdf5")

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=default_src)
    ap.add_argument("--name", default="phyworld_collision",
                    help="output dataset name (-> $STABLEWM_HOME/<name>.h5)")
    ap.add_argument("--dst", default=None)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--limit", type=int, default=0, help="cap total trajectories (0 = all)")
    ap.add_argument("--action-mode", choices=["constant", "future_acceleration"],
                    default="constant", help="constant is the leak-free passive-video protocol; "
                    "future_acceleration reproduces the privileged legacy protocol")
    ap.add_argument("--mass-from-init", action="store_true", default=True,
                    help="use init_streams cols 0,1 as masses (default: True)")
    args = ap.parse_args()
    if args.dst is None:
        stablewm_home = os.environ.get("STABLEWM_HOME") or os.path.expanduser("~/.stable_worldmodel")
        args.dst = os.path.join(stablewm_home, f"{args.name}.h5")

    t0 = time.time()
    with h5py.File(args.src, "r") as f:
        groups = sorted(f["video_streams"].keys())
        pairs = []
        for g in groups:
            n = f[f"video_streams/{g}"].shape[0]
            pairs.extend([(g, i) for i in range(n)])
        if args.limit:
            pairs = pairs[: args.limit]
        n_traj = len(pairs)

        # peek
        first_blob = bytes(f[f"video_streams/{pairs[0][0]}"][pairs[0][1]])
        first_frames = decode_mp4(first_blob)
        T, H0, W0, _ = first_frames.shape
        S = args.img_size
        N = n_traj * T
        print(f"trajectories: {n_traj} | frames/traj: {T} | native {H0}x{W0} -> {S}x{S}")
        print(f"total frames N = {N}")

        os.makedirs(os.path.dirname(args.dst), exist_ok=True)
        with h5py.File(args.dst, "w") as out:
            d_pixels = out.create_dataset(
                "pixels", shape=(N, S, S, 3), dtype="uint8",
                chunks=(min(64, T), S, S, 3),
                compression="gzip", compression_opts=4,
            )
            d_action  = out.create_dataset("action",  shape=(N, 4), dtype="float32")
            d_proprio = out.create_dataset("proprio", shape=(N, 4), dtype="float32")
            d_state   = out.create_dataset("state",   shape=(N, 4), dtype="float32")
            d_mass    = out.create_dataset("mass",    shape=(N, 2), dtype="float32")
            d_coll    = out.create_dataset("collision_event", shape=(N,), dtype="uint8")
            d_ep_len    = out.create_dataset("ep_len",     shape=(n_traj,), dtype="int32")
            d_ep_offset = out.create_dataset("ep_offset",  shape=(n_traj,), dtype="int64")
            d_episode_idx = out.create_dataset("episode_idx", shape=(N,), dtype="int64")
            d_step_idx    = out.create_dataset("step_idx",    shape=(N,), dtype="int64")

            offset = 0
            n_collision_frames = 0
            for ep_i, (g, ti) in enumerate(pairs):
                blob = bytes(f[f"video_streams/{g}"][ti])
                frames = decode_mp4(blob) if ep_i > 0 else first_frames
                if frames.shape[1] != S or frames.shape[2] != S:
                    frames = resize_frames(frames, S)

                pos2 = np.asarray(f[f"position_streams/{g}"][ti], dtype=np.float32)  # (T, 2, 2)
                init = np.asarray(f[f"init_streams/{g}"][ti], dtype=np.float32)      # (4,)
                # Heuristic from data inspection: init cols 0,1 look like radius/size
                # of ball1 / ball2 (range ~0.5..1.5). We use radius^2 as a proxy for mass
                # (assuming uniform 2D density). This matches what we saw in momentum
                # conservation (m1*v1 + m2*v2 conserved with m=r^2).
                m1 = float(init[0]) ** 2
                m2 = float(init[1]) ** 2

                vel, acc = compute_per_ball_kinematics(pos2)            # (T, 2, 2)
                coll_mask = detect_collision(acc, threshold=0.05)        # (T,)
                n_collision_frames += int(coll_mask.sum())

                lo, hi = offset, offset + T
                d_pixels[lo:hi]      = frames
                # flatten ball x xy: (T, 2, 2) -> (T, 4) as (x1, y1, x2, y2)
                d_proprio[lo:hi]     = pos2.reshape(T, 4)
                d_state[lo:hi]       = vel.reshape(T, 4)
                if args.action_mode == "constant":
                    d_action[lo:hi] = 0.0
                else:
                    d_action[lo:hi] = acc.reshape(T, 4)
                d_mass[lo:hi]        = np.tile(np.array([m1, m2], np.float32), (T, 1))
                d_coll[lo:hi]        = coll_mask
                d_episode_idx[lo:hi] = ep_i
                d_step_idx[lo:hi]    = np.arange(T, dtype=np.int64)
                d_ep_len[ep_i]       = T
                d_ep_offset[ep_i]    = lo
                offset = hi

                if (ep_i + 1) % 200 == 0 or ep_i == n_traj - 1:
                    rate = (ep_i + 1) / max(time.time() - t0, 1e-6)
                    eta  = (n_traj - ep_i - 1) / max(rate, 1e-6)
                    print(f"  [{ep_i+1}/{n_traj}] {rate:.1f} traj/s ETA {eta:.0f}s")

            out.attrs["source"] = args.src
            out.attrs["frames_per_traj"] = T
            out.attrs["img_size"] = S
            out.attrs["n_collision_frames"] = n_collision_frames
            out.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        print(f"\ndone in {time.time()-t0:.1f}s -> {args.dst}")
        print(f"  total frames N = {N}, collision frames = {n_collision_frames} "
              f"({100*n_collision_frames/N:.2f}%)")


if __name__ == "__main__":
    main()
