"""Convert phyworld uniform_motion_eval.hdf5 -> le-wm pusht-style h5.

What this script does
---------------------
phyworld stores each trajectory as an MP4 byte-blob inside hdf5 groups
(`video_streams/<group_id>/<traj_idx>`), plus 2D positions in
`position_streams/...`. It has *no* action signal — it's a passive physics
dataset.

le-wm expects a single flat-stacked hdf5 with these keys
(see `wm/le-wm/.venv/.../stable_worldmodel/data/dataset.py`):
    pixels        (N, H, W, 3) uint8     — every frame, all episodes concatenated
    action        (N, A)        float32  — per-frame action (mandatory)
    proprio       (N, P)        float32  — per-frame proprioception
    ep_len        (E,)          int32    — frames per episode
    ep_offset     (E,)          int64    — start row in N for each episode
    episode_idx   (N,)          int64
    step_idx      (N,)          int64

This script bridges the two formats:
- decodes each MP4 with imageio, resizes to a square `--img-size`,
- writes the position array directly as `proprio`,
- synthesises `action[t] = position[t+1] - position[t]` (velocity), so
  the column has non-zero variance (le-wm's normalizer divides by std).
- builds the episode bookkeeping arrays.

Output goes to `$STABLEWM_HOME/<--name>.h5` (default
`~/.stable_worldmodel/phyworld_uniform_motion.h5`), which is where
`stable_worldmodel.HDF5Dataset(name=...)` looks for it.

Run from this directory (relative `--src` defaults assume that):
    python scripts/convert_to_lewm.py
"""
import argparse, os, tempfile, time
from pathlib import Path

import h5py
import numpy as np
import imageio.v3 as iio
from PIL import Image


def decode_mp4(blob: bytes) -> np.ndarray:
    """bytes -> (T, H, W, 3) uint8 via temp-file (imageio handles mp4)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tf.write(blob); path = tf.name
    try:
        return iio.imread(path)  # (T, H, W, 3) uint8
    finally:
        os.unlink(path)


def resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    """(T, H, W, 3) -> (T, size, size, 3) via PIL bilinear."""
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    out = np.empty((frames.shape[0], size, size, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        out[i] = np.asarray(Image.fromarray(f).resize((size, size), Image.BILINEAR))
    return out


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_src = str(repo_root / "data" / "uniform_motion_eval.hdf5")

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=default_src,
                    help=f"phyworld hdf5 (default: {default_src})")
    ap.add_argument("--name", default="phyworld_uniform_motion",
                    help="output dataset name; written to $STABLEWM_HOME/<name>.h5 "
                         "(default: phyworld_uniform_motion). Used unmodified by le-wm "
                         "as `dataset.name` in config/train/data/<task>.yaml")
    ap.add_argument("--dst", default=None,
                    help="explicit output path; overrides --name")
    ap.add_argument("--img-size", type=int, default=224,
                    help="square resize target (default 224 — matches le-wm img_size)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap total trajectories for a quick test (0 = all)")
    ap.add_argument("--action-mode", choices=["constant", "future_velocity"],
                    default="constant", help="constant is the leak-free passive-video protocol; "
                    "future_velocity reproduces the privileged legacy protocol")
    args = ap.parse_args()
    if args.dst is None:
        stablewm_home = os.environ.get("STABLEWM_HOME") or os.path.expanduser("~/.stable_worldmodel")
        args.dst = os.path.join(stablewm_home, f"{args.name}.h5")

    t0 = time.time()
    with h5py.File(args.src, "r") as f:
        groups = sorted(f["video_streams"].keys())
        # collect (group, traj_idx) pairs
        pairs = []
        for g in groups:
            n = f[f"video_streams/{g}"].shape[0]
            pairs.extend([(g, i) for i in range(n)])
        if args.limit:
            pairs = pairs[: args.limit]
        n_traj = len(pairs)
        # peek at first to get T, H, W
        first = decode_mp4(bytes(f[f"video_streams/{pairs[0][0]}"][pairs[0][1]]))
        T, H0, W0, _ = first.shape
        print(f"trajectories: {n_traj} | per-traj frames: {T} | native {H0}x{W0} -> {args.img_size}x{args.img_size}")
        # dimension checks
        for g in groups:
            ps = f[f"position_streams/{g}"].shape
            assert ps[1] == T, f"position T={ps[1]} != video T={T} in {g}"

        N = n_traj * T
        S = args.img_size

        # write
        os.makedirs(os.path.dirname(args.dst), exist_ok=True)
        with h5py.File(args.dst, "w") as out:
            d_pixels = out.create_dataset(
                "pixels", shape=(N, S, S, 3), dtype="uint8",
                chunks=(min(64, T), S, S, 3),
                compression="gzip", compression_opts=4,
            )
            d_action  = out.create_dataset("action",  shape=(N, 2), dtype="float32")
            d_proprio = out.create_dataset("proprio", shape=(N, 2), dtype="float32")
            d_ep_len     = out.create_dataset("ep_len",      shape=(n_traj,), dtype="int32")
            d_ep_offset  = out.create_dataset("ep_offset",   shape=(n_traj,), dtype="int64")
            d_episode_idx= out.create_dataset("episode_idx", shape=(N,), dtype="int64")
            d_step_idx   = out.create_dataset("step_idx",    shape=(N,), dtype="int64")

            offset = 0
            for ep_i, (g, ti) in enumerate(pairs):
                blob = bytes(f[f"video_streams/{g}"][ti])
                frames = decode_mp4(blob) if ep_i > 0 else first  # reuse the peek
                if frames.shape[0] != T:
                    raise ValueError(f"traj {g}/{ti} has T={frames.shape[0]} != {T}")
                if frames.shape[1] != S or frames.shape[2] != S:
                    frames = resize_frames(frames, S)

                pos = np.asarray(f[f"position_streams/{g}"][ti], dtype=np.float32)  # (T, 2)
                # velocity as action: a[t] = pos[t+1] - pos[t], a[T-1] = a[T-2] (no future)
                if args.action_mode == "constant":
                    action = np.zeros_like(pos)
                else:
                    action = np.empty_like(pos)
                    action[:-1] = pos[1:] - pos[:-1]
                    action[-1] = action[-2]

                lo, hi = offset, offset + T
                d_pixels[lo:hi]      = frames
                d_proprio[lo:hi]     = pos
                d_action[lo:hi]      = action
                d_episode_idx[lo:hi] = ep_i
                d_step_idx[lo:hi]    = np.arange(T, dtype=np.int64)
                d_ep_len[ep_i]       = T
                d_ep_offset[ep_i]    = lo
                offset = hi

                if (ep_i + 1) % 100 == 0 or ep_i == n_traj - 1:
                    rate = (ep_i + 1) / max(time.time() - t0, 1e-6)
                    eta  = (n_traj - ep_i - 1) / max(rate, 1e-6)
                    print(f"  [{ep_i+1}/{n_traj}] {rate:.1f} traj/s ETA {eta:.0f}s")

            out.attrs["source"] = args.src
            out.attrs["frames_per_traj"] = T
            out.attrs["img_size"] = S
            out.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        print(f"\ndone in {time.time()-t0:.1f}s -> {args.dst}")
        print(f"  total frames N = {N}")


if __name__ == "__main__":
    main()
