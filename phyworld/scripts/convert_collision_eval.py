"""Convert phyworld collision_eval.hdf5 -> le-wm h5, **keeping per-traj OOD partition labels**.

Partitions follow the phyworld paper:
    ID range     : r ∈ [0.7, 1.5], v ∈ [1, 4]
    OOD range    : r ∈ [0.3, 0.6] ∪ [1.5, 2.0], v ∈ [0, 0.8] ∪ [4.5, 6.0]

For each traj we tag a partition label:
    0 = ID
    1 = r-OOD only
    2 = v-OOD only
    3 = both-OOD
plus per-frame partition labels stored as `partition` dataset.

Reuses the same kinematics + collision_event detection logic as
convert_collision_to_lewm.py so the resulting file is directly probe-compatible.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import imageio.v3 as iio
from PIL import Image

# share helpers with main conversion
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from convert_collision_to_lewm import (  # type: ignore
    decode_mp4,
    resize_frames,
    compute_per_ball_kinematics,
    detect_collision,
)


def partition_label(r1: float, r2: float, v1: float, v2: float,
                    r_id=(0.7, 1.5), v_id=(1.0, 4.0)) -> int:
    r_ok = (r_id[0] <= r1 <= r_id[1]) and (r_id[0] <= r2 <= r_id[1])
    v_ok = (v_id[0] <= abs(v1) <= v_id[1]) and (v_id[0] <= abs(v2) <= v_id[1])
    if r_ok and v_ok:
        return 0
    if not r_ok and v_ok:
        return 1
    if r_ok and not v_ok:
        return 2
    return 3


PART_NAMES = {0: "ID", 1: "r-OOD", 2: "v-OOD", 3: "both-OOD"}


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_src = str(repo_root / "data" / "collision_eval.hdf5")

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=default_src)
    ap.add_argument("--name", default="phyworld_collision_eval")
    ap.add_argument("--dst", default=None)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--limit", type=int, default=0)
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

        first_blob = bytes(f[f"video_streams/{pairs[0][0]}"][pairs[0][1]])
        first_frames = decode_mp4(first_blob)
        T, H0, W0, _ = first_frames.shape
        S = args.img_size
        N = n_traj * T
        print(f"trajectories: {n_traj} | frames/traj: {T} | native {H0}x{W0} -> {S}x{S}")
        print(f"total frames N = {N}")

        os.makedirs(os.path.dirname(args.dst), exist_ok=True)
        with h5py.File(args.dst, "w") as out:
            d_pixels  = out.create_dataset("pixels",  shape=(N, S, S, 3), dtype="uint8",
                                           chunks=(min(64, T), S, S, 3),
                                           compression="gzip", compression_opts=4)
            d_action  = out.create_dataset("action",  shape=(N, 4), dtype="float32")
            d_proprio = out.create_dataset("proprio", shape=(N, 4), dtype="float32")
            d_state   = out.create_dataset("state",   shape=(N, 4), dtype="float32")
            d_mass    = out.create_dataset("mass",    shape=(N, 2), dtype="float32")
            d_coll    = out.create_dataset("collision_event", shape=(N,), dtype="uint8")
            d_ep_len  = out.create_dataset("ep_len",      shape=(n_traj,), dtype="int32")
            d_ep_off  = out.create_dataset("ep_offset",   shape=(n_traj,), dtype="int64")
            d_ep_idx  = out.create_dataset("episode_idx", shape=(N,),     dtype="int64")
            d_step    = out.create_dataset("step_idx",    shape=(N,),     dtype="int64")
            # OOD-specific datasets
            d_part_t  = out.create_dataset("partition_traj",  shape=(n_traj,), dtype="uint8")
            d_part_f  = out.create_dataset("partition",       shape=(N,),     dtype="uint8")
            d_init    = out.create_dataset("init",            shape=(n_traj, 4), dtype="float32")

            offset = 0
            counts = {0:0, 1:0, 2:0, 3:0}
            for ep_i, (g, ti) in enumerate(pairs):
                blob = bytes(f[f"video_streams/{g}"][ti])
                frames = decode_mp4(blob) if ep_i > 0 else first_frames
                if frames.shape[1] != S or frames.shape[2] != S:
                    frames = resize_frames(frames, S)
                pos2 = np.asarray(f[f"position_streams/{g}"][ti], dtype=np.float32)
                init = np.asarray(f[f"init_streams/{g}"][ti],     dtype=np.float32)
                r1, r2, v1, v2 = float(init[0]), float(init[1]), float(init[2]), float(init[3])
                part = partition_label(r1, r2, v1, v2)
                counts[part] += 1
                m1 = r1 ** 2
                m2 = r2 ** 2

                T_i = frames.shape[0]
                vel, acc = compute_per_ball_kinematics(pos2)
                coll = detect_collision(acc)
                end = offset + T_i

                d_pixels[offset:end]  = frames
                d_action[offset:end]  = acc.reshape(T_i, 4)
                d_proprio[offset:end] = pos2.reshape(T_i, 4)
                d_state[offset:end]   = vel.reshape(T_i, 4)
                d_mass[offset:end]    = np.broadcast_to(np.array([m1, m2], np.float32), (T_i, 2))
                d_coll[offset:end]    = coll
                d_ep_idx[offset:end]  = ep_i
                d_step[offset:end]    = np.arange(T_i)
                d_part_f[offset:end]  = part
                d_part_t[ep_i]        = part
                d_init[ep_i]          = init[:4]
                d_ep_len[ep_i]        = T_i
                d_ep_off[ep_i]        = offset
                offset = end

                if (ep_i + 1) % 200 == 0:
                    print(f"  {ep_i + 1}/{n_traj}  partition counts: {counts}", flush=True)

    print(f"\nDone in {time.time()-t0:.1f}s -> {args.dst}")
    print(f"Trajectories by partition:")
    for k, v in counts.items():
        print(f"  {PART_NAMES[k]:10s} {v}")


if __name__ == "__main__":
    main()
