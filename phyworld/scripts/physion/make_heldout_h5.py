"""Make a held-out-scene training HDF5 from physionpp_readout.h5 (2026-07-08).

Real property-OOD protocol: EXCLUDE whole scenes from training, so rollout on
those scenes is genuine held-out generalization (not "seen-then-partitioned").

Default held-out (within-category OOD — each has a same-property training sibling):
  bouncy_wall      (train sees bouncy_platform)
  deform_clothhang (train sees deform_clothhit)
  mass_waterpush   (train sees mass_collision + mass_dominoes)

Keeps scene_idx numbering identical to the full h5 (so the SAME rollout_eval can
report both held-out ckpt and full-trained ckpt on the same scene ids).
Copies per-episode contiguous blocks -> memory-safe (no whole-pixels load).

Usage: python make_heldout_h5.py [--heldout bouncy_wall,deform_clothhang,mass_waterpush] [--name physionpp_htrain]
"""
import argparse
import os
import time

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="physionpp_readout")
    ap.add_argument("--name", default="physionpp_htrain")
    ap.add_argument("--heldout", default="bouncy_wall,deform_clothhang,mass_waterpush")
    args = ap.parse_args()

    swm = os.environ.get("STABLEWM_HOME", "/data1/likun-share/junjxu/.stable_worldmodel")
    src = os.path.join(swm, "datasets", f"{args.src}.h5")
    dst = os.path.join(swm, "datasets", f"{args.name}.h5")

    f = h5py.File(src, "r")
    names = f.attrs["scene_names"].split(",")
    heldout_names = [s for s in args.heldout.split(",") if s]
    heldout = {names.index(s) for s in heldout_names}
    scene = f["scene_idx"][:]; ep = f["episode_idx"][:]; step = f["step_idx"][:]
    S = f["pixels"].shape[1]

    uniq = np.unique(ep)
    keep = [e for e in uniq if int(scene[ep == e][0]) not in heldout]
    print(f"[heldout={heldout_names} -> ids {sorted(heldout)}]")
    print(f"keep {len(keep)}/{len(uniq)} clips for training; held-out {len(uniq)-len(keep)} clips reserved for OOD eval")

    t0 = time.time()
    with h5py.File(dst, "w") as out:
        d_px = out.create_dataset("pixels", (0, S, S, 3), maxshape=(None, S, S, 3), dtype="uint8",
                                  chunks=(16, S, S, 3), compression="gzip", compression_opts=4)
        d_ac = out.create_dataset("action", (0, 3), maxshape=(None, 3), dtype="float32")
        d_pr = out.create_dataset("proprio", (0, 3), maxshape=(None, 3), dtype="float32")
        d_st = out.create_dataset("state", (0, 3), maxshape=(None, 3), dtype="float32")
        d_ei = out.create_dataset("episode_idx", (0,), maxshape=(None,), dtype="int64")
        d_si = out.create_dataset("step_idx", (0,), maxshape=(None,), dtype="int64")
        d_sc = out.create_dataset("scene_idx", (0,), maxshape=(None,), dtype="int64")
        ep_len, ep_off, off = [], [], 0
        for new_i, e in enumerate(keep):
            rows = np.nonzero(ep == e)[0]; rows = rows[np.argsort(step[rows])]
            assert np.all(np.diff(rows) == 1), f"episode {e} not contiguous"
            r0, r1, T = rows[0], rows[-1] + 1, len(rows)
            n = off + T
            for d in (d_px, d_ac, d_pr, d_st, d_ei, d_si, d_sc):
                d.resize(n, axis=0)
            d_px[off:n] = f["pixels"][r0:r1]
            d_ac[off:n] = f["action"][r0:r1]
            d_pr[off:n] = f["proprio"][r0:r1]
            d_st[off:n] = f["state"][r0:r1]
            d_ei[off:n] = new_i
            d_si[off:n] = np.arange(T, dtype=np.int64)
            d_sc[off:n] = f["scene_idx"][r0:r1]
            ep_len.append(T); ep_off.append(off); off = n
            if (new_i + 1) % 100 == 0:
                print(f"  [{new_i+1}/{len(keep)}] N={off} {(new_i+1)/(time.time()-t0):.1f} clip/s")
        out.create_dataset("ep_len", data=np.array(ep_len, "int32"))
        out.create_dataset("ep_offset", data=np.array(ep_off, "int64"))
        out.attrs["scene_names"] = ",".join(names)      # keep FULL numbering
        out.attrs["heldout_scenes"] = ",".join(heldout_names)
        out.attrs["n_frames"] = off
        out.attrs["n_clips"] = len(ep_len)
    f.close()
    print(f"[done] {time.time()-t0:.0f}s -> {dst}: {len(ep_len)} clips, {off} frames")


if __name__ == "__main__":
    main()
