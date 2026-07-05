"""Convert Physion++ -> le-wm flat HDF5, for training lewm ON real physics data.

Goal (per user): run the FULL phyworld research framework (free-rollout +
structured/consistency + rollout OOD eval) directly on real-looking Physion++,
and see if we can make OOD + long-horizon good on real data.

Physion++ gives us what phyworld had:
  - pixels        : the _img.mp4 frames (mp4 frame-count == pkl frame-count, exact)
  - proprio       : the TARGET object's 3D position (auto-detected as the moving
                    object: per-object position std, argmax). Real physical state!
  - state         : the target object's 3D velocity (for probe / structured targets)
  - action        : PLACEHOLDER noise (Physion++ is passive, no control) -> lewm
                    degrades to V-JEPA-style; free-rollout still trains the dynamics.
  - scene_idx     : OOD dimension = physical-property scene type (mass/friction/
                    bouncy/deform). Property *values* are hidden (Physion++ tests
                    online inference), so OOD is defined by held-out scene/shape.

Output: $STABLEWM_HOME/datasets/<name>.h5
"""
import argparse
import glob
import os
import pickle
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

PP = Path("/data1/likun-share/junjxu/physion_raw/physion_plus")


def target_index(pos):
    """pos (T, nobj, 3) -> index of the moving object (the target)."""
    return int(pos.std(axis=0).sum(axis=1).argmax())


def load_trial(mp4, pkl, size, frameskip):
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (size, size),
                                 interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        return None
    frames = np.stack(frames)
    d = pickle.load(open(pkl, "rb"))
    ks = sorted(d["frames"])
    pos = np.stack([d["frames"][k]["objects"]["positions"] for k in ks])   # (T, nobj, 3)
    vel = np.stack([d["frames"][k]["objects"]["velocities"] for k in ks])
    T = min(len(frames), len(ks))
    frames, pos, vel = frames[:T], pos[:T], vel[:T]
    ti = target_index(pos)
    proprio = pos[:, ti, :].astype("float32")   # (T, 3)
    state = vel[:, ti, :].astype("float32")
    if frameskip > 1:
        frames = frames[::frameskip]
        proprio = proprio[::frameskip]
        state = state[::frameskip]
    return frames, proprio, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="readout", choices=["readout", "test", "train"])
    ap.add_argument("--name", default="physionpp_readout")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--frameskip", type=int, default=4, help="352 frames -> ~88 @fs4")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scenes", default="", help="comma-sep scene filter, e.g. mass_collision,friction_collision")
    ap.add_argument("--action-noise", type=float, default=0.01)
    args = ap.parse_args()

    stablewm = os.environ.get("STABLEWM_HOME") or os.path.expanduser("~/.stable_worldmodel")
    dst = os.path.join(stablewm, "datasets", f"{args.name}.h5")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    rng = np.random.default_rng(0)

    root = PP / f"{args.split}_ext" / f"{args.split}_data_v1"
    mp4s = sorted(glob.glob(str(root / "**" / "*_img.mp4"), recursive=True))
    scenes = [s for s in args.scenes.split(",") if s] if args.scenes else None
    if scenes:
        mp4s = [m for m in mp4s if any(s in m for s in scenes)]
    if args.limit:
        mp4s = mp4s[: args.limit]
    S = args.img_size
    print(f"[convert] {len(mp4s)} trials -> {dst} (frameskip={args.frameskip}, scenes={scenes or 'all'})")

    t0 = time.time()
    scene_names = []
    with h5py.File(dst, "w") as out:
        d_px = out.create_dataset("pixels", (0, S, S, 3), maxshape=(None, S, S, 3), dtype="uint8",
                                  chunks=(16, S, S, 3), compression="gzip", compression_opts=4)
        d_ac = out.create_dataset("action", (0, 3), maxshape=(None, 3), dtype="float32")
        d_pr = out.create_dataset("proprio", (0, 3), maxshape=(None, 3), dtype="float32")
        d_st = out.create_dataset("state", (0, 3), maxshape=(None, 3), dtype="float32")
        d_ei = out.create_dataset("episode_idx", (0,), maxshape=(None,), dtype="int64")
        d_si = out.create_dataset("step_idx", (0,), maxshape=(None,), dtype="int64")
        d_sc = out.create_dataset("scene_idx", (0,), maxshape=(None,), dtype="int64")  # OOD label
        ep_len, ep_off, off = [], [], 0
        for ei, mp4 in enumerate(mp4s):
            pkl = mp4.replace("_img.mp4", ".pkl")
            if not os.path.exists(pkl):
                continue
            try:
                loaded = load_trial(mp4, pkl, S, args.frameskip)
            except Exception as e:
                print("  !! skip", os.path.basename(mp4), e)
                continue
            if loaded is None:
                continue
            fr, pr, st = loaded
            T = len(fr)
            sc = Path(mp4).parts[-3].replace("_pp", "")
            if sc not in scene_names:
                scene_names.append(sc)
            sci = scene_names.index(sc)
            n = off + T
            for d in (d_px, d_ac, d_pr, d_st, d_ei, d_si, d_sc):
                d.resize(n, axis=0)
            d_px[off:n] = fr
            d_ac[off:n] = (rng.standard_normal((T, 3)) * args.action_noise).astype("float32")
            d_pr[off:n] = pr
            d_st[off:n] = st
            d_ei[off:n] = ei
            d_si[off:n] = np.arange(T, dtype=np.int64)
            d_sc[off:n] = sci
            ep_len.append(T)
            ep_off.append(off)
            off = n
            if (ei + 1) % 100 == 0:
                print(f"  [{ei+1}/{len(mp4s)}] N={off} {(ei+1)/(time.time()-t0):.1f} clip/s")
        out.create_dataset("ep_len", data=np.array(ep_len, "int32"))
        out.create_dataset("ep_offset", data=np.array(ep_off, "int64"))
        out.attrs["scene_names"] = ",".join(scene_names)
        out.attrs["n_frames"] = off
        out.attrs["n_clips"] = len(ep_len)
        out.attrs["action"] = "placeholder_gaussian_noise"
    print(f"[done] {time.time()-t0:.0f}s {len(ep_len)} clips, {off} frames, scenes={scene_names}")


if __name__ == "__main__":
    main()
