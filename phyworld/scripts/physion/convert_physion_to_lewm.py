"""Convert Physion dynamics MP4s -> le-wm flat HDF5 (range-A training data).

Design decisions (range A, 2026-07-05) — recorded here per user's request to
自主决策 + 记录理由:

- SCOPE: Collide scenario, ~1000 clips. Full 8-scenario dynamics would be
  ~545 GB as 224² uint8 HDF5 (disk /data1 only has ~211 G). Collide chosen to
  (a) align with the range-B Collide OCP probe -> direct "train-on-Physion vs
  zero-shot" comparison, (b) collision physics matches lewm's training strength.

- ACTION: Physion is PASSIVE (no control signal). le-wm's forward pass needs a
  per-frame `action` with non-zero variance (its column normalizer divides by
  std). We synthesise a PLACEHOLDER action = zero-mean small Gaussian noise.
  Effect: lewm degrades to a V-JEPA-style pure-video JEPA — the predictor gets
  no informative action, so representation learning is driven by sigreg + next-
  embedding prediction. This is the honest way to run a latent WM on passive
  video; we do NOT inject fake "action" semantics (e.g. optical flow) that could
  leak or mislead.

- NO proprio/state: MP4s carry no object positions; OCP readout only needs the
  encoder representation. config keys_to_load = [pixels, action].

Output: $STABLEWM_HOME/datasets/<name>.h5  (HDF5Dataset(name=...) finds it there)
"""
import argparse
import glob
import os
import time

import cv2
import h5py
import numpy as np


def load_video_224(path, size=224):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA))
    cap.release()
    return np.stack(frames) if frames else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, help="dir of Physion dynamics *.mp4")
    ap.add_argument("--name", default="physion_collide_dyn")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--limit", type=int, default=1000, help="0 = all clips in dir")
    ap.add_argument("--action-dim", type=int, default=2)
    ap.add_argument("--action-noise", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stablewm = os.environ.get("STABLEWM_HOME") or os.path.expanduser("~/.stable_worldmodel")
    dst = os.path.join(stablewm, "datasets", f"{args.name}.h5")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    mp4s = sorted(glob.glob(os.path.join(args.src_dir, "*.mp4")))
    if args.limit:
        mp4s = mp4s[: args.limit]
    S = args.img_size
    print(f"[convert] {len(mp4s)} clips, img={S}, action_dim={args.action_dim} (placeholder noise) -> {dst}")

    t0 = time.time()
    with h5py.File(dst, "w") as out:
        d_pixels = out.create_dataset("pixels", shape=(0, S, S, 3), maxshape=(None, S, S, 3),
                                      dtype="uint8", chunks=(32, S, S, 3),
                                      compression="gzip", compression_opts=4)
        d_action = out.create_dataset("action", shape=(0, args.action_dim),
                                      maxshape=(None, args.action_dim), dtype="float32")
        d_epidx = out.create_dataset("episode_idx", shape=(0,), maxshape=(None,), dtype="int64")
        d_stepidx = out.create_dataset("step_idx", shape=(0,), maxshape=(None,), dtype="int64")
        ep_len, ep_off, off = [], [], 0
        for ei, p in enumerate(mp4s):
            fr = load_video_224(p, S)
            if fr is None:
                print(f"  !! skip unreadable {p}")
                continue
            T = fr.shape[0]
            n = off + T
            for d in (d_pixels, d_action, d_epidx, d_stepidx):
                d.resize(n, axis=0)
            d_pixels[off:n] = fr
            d_action[off:n] = (rng.standard_normal((T, args.action_dim)) * args.action_noise).astype("float32")
            d_epidx[off:n] = ei
            d_stepidx[off:n] = np.arange(T, dtype=np.int64)
            ep_len.append(T)
            ep_off.append(off)
            off = n
            if (ei + 1) % 100 == 0 or ei == len(mp4s) - 1:
                rate = (ei + 1) / max(time.time() - t0, 1e-6)
                print(f"  [{ei+1}/{len(mp4s)}] N={off} {rate:.1f} clip/s ETA {(len(mp4s)-ei-1)/max(rate,1e-6):.0f}s")
        out.create_dataset("ep_len", data=np.array(ep_len, "int32"))
        out.create_dataset("ep_offset", data=np.array(ep_off, "int64"))
        out.attrs["n_frames"] = off
        out.attrs["n_clips"] = len(ep_len)
        out.attrs["action"] = "placeholder_zero_mean_gaussian_noise"
        out.attrs["source"] = args.src_dir
    print(f"[done] {time.time()-t0:.0f}s  {len(ep_len)} clips, {off} frames -> {dst}")


if __name__ == "__main__":
    main()
