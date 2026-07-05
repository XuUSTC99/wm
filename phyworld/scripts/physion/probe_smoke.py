"""Smoke test: does the range-B core link work end-to-end?
  Physion redyellow MP4  ->  frames  ->  lewm encoder (frozen)  ->  (T, D) emb

Verifies ckpt deserialization, image preprocessing (must match FT / rollout eval),
and JEPA.encode() before we build the full OCP readout. CPU-only to not disturb
any training on the GPUs.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/home/likun-share/junjxu/wm")
sys.path.insert(0, str(ROOT / "le-wm"))  # ckpt is a pickled JEPA object -> needs le-wm on path

CKPT = "/data1/likun-share/junjxu/.stable_worldmodel/collision_rerun_w5p0_f2_id1k/collision_rerun_w5p0_f2_id1k_epoch_20_object.ckpt"
MP4 = "/data1/likun-share/junjxu/physion_raw/_core/Physion/Collide/mp4s-redyellow/pilot_it2_collision_assorted_targets_box-redyellow_0001_img.mp4"

# Must match training (utils.get_img_preprocessor) / rollout_eval_id1k.py exactly.
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_video_224(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)  # cv2 reads BGR
        frames.append(cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA))
    cap.release()
    return np.stack(frames)  # (T, 224, 224, 3) uint8


@torch.no_grad()
def encode(model, frames_u8, dev):
    x = torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float().to(dev) / 255.0
    x = (x - IMNET_MEAN.to(dev)) / IMNET_STD.to(dev)
    info = {"pixels": x.unsqueeze(0)}  # (1, T, C, H, W)
    return model.encode(info)["emb"][0]  # (T, D)


def main():
    dev = "cpu"
    print(f"[load] {CKPT}")
    model = torch.load(CKPT, map_location="cpu", weights_only=False).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[model] {type(model).__name__}  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    frames = load_video_224(MP4)
    print(f"[video] {Path(MP4).name}  frames={frames.shape} dtype={frames.dtype}")

    emb = encode(model, frames, dev)
    print(f"[emb]   shape={tuple(emb.shape)}  nan={torch.isnan(emb).any().item()}  "
          f"mean={emb.mean().item():.4f}  std={emb.std().item():.4f}")
    print("[OK] range-B core link works: MP4 -> frames -> encoder -> per-frame embeddings")


if __name__ == "__main__":
    main()
