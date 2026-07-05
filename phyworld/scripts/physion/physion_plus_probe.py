"""Physion++ zero-shot OCP probe (range-B style).

Physion++ readout/test trials store the OCP outcome inside the per-frame pkl:
  pkl['frames'][<frame>]['labels']['target_contacting_zone'] (bool).
outcome = 1 if the target ever contacts the zone across the trial, else 0.
Video: <trial>_img.mp4 (RGB, random colors). We freeze the lewm encoder, pool the
first ~1.5s of frame embeddings, and do stratified k-fold logistic regression —
same protocol as ocp_probe.py, so numbers are comparable to the Physion probe.

Note: this is a fast linear-separability signal, NOT the official Physion++
property-conditioned protocol. --random-init gives the untrained-encoder baseline.

Usage:
  python physion_plus_probe.py --split readout --ctx 45 --device cuda:2 [--random-init]
"""
import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/home/likun-share/junjxu/wm")
sys.path.insert(0, str(ROOT / "le-wm"))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PP_ROOT = Path("/data1/likun-share/junjxu/physion_raw/physion_plus")
DEFAULT_CKPT = ("/data1/likun-share/junjxu/.stable_worldmodel/"
                "collision_rerun_w5p0_f2_id1k/collision_rerun_w5p0_f2_id1k_epoch_20_object.ckpt")
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pkl_label(pkl_path):
    try:
        d = pickle.load(open(pkl_path, "rb"))
        fr = d["frames"]
        return int(any(bool(fr[k]["labels"].get("target_contacting_zone", False)) for k in fr))
    except Exception:
        return None


def load_video_224(path, ctx):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (224, 224),
                                 interpolation=cv2.INTER_AREA))
        if ctx and len(frames) >= ctx:
            break
    cap.release()
    return np.stack(frames) if frames else None


@torch.no_grad()
def encode(model, frames_u8, dev):
    x = torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float().to(dev) / 255.0
    x = (x - IMNET_MEAN.to(dev)) / IMNET_STD.to(dev)
    return model.encode({"pixels": x.unsqueeze(0)})["emb"][0]


def pool(emb, mode):
    if mode == "mean":
        return emb.mean(0)
    if mode == "meanstd":
        return torch.cat([emb.mean(0), emb.std(0)], -1)
    if mode == "last":
        return emb[-1]
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="readout", choices=["readout", "test"])
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--ctx", type=int, default=45)
    ap.add_argument("--pool", default="meanstd")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="cap trials for a quick run")
    ap.add_argument("--random-init", action="store_true")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    root = PP_ROOT / f"{args.split}_ext" / f"{args.split}_data_v1"
    mp4s = sorted(glob.glob(str(root / "**" / "*_img.mp4"), recursive=True))
    if args.limit:
        mp4s = mp4s[: args.limit]
    print(f"[data] Physion++ {args.split}: {len(mp4s)} _img.mp4 under {root}")

    model = torch.load(args.ckpt, map_location="cpu", weights_only=False).to(dev).eval()
    if args.random_init:
        n = 0
        for mod in list(model.encoder.modules()) + list(model.projector.modules()):
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters(); n += 1
        print(f"[baseline] reinit {n} submodules -> UNTRAINED encoder")
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[ckpt] {'RANDOM-INIT' if args.random_init else Path(args.ckpt).name}  ctx={args.ctx} pool={args.pool} dev={dev}")

    X, y, miss = [], [], 0
    for i, mp4 in enumerate(mp4s):
        pkl = mp4.replace("_img.mp4", ".pkl")
        if not os.path.exists(pkl):
            miss += 1; continue
        lab = pkl_label(pkl)
        if lab is None:
            miss += 1; continue
        fr = load_video_224(mp4, args.ctx)
        if fr is None:
            miss += 1; continue
        X.append(pool(encode(model, fr, dev), args.pool).cpu().numpy())
        y.append(lab)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(mp4s)}] collected={len(y)}")
    X, y = np.stack(X), np.array(y)
    chance = max(y.mean(), 1 - y.mean())
    print(f"[matched] n={len(y)} pos={int(y.sum())} neg={int(len(y)-y.sum())} "
          f"chance={chance:.3f} skipped={miss}")

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(args.folds, shuffle=True, random_state=0)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    print(f"[Physion++ OCP probe] acc={acc.mean():.3f}±{acc.std():.3f} "
          f"auc={auc.mean():.3f}±{auc.std():.3f} (chance={chance:.3f}) feat_dim={X.shape[1]}")


if __name__ == "__main__":
    main()
