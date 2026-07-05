"""Quick range-B probe: is a frozen lewm encoder's representation linearly
predictive of Physion OCP (object-contact) outcome?

NOT the official Physion protocol (which trains a readout on the *readout* split
and tests on the *test* split). This is a fast sanity signal: extract features
from the Test-Core redyellow MP4s and do stratified k-fold logistic regression
on them. Tells us whether the transferred representation carries contact-relevant
physics before we invest in the full readout pipeline.

Usage:
  python ocp_probe.py --scenario Collide --ctx 45 --device cuda:1
"""
import argparse
import csv
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/home/likun-share/junjxu/wm")
sys.path.insert(0, str(ROOT / "le-wm"))  # ckpt is a pickled JEPA -> needs le-wm on path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CORE = Path("/data1/likun-share/junjxu/physion_raw/_core/Physion")
DEFAULT_CKPT = ("/data1/likun-share/junjxu/.stable_worldmodel/"
                "collision_rerun_w5p0_f2_id1k/collision_rerun_w5p0_f2_id1k_epoch_20_object.ckpt")
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_labels():
    """trial_name(without -redyellow) -> {0,1} OCP outcome."""
    d = {}
    with open(CORE / "labels.csv") as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            if len(row) < 2 or not row[0]:
                continue
            d[row[0]] = 1 if row[1].strip().lower() == "true" else 0
    return d


def mp4_key(path):
    # pilot_..._box-redyellow_0001_img.mp4 -> pilot_..._box_0001_img  (matches labels.csv key)
    return Path(path).stem.replace("-redyellow", "")


def load_video_224(path, ctx):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
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
    return np.stack(frames), fps


@torch.no_grad()
def encode(model, frames_u8, dev):
    x = torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float().to(dev) / 255.0
    x = (x - IMNET_MEAN.to(dev)) / IMNET_STD.to(dev)
    return model.encode({"pixels": x.unsqueeze(0)})["emb"][0]  # (T, D)


def pool(emb, mode):
    if mode == "mean":
        return emb.mean(0)
    if mode == "meanstd":
        return torch.cat([emb.mean(0), emb.std(0)], -1)
    if mode == "last":
        return emb[-1]
    if mode == "meancat_last":
        return torch.cat([emb.mean(0), emb[-1]], -1)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="Collide")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--ctx", type=int, default=45, help="use first N frames (~1.5s context)")
    ap.add_argument("--pool", default="meanstd", choices=["mean", "meanstd", "last", "meancat_last"])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--random-init", action="store_true",
                    help="reinit encoder+projector -> untrained baseline (is the signal real physics or just architecture?)")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    labels = load_labels()
    mp4s = sorted(glob(str(CORE / args.scenario / "mp4s-redyellow" / "*.mp4")))
    print(f"[data] {args.scenario}: {len(mp4s)} mp4s, {sum(k.startswith('') for k in labels)} labels total")

    model = torch.load(args.ckpt, map_location="cpu", weights_only=False).to(dev).eval()
    if args.random_init:
        n = 0
        for mod in list(model.encoder.modules()) + list(model.projector.modules()):
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
                n += 1
        print(f"[baseline] reinitialized {n} submodules -> UNTRAINED encoder")
    for p in model.parameters():
        p.requires_grad_(False)
    tag = "RANDOM-INIT" if args.random_init else Path(args.ckpt).name
    print(f"[ckpt] {tag}  ctx={args.ctx} pool={args.pool} dev={dev}")

    X, y, miss, fps0 = [], [], 0, None
    for p in mp4s:
        key = mp4_key(p)
        if key not in labels:
            miss += 1
            continue
        frames, fps = load_video_224(p, args.ctx)
        fps0 = fps0 or fps
        emb = encode(model, frames, dev)
        X.append(pool(emb, args.pool).cpu().numpy())
        y.append(labels[key])
    X, y = np.stack(X), np.array(y)
    chance = max(y.mean(), 1 - y.mean())
    print(f"[matched] n={len(y)}  pos={y.sum()} neg={len(y)-y.sum()}  "
          f"chance(majority)={chance:.3f}  unmatched_mp4={miss}  fps~{fps0:.1f} (ctx {args.ctx}f≈{args.ctx/max(fps0,1):.2f}s)")

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(args.folds, shuffle=True, random_state=0)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    print(f"[OCP probe] acc={acc.mean():.3f}±{acc.std():.3f}  auc={auc.mean():.3f}±{auc.std():.3f}  "
          f"(chance acc={chance:.3f})  feat_dim={X.shape[1]}")


if __name__ == "__main__":
    main()
