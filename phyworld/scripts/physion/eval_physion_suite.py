"""One-command Physion + Physion++ OCP evaluation for ANY lewm checkpoint.

Purpose (the whole point of this line of work): a REUSABLE evaluation harness so
that when the lewm session finds a better phyworld method, its checkpoint can be
dropped onto Physion / Physion++ with a single command:

    python eval_physion_suite.py --ckpt <path/to/ckpt> --tag <method_name>

What it does: freeze the ckpt's encoder, extract first-~1.5s frame features, run
stratified k-fold logistic regression for OCP (object-contact prediction). Reports
per-scenario acc/AUC for Physion (8 scenarios) + Physion++ readout (aggregate),
plus the Physion mean AUC. Writes a JSON so methods can be compared over time.

Add --random-init to get the untrained-encoder baseline (what any signal must beat).

Reuses the already-validated probes (ocp_probe.py / physion_plus_probe.py) so the
metric definition stays identical to the zero-shot baseline runs.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/likun-share/junjxu/wm")
sys.path.insert(0, str(ROOT / "le-wm"))                       # ckpt is a pickled JEPA
sys.path.insert(0, str(ROOT / "phyworld" / "scripts" / "physion"))

import ocp_probe as P            # load_labels, mp4_key, load_video_224, encode, pool, CORE
import physion_plus_probe as PP  # pkl_label, load_video_224, encode, pool, PP_ROOT

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PHYSION_SCENARIOS = ["Dominoes", "Support", "Collide", "Contain", "Drop", "Roll", "Link", "Drape"]


def kfold_scores(X, y, folds=5):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(folds, shuffle=True, random_state=0)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return float(acc.mean()), float(acc.std()), float(auc.mean()), float(auc.std())


def eval_physion(model, dev, ctx, pool):
    labels = P.load_labels()
    out = {}
    for sc in PHYSION_SCENARIOS:
        mp4s = sorted(glob.glob(str(P.CORE / sc / "mp4s-redyellow" / "*.mp4")))
        X, y = [], []
        for m in mp4s:
            k = P.mp4_key(m)
            if k not in labels:
                continue
            fr, _ = P.load_video_224(m, ctx)  # ocp_probe returns (frames, fps)
            X.append(P.pool(P.encode(model, fr, dev), pool).cpu().numpy())
            y.append(labels[k])
        if len(set(y)) < 2:
            out[sc] = dict(n=len(y), note="skipped (single class or no data)")
            continue
        X, y = np.stack(X), np.array(y)
        a, asd, u, usd = kfold_scores(X, y)
        out[sc] = dict(n=len(y), acc=a, acc_std=asd, auc=u, auc_std=usd,
                       chance=float(max(y.mean(), 1 - y.mean())))
    return out


def eval_physionpp(model, dev, ctx, pool, split="readout"):
    root = PP.PP_ROOT / f"{split}_ext" / f"{split}_data_v1"
    mp4s = sorted(glob.glob(str(root / "**" / "*_img.mp4"), recursive=True))
    if not mp4s:
        return dict(note=f"{split} not extracted at {root}")
    X, y = [], []
    for m in mp4s:
        pkl = m.replace("_img.mp4", ".pkl")
        if not os.path.exists(pkl):
            continue
        lab = PP.pkl_label(pkl)
        if lab is None:
            continue
        fr = PP.load_video_224(m, ctx)
        if fr is None:
            continue
        X.append(PP.pool(PP.encode(model, fr, dev), pool).cpu().numpy())
        y.append(lab)
    X, y = np.stack(X), np.array(y)
    a, asd, u, usd = kfold_scores(X, y)
    return dict(n=len(y), acc=a, acc_std=asd, auc=u, auc_std=usd,
                chance=float(max(y.mean(), 1 - y.mean())))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="path to a lewm *_object.ckpt (pickled JEPA)")
    ap.add_argument("--tag", default="", help="label for this method in output/JSON")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--ctx", type=int, default=45, help="first N frames (~1.5s at 30fps)")
    ap.add_argument("--pool", default="meanstd", choices=["mean", "meanstd", "last"])
    ap.add_argument("--random-init", action="store_true", help="untrained-encoder baseline")
    ap.add_argument("--skip-pp", action="store_true", help="skip Physion++ (Physion only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    model = torch.load(args.ckpt, map_location="cpu", weights_only=False).to(dev).eval()
    if args.random_init:
        n = 0
        for mod in list(model.encoder.modules()) + list(model.projector.modules()):
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters(); n += 1
        print(f"[baseline] reinit {n} submodules -> UNTRAINED encoder")
    for p in model.parameters():
        p.requires_grad_(False)
    tag = args.tag or ("RANDOM" if args.random_init else Path(args.ckpt).stem)
    print(f"===== Physion suite eval: {tag}  (ctx={args.ctx} pool={args.pool} dev={dev}) =====")

    phys = eval_physion(model, dev, args.ctx, args.pool)
    print("--- Physion (8 scenarios, OCP 5-fold) ---")
    aucs = []
    for sc, r in phys.items():
        if "auc" not in r:
            print(f"  {sc:9s} {r.get('note','')}"); continue
        print(f"  {sc:9s} n={r['n']:3d}  acc={r['acc']:.3f}  auc={r['auc']:.3f}  (chance {r['chance']:.2f})")
        aucs.append(r["auc"])
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    print(f"  >> Physion mean AUC = {mean_auc:.3f}")

    ppr = None
    if not args.skip_pp:
        ppr = eval_physionpp(model, dev, args.ctx, args.pool, "readout")
        if "auc" in ppr:
            print(f"--- Physion++ readout (OCP 5-fold) ---")
            print(f"  n={ppr['n']}  acc={ppr['acc']:.3f}  auc={ppr['auc']:.3f}  (chance {ppr['chance']:.2f})")
        else:
            print(f"--- Physion++ : {ppr.get('note')}")

    out = args.out or str(ROOT / "reports" / "physion" / f"eval_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(tag=tag, ckpt=args.ckpt, ctx=args.ctx, pool=args.pool,
                   physion=phys, physion_mean_auc=mean_auc, physion_pp_readout=ppr),
              open(out, "w"), indent=2, default=float)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
