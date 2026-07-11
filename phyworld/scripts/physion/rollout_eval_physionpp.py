"""Rollout OOD + long-horizon eval for lewm trained ON Physion++ (real data).

Same protocol as phyworld rollout_eval_id1k, adapted to real Physion++:
  - partition = physical-property SCENE (mass/friction/bouncy/deform) instead of
    phyworld's r/m/v-OOD numeric ranges.
  - proprio = target 3D position, state = target 3D velocity (real, from pkl).
  - horizon pushed to long range (up to 64) — Physion++ clips are ~90 frames.

Metrics:
  - latent cos/nMSE by horizon (long-horizon drift) and by scene (property OOD)
  - target pos/vel decode ρ from ROLLED-OUT latents (Ridge probe fit on REAL embs
    of train trajs, applied to predicted embs of held-out trajs).

Usage: python rollout_eval_physionpp.py --ckpt <physionpp_ckpt> --tag <name> --device cuda:N
"""
import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path("/home/likun-share/junjxu/wm")
sys.path.insert(0, str(ROOT / "le-wm"))
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

SWM = Path(os.environ.get("STABLEWM_HOME", "/data1/likun-share/junjxu/.stable_worldmodel"))
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
HS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h5", default=str(SWM / "datasets/physionpp_readout.h5"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-trajs", type=int, default=300)
    ap.add_argument("--tag", default="")
    ap.add_argument("--group-scenes", default="", help="comma-sep scene names; report mixed horizon nMSE/cos for this group (held-out OOD, robust vs per-scene artifact)")
    args = ap.parse_args()
    dev = args.device
    t0 = time.time()

    model = torch.load(args.ckpt, map_location="cpu", weights_only=False).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[ckpt] {Path(args.ckpt).name}  {args.tag}", flush=True)

    f = h5py.File(args.h5, "r")
    ep_idx = f["episode_idx"][:]; step_idx = f["step_idx"][:]
    proprio = f["proprio"][:]; action = f["action"][:]; state = f["state"][:]
    scene = f["scene_idx"][:]
    scene_names = f.attrs["scene_names"].split(",")
    pixels = f["pixels"]
    a_mean = np.nan_to_num(action).mean(0); a_std = np.nan_to_num(action).std(0) + 1e-8

    uniq = np.unique(ep_idx)
    rng = np.random.default_rng(0); perm = rng.permutation(uniq)
    sel = perm[: args.max_trajs] if args.max_trajs else perm
    n_tr = int(round(len(sel) * 0.8)); train_eps = set(sel[:n_tr].tolist())

    @torch.no_grad()
    def encode_frames(fr):
        x = torch.from_numpy(fr).permute(0, 3, 1, 2).float().to(dev) / 255.0
        x = (x - IMNET_MEAN.to(dev)) / IMNET_STD.to(dev)
        return model.encode({"pixels": x.unsqueeze(0)})["emb"][0]

    @torch.no_grad()
    def ar_rollout(real_emb, act_norm):
        T = real_emb.size(0); emb_hist = real_emb[:HS].clone()
        ae_all = model.action_encoder(act_norm.unsqueeze(0))[0]
        preds = []
        for k in range(HS, T):
            e_in = emb_hist[-HS:].unsqueeze(0); a_in = ae_all[k - HS:k].unsqueeze(0)
            p = model.predict(e_in, a_in)[0, -1]
            preds.append(p); emb_hist = torch.cat([emb_hist, p.unsqueeze(0)], 0)
        return torch.stack(preds, 0)

    real_E, pred_E, meta, pos_all, vel_all = [], [], [], [], []
    nd = 0
    for ep in sel:
        rows = np.nonzero(ep_idx == ep)[0]; rows = rows[np.argsort(step_idx[rows])]
        T = len(rows)
        if T <= HS + 1:
            continue
        frames = pixels[rows[0]:rows[0] + T] if np.all(np.diff(rows) == 1) else pixels[:][rows]
        real_emb = encode_frames(frames)
        raw = np.nan_to_num(action[rows]).astype(np.float64)
        act_norm = torch.from_numpy(((raw - a_mean) / a_std).astype(np.float32)).to(dev)
        pred = ar_rollout(real_emb, act_norm)
        sc = int(scene[rows[0]]); in_tr = ep in train_eps
        re = real_emb.cpu().numpy(); pe = pred.cpu().numpy()
        for j, k in enumerate(range(HS, T)):
            real_E.append(re[k]); pred_E.append(pe[j])
            pos_all.append(proprio[rows][k]); vel_all.append(state[rows][k])
            meta.append((int(ep), k - HS + 1, sc, in_tr))
        nd += 1
        if nd % 100 == 0:
            print(f"  {nd} trajs t={time.time()-t0:.0f}s", flush=True)
    f.close()

    real_E = np.array(real_E); pred_E = np.array(pred_E)
    pos_all = np.array(pos_all); vel_all = np.array(vel_all); meta = np.array(meta)
    horizon = meta[:, 1]; scn = meta[:, 2]; in_train = meta[:, 3].astype(bool)
    print(f"[data] {nd} trajs, {len(real_E)} (frame,pred) pairs, dim={real_E.shape[1]}", flush=True)

    mu = real_E[in_train].mean(0); sd = real_E[in_train].std(0) + 1e-6
    def z(x): return (x - mu) / sd
    rp = Ridge(alpha=1.0).fit(z(real_E[in_train]), pos_all[in_train])
    rv = Ridge(alpha=1.0).fit(z(real_E[in_train]), vel_all[in_train])
    te = ~in_train

    def lat(mask):
        r = real_E[mask]; p = pred_E[mask]
        cos = (r * p).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(p, axis=1) + 1e-8)
        nmse = ((p - r) ** 2).sum(1) / ((r - r.mean(0)) ** 2).sum(1).mean()
        return cos.mean(), nmse.mean()

    def dec(tag, X, mask):
        pp = rp.predict(z(X[mask])); pv = rv.predict(z(X[mask]))
        line = f"  {tag:22s}"
        for d in range(pos_all.shape[1]):
            line += f" pos{d}ρ={pearsonr(pos_all[mask][:, d], pp[:, d])[0]:+.2f}"
        for d in range(vel_all.shape[1]):
            line += f" vel{d}ρ={pearsonr(vel_all[mask][:, d], pv[:, d])[0]:+.2f}"
        print(line)

    print(f"\n=== Physion++ AR ROLLOUT  {args.tag}  (HS={HS}) ===")
    print("--- latent fidelity vs HORIZON (long-horizon drift, test) ---")
    for h in [1, 2, 4, 8, 16, 32, 64]:
        m = te & (horizon == h)
        if m.sum() < 20:
            continue
        cos, nmse = lat(m); print(f"  h={h:3d}  n={m.sum():5d}  cos={cos:+.4f}  nMSE={nmse:.4f}")

    print("--- latent fidelity by SCENE (physical-property OOD, test) ---")
    for s in sorted(np.unique(scn)):
        m = te & (scn == s)
        if m.sum() < 50:
            continue
        cos, nmse = lat(m); print(f"  {scene_names[s]:20s} n={m.sum():5d}  cos={cos:+.4f}  nMSE={nmse:.4f}")

    print("--- decoded target pos/vel ρ from ROLLED-OUT latents (test, by scene) ---")
    for s in sorted(np.unique(scn)):
        m = te & (scn == s)
        if m.sum() < 50:
            continue
        dec(f"PRED {scene_names[s]}", pred_E, m)

    if args.group_scenes:
        gids = [scene_names.index(s) for s in args.group_scenes.split(",") if s]
        gm = np.isin(scn, gids)
        print(f"\n--- GROUP [{args.group_scenes}] mixed horizon (all frames; OOD if held-out ckpt) ---")
        for h in [1, 2, 4, 8, 16, 32, 64]:
            m = gm & (horizon == h)
            if m.sum() < 20:
                continue
            cos, nmse = lat(m)
            print(f"  h={h:3d}  n={m.sum():5d}  cos={cos:+.4f}  nMSE={nmse:.4f}")


if __name__ == "__main__":
    main()
