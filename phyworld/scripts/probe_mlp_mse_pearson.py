"""2-layer MLP probe on cached encoder embeddings (LeWM-paper aligned).

Uses LeWM's actual MLP class from `stable_pretraining.backbone.mlp` to match the
exact architecture used in LeWM Table 1 physical-quantity probing. Reports
MSE + Pearson ρ for regression targets, AUC for collision event.

Architecture: stable_pretraining.backbone.mlp.MLP(in_dim, [hidden_dim, out_dim])
              = Linear(in_dim, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, out_dim) → Dropout
Default: hidden_dim=512, dropout=0.0 (matches LeWM/stable_pretraining default).
         For high-dim DiT-XL features (4608-D K=4 input), --dropout 0.3 mitigates overfit.
         Note: LeWM's MLP class adds trailing Dropout after final Linear, so dropout > 0
         can compress prediction scale on regression targets (ρ stays, MSE inflates).

Protocol matches Ridge probe (scripts/probe_ood_fullfit.py):
  - K=4 multi-frame concat input
  - 80/20 train/test split BY TRAJ (random seed=0, mixes all OOD partitions)
  - Standardize features (z-score using train stats)
  - Report aggregate + per-partition metrics

Usage:
    python phyworld/scripts/probe_mlp_mse_pearson.py --domain collision
    python phyworld/scripts/probe_mlp_mse_pearson.py --domain uniform_motion
    python phyworld/scripts/probe_mlp_mse_pearson.py --domain all
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# LeWM's actual MLP class — Linear → ReLU → Dropout → Linear → Dropout
LEWM_VENV = Path('/home/qlib/agent_memory/wm/le-wm/.venv/lib/python3.10/site-packages')
sys.path.insert(0, str(LEWM_VENV))
from stable_pretraining.backbone.mlp import MLP as LeWM_MLP


# ---------- shared helpers ----------

PART_NAMES = {0: "ID", 1: "r-OOD", 2: "v-OOD", 3: "both-OOD"}


def build_k4(emb, ep_idx, step_idx):
    """K=4 multi-frame concat. Pads first 3 frames as invalid."""
    N, D = emb.shape
    feats = np.zeros((N, 4 * D), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)
    by_ep = {}
    for i in range(N):
        by_ep.setdefault(int(ep_idx[i]), []).append((int(step_idx[i]), i))
    for ep, lst in by_ep.items():
        lst.sort()
        ordered = [idx for _, idx in lst]
        for k, idx in enumerate(ordered):
            if k >= 3:
                ctx = ordered[k - 3:k + 1]
                feats[idx] = np.concatenate([emb[j] for j in ctx])
                valid[idx] = True
    return feats, valid


def train_mlp(X_tr, y_tr, in_dim, out_dim, *,
              hidden=512, dropout=0.3, epochs=50, bs=512, lr=1e-3,
              device='cuda', seed=0):
    """Train LeWM-style 2-layer MLP with MSE loss.

    Architecture: Linear(in_dim, hidden) → ReLU → Dropout(p) → Linear(hidden, out_dim) → Dropout(p)
    via stable_pretraining.backbone.mlp.MLP(in_dim, [hidden, out_dim], dropout=p).
    """
    torch.manual_seed(seed)
    net = LeWM_MLP(
        in_dim,
        [hidden, out_dim],
        norm_layer=None,
        activation_layer=torch.nn.ReLU,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.from_numpy(X_tr).float().to(device)
    y_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    if y_t.ndim == 1:
        y_t = y_t.unsqueeze(1)
    N = X_t.shape[0]

    for ep in range(epochs):
        net.train()
        idx = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            sel = idx[i:i + bs]
            opt.zero_grad()
            pred = net(X_t[sel])
            l = loss_fn(pred, y_t[sel])
            l.backward()
            opt.step()
    net.eval()
    return net


@torch.no_grad()
def predict_mlp(net, X, device='cuda', bs=1024):
    X_t = torch.from_numpy(X).float().to(device)
    out = []
    for i in range(0, X_t.shape[0], bs):
        out.append(net(X_t[i:i + bs]).cpu().numpy())
    return np.concatenate(out, 0)


def mse_rho(true, pred, train_std=None):
    """Normalized MSE + Pearson ρ. 2D targets get per-dim metrics averaged.

    Normalized MSE = mean(((ŷ − y) / σ_train)²), where σ_train is per-dim std
    computed on the training set. This matches LeWM-paper Table 1 convention
    (norm MSE ≈ 1 − ρ², scale-invariant across datasets).
    """
    if true.ndim == 1 or true.shape[1] == 1:
        t = true.ravel(); p = pred.ravel()
        sd = float(train_std) if train_std is not None else 1.0
        return (
            float((((p - t) / sd) ** 2).mean()),
            float(pearsonr(t, p)[0]) if t.std() > 1e-9 else float('nan'),
        )
    mses, rhos = [], []
    for d in range(true.shape[1]):
        sd = float(train_std[d]) if train_std is not None else 1.0
        mses.append((((pred[:, d] - true[:, d]) / sd) ** 2).mean())
        if true[:, d].std() > 1e-9:
            rhos.append(pearsonr(true[:, d], pred[:, d])[0])
    return float(np.mean(mses)), float(np.mean(rhos))


# ---------- collision (2D pos, 2D vel, 1D collision_event) ----------

def run_collision(emb_path, label, args):
    print(f"\n{'=' * 75}")
    print(f" {label}  [COLLISION, dropout={args.dropout}]")
    print(f"{'=' * 75}")
    with h5py.File(args.collision_eval, "r") as f:
        prop = f['proprio'][:]; state = f['state'][:]; coll = f['collision_event'][:]
        ep_idx = f['episode_idx'][:]; step_idx = f['step_idx'][:]; part = f['partition'][:]

    pos_x_2d = prop[:, [0, 2]].astype(np.float32)
    vel_x_2d = state[:, [0, 2]].astype(np.float32)

    emb = np.load(emb_path)
    feats, valid = build_k4(emb, ep_idx, step_idx)
    rng = np.random.default_rng(args.seed)
    uniq = np.unique(ep_idx); perm = rng.permutation(uniq)
    n_tr = int(round(len(uniq) * 0.8))
    train_eps = set(perm[:n_tr].tolist())
    base_tr = np.array([e in train_eps for e in ep_idx])
    mask_tr = base_tr & valid; mask_te = (~base_tr) & valid

    mu, sigma = feats[mask_tr].mean(0), feats[mask_tr].std(0) + 1e-6
    Xtr = (feats[mask_tr] - mu) / sigma
    # target stats (per-dim, from train) for normalized MSE
    pos_train_std = pos_x_2d[mask_tr].std(0)
    vel_train_std = vel_x_2d[mask_tr].std(0)
    print(f"  train={mask_tr.sum()}, test={mask_te.sum()}, feat_dim={feats.shape[1]}")
    print(f"  pos train std (per-dim): {pos_train_std}")
    print(f"  vel train std (per-dim): {vel_train_std}")

    t0 = time.time()
    net_p = train_mlp(Xtr, pos_x_2d[mask_tr], feats.shape[1], 2,
                      hidden=args.hidden, dropout=args.dropout,
                      epochs=args.epochs, bs=args.bs, lr=args.lr, seed=args.seed)
    net_v = train_mlp(Xtr, vel_x_2d[mask_tr], feats.shape[1], 2,
                      hidden=args.hidden, dropout=args.dropout,
                      epochs=args.epochs, bs=args.bs, lr=args.lr, seed=args.seed)
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)
    clf.fit(Xtr, coll[mask_tr])
    print(f"  MLPs + LogReg trained in {time.time()-t0:.1f}s")

    print(f" {'partition':12s}  {'n':>5s}  {'pos nMSE':>9s}  {'pos ρ':>8s}  "
          f"{'vel nMSE':>9s}  {'vel ρ':>8s}  {'coll AUC':>9s}")
    rows = [('AGGREGATE', mask_te)]
    for p in sorted(np.unique(part)):
        m = mask_te & (part == p)
        if m.sum() >= 50:
            rows.append((PART_NAMES[p], m))
    for name, m in rows:
        Xte = (feats[m] - mu) / sigma
        pred_p = predict_mlp(net_p, Xte); pred_v = predict_mlp(net_v, Xte)
        mse_p, rho_p = mse_rho(pos_x_2d[m], pred_p, pos_train_std)
        mse_v, rho_v = mse_rho(vel_x_2d[m], pred_v, vel_train_std)
        try:
            auc = roc_auc_score(coll[m], clf.predict_proba(Xte)[:, 1])
        except Exception:
            auc = float('nan')
        print(f" {name:12s}  {m.sum():>5d}  {mse_p:>9.4f}  {rho_p:>+8.4f}  "
              f"{mse_v:>9.4f}  {rho_v:>+8.4f}  {auc:>9.4f}")


# ---------- uniform_motion (1D pos, 1D vx) ----------

def run_single_ball(emb_path, label, lewm_h5, src_hdf5, args, domain_tag):
    """Single-ball phyworld probe (uniform_motion or parabola)."""
    print(f"\n{'=' * 75}")
    print(f" {label}  [{domain_tag.upper()}, dropout={args.dropout}]")
    print(f"{'=' * 75}")
    with h5py.File(lewm_h5, 'r') as f1:
        prop = f1['proprio'][:]; action = f1['action'][:]
        ep_idx = f1['episode_idx'][:]; step_idx = f1['step_idx'][:]
    with h5py.File(src_hdf5, 'r') as f2:
        init = np.concatenate([f2['init_streams'][k][...] for k in sorted(f2['init_streams'])], 0)

    def pl(r, v):
        r_ok = 0.7 <= r <= 1.5
        v_ok = 1.0 <= abs(v) <= 4.0
        if r_ok and v_ok: return 0
        if not r_ok and v_ok: return 1
        if r_ok and not v_ok: return 2
        return 3
    parts = np.array([pl(float(init[i, 0]), float(init[i, 1])) for i in range(len(init))],
                     dtype=np.uint8)
    part = parts[ep_idx]
    # full 2D (x, y); auto-skip dims with zero train variance below (uniform_motion vy ≡ 0)
    pos_xy = prop[:, :2].astype(np.float32)
    vel_xy = action[:, :2].astype(np.float32)

    emb = np.load(emb_path)
    feats, valid = build_k4(emb, ep_idx, step_idx)
    rng = np.random.default_rng(args.seed)
    uniq = np.unique(ep_idx); perm = rng.permutation(uniq)
    n_tr = int(round(len(uniq) * 0.8))
    train_eps = set(perm[:n_tr].tolist())
    base_tr = np.array([e in train_eps for e in ep_idx])
    mask_tr = base_tr & valid; mask_te = (~base_tr) & valid

    mu, sigma = feats[mask_tr].mean(0), feats[mask_tr].std(0) + 1e-6
    Xtr = (feats[mask_tr] - mu) / sigma
    pos_train_std = pos_xy[mask_tr].std(0)
    vel_train_std = vel_xy[mask_tr].std(0)
    pos_dims = [d for d in range(2) if pos_train_std[d] > 1e-6]
    vel_dims = [d for d in range(2) if vel_train_std[d] > 1e-6]
    pos_names = ["pos_x", "pos_y"]; vel_names = ["vx", "vy"]
    print(f"  train={mask_tr.sum()}, test={mask_te.sum()}, feat_dim={feats.shape[1]}")
    print(f"  active dims: pos={[pos_names[d] for d in pos_dims]}, "
          f"vel={[vel_names[d] for d in vel_dims]}")
    print(f"  pos train std: {pos_train_std}, vel train std: {vel_train_std}")

    t0 = time.time()
    net_p = train_mlp(Xtr, pos_xy[mask_tr], feats.shape[1], 2,
                      hidden=args.hidden, dropout=args.dropout,
                      epochs=args.epochs, bs=args.bs, lr=args.lr, seed=args.seed)
    net_v = train_mlp(Xtr, vel_xy[mask_tr], feats.shape[1], 2,
                      hidden=args.hidden, dropout=args.dropout,
                      epochs=args.epochs, bs=args.bs, lr=args.lr, seed=args.seed)
    print(f"  MLPs trained in {time.time()-t0:.1f}s")

    header = f" {'partition':12s}  {'n':>5s}"
    for d in pos_dims: header += f"  {pos_names[d]+' nMSE':>10s} {pos_names[d]+' ρ':>9s}"
    for d in vel_dims: header += f"  {vel_names[d]+' nMSE':>10s} {vel_names[d]+' ρ':>9s}"
    print(header)

    def per_dim(true_full, pred_full, dims, sd):
        out = []
        for d in dims:
            t = true_full[:, d]; p = pred_full[:, d]
            nmse = float((((p - t) / sd[d]) ** 2).mean())
            rho = float(np.corrcoef(t, p)[0, 1]) if t.std() > 1e-9 else float('nan')
            out.append((nmse, rho))
        return out

    rows = [('AGGREGATE', mask_te)]
    for p in sorted(np.unique(part)):
        m = mask_te & (part == p)
        if m.sum() >= 50:
            rows.append((PART_NAMES[p], m))
    for name, m in rows:
        Xte = (feats[m] - mu) / sigma
        pred_p = predict_mlp(net_p, Xte); pred_v = predict_mlp(net_v, Xte)
        pm_pos = per_dim(pos_xy[m], pred_p, pos_dims, pos_train_std)
        pm_vel = per_dim(vel_xy[m], pred_v, vel_dims, vel_train_std)
        line = f" {name:12s}  {m.sum():>5d}"
        for nmse, rho in pm_pos: line += f"  {nmse:>10.4f} {rho:>+9.4f}"
        for nmse, rho in pm_vel: line += f"  {nmse:>10.4f} {rho:>+9.4f}"
        print(line)


def run_uniform(emb_path, label, args):
    return run_single_ball(emb_path, label, args.uniform_lewm, args.uniform_src,
                           args, domain_tag="uniform_motion")


def run_parabola(emb_path, label, args):
    return run_single_ball(emb_path, label, args.parabola_lewm, args.parabola_src,
                           args, domain_tag="parabola")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domain", choices=["collision", "uniform_motion", "parabola", "all"],
                    default="all")
    ap.add_argument("--emb-dir", default="/home/qlib/agent_memory/wm/artifacts/embeddings")
    ap.add_argument("--collision-eval", default="/home/qlib/.stable_worldmodel/phyworld_collision_eval.h5")
    ap.add_argument("--uniform-lewm", default="/home/qlib/.stable_worldmodel/phyworld_uniform_motion.h5")
    ap.add_argument("--uniform-src",
        default="/home/qlib/agent_memory/wm/phyworld/data/uniform_motion_eval.hdf5")
    ap.add_argument("--parabola-lewm", default="/home/qlib/.stable_worldmodel/phyworld_parabola.h5")
    ap.add_argument("--parabola-src",
        default="/home/qlib/agent_memory/wm/phyworld/data/parabola_eval.hdf5")
    ap.add_argument("--hidden", type=int, default=512, help="MLP hidden dim")
    ap.add_argument("--dropout", type=float, default=0.0,
        help="Dropout probability (LeWM default 0; try 0.3 if DiT-XL overfits)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.domain in ("collision", "all"):
        print("\n" + "#" * 75)
        print(f"# COLLISION  (2-layer MLP, hidden={args.hidden}, dropout={args.dropout})")
        print("#" * 75)
        run_collision(f'{args.emb_dir}/lewm_pusht_only_collision_eval_emb_52k_noproj.npy',
                      'LeWM pusht-only frozen (5.5M)', args)
        run_collision(f'{args.emb_dir}/lewm_16ep_epoch16_collision_eval_emb_52k_noproj.npy',
                      'LeWM paper-init+collision 16ep FT (5.5M, PARTITION-LEAKED)', args)
        run_collision(f'{args.emb_dir}/dit_xl_zeroshot_collision_eval_emb_52k.npy',
                      'DiT-XL zero-shot (749.8M)', args)
        ft_id1k = f'{args.emb_dir}/lewm_collision_paperinit_id1k_collision_eval_emb_52k_noproj.npy'
        if os.path.exists(ft_id1k):
            run_collision(ft_id1k,
                          'LeWM collision_paperinit FT 20ep ID-ONLY 1k (5.5M)', args)
        dit_id1k = f'{args.emb_dir}/dit_xl_lora_id1k_collision_eval_emb_52k.npy'
        if os.path.exists(dit_id1k):
            run_collision(dit_id1k,
                          'DiT-XL LoRA FT 8ep ID-ONLY 1k (749.8M)', args)

    if args.domain in ("uniform_motion", "all"):
        print("\n" + "#" * 75)
        print(f"# UNIFORM_MOTION  (2-layer MLP, hidden={args.hidden}, dropout={args.dropout})")
        print("#" * 75)
        run_uniform(f'{args.emb_dir}/lewm_pusht_only_uniform_motion_emb_37k_noproj.npy',
                    'LeWM pusht-only frozen (5.5M, no phyworld)', args)
        run_uniform(f'{args.emb_dir}/dit_xl_zeroshot_uniform_motion_emb_37k.npy',
                    'DiT-XL zero-shot (749.8M, no phyworld)', args)
        run_uniform(f'{args.emb_dir}/lewm_uniform_paperinit_leakfree_uniform_motion_emb_37k_noproj.npy',
                    'LeWM uniform_paperinit FT 20ep PARTITION-LEAKED (5.5M)', args)
        ft_id1k = f'{args.emb_dir}/lewm_uniform_paperinit_id1k_uniform_motion_emb_37k_noproj.npy'
        if os.path.exists(ft_id1k):
            run_uniform(ft_id1k,
                        'LeWM uniform_paperinit FT 20ep ID-ONLY 1k (5.5M)', args)
        dit_id1k = f'{args.emb_dir}/dit_xl_lora_id1k_uniform_motion_emb_37k.npy'
        if os.path.exists(dit_id1k):
            run_uniform(dit_id1k,
                        'DiT-XL LoRA FT 4ep ID-ONLY 1k (749.8M)', args)

    if args.domain in ("parabola", "all"):
        print("\n" + "#" * 75)
        print(f"# PARABOLA  (2-layer MLP, hidden={args.hidden}, dropout={args.dropout})")
        print("#" * 75)
        run_parabola(f'{args.emb_dir}/lewm_pusht_only_parabola_emb_noproj.npy',
                     'LeWM pusht-only frozen (5.5M, no phyworld)', args)
        run_parabola(f'{args.emb_dir}/dit_xl_zeroshot_parabola_emb.npy',
                     'DiT-XL zero-shot (749.8M, no phyworld)', args)
        ft_cache = f'{args.emb_dir}/lewm_parabola_paperinit_leakfree_parabola_emb_noproj.npy'
        if os.path.exists(ft_cache):
            run_parabola(ft_cache,
                         'LeWM parabola_paperinit FT 20ep PARTITION-LEAKED (5.5M)', args)
        ft_id1k = f'{args.emb_dir}/lewm_parabola_paperinit_id1k_parabola_emb_noproj.npy'
        if os.path.exists(ft_id1k):
            run_parabola(ft_id1k,
                         'LeWM parabola_paperinit FT 20ep ID-ONLY 1k (5.5M)', args)
        dit_id1k = f'{args.emb_dir}/dit_xl_lora_id1k_parabola_emb.npy'
        if os.path.exists(dit_id1k):
            run_parabola(dit_id1k,
                         'DiT-XL LoRA FT 8ep ID-ONLY 1k (749.8M)', args)


if __name__ == "__main__":
    main()
