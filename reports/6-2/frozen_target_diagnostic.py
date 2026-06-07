#!/usr/bin/env python3
"""Diagnostic: cos(pred, target) where pred is the AR rollout from the trained
ckpt, but target is computed by a FROZEN pusht encoder (not the ckpt's own
encoder). This breaks the self-similarity loop in the original metric.

Original metric (self-similarity loop):
    cos( trained_encoder(real_frame), trained_predictor(trained_encoder(hist), act) )

This diagnostic:
    cos( FROZEN_pusht_encoder(real_frame), trained_predictor(trained_encoder(hist), act) )

Interpretation:
  - High cos → predictor's output stays close to *pusht semantic space*
  - Low cos → predictor learned a representation orthogonal to pusht semantics
  - Relative ranking across (w, f) tells us which ckpts retained pusht-like info

Usage:
    python frozen_target_diagnostic.py --domain parabola --ckpt <name> [--gpu N]
"""
import argparse, sys, time, os, h5py, numpy as np, torch
from pathlib import Path

torch.set_grad_enabled(False)  # eval-only, no autograd

# Resolve project root (this file at <ROOT>/reports/6-2/frozen_target_diagnostic.py)
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'le-wm'))

_SWM = Path(os.environ.get('STABLEWM_HOME', str(Path.home() / '.stable_worldmodel')))
_DS = _SWM / 'datasets'
FROZEN_WEIGHTS = _SWM / 'lewm_paper_pusht' / 'weights.pt'

DOMAINS = {
    "collision": dict(train_h5=_DS/"phyworld_collision_id1k.h5",
                      eval_h5=_DS/"phyworld_collision_eval.h5",
                      src_hdf5=_ROOT/"phyworld/data/collision_eval.hdf5", ncol=4),
    "uniform_motion": dict(train_h5=_DS/"phyworld_uniform_motion_id1k.h5",
                           eval_h5=_DS/"phyworld_uniform_motion.h5",
                           src_hdf5=_ROOT/"phyworld/data/uniform_motion_eval.hdf5", ncol=2),
    "parabola": dict(train_h5=_DS/"phyworld_parabola_id1k.h5",
                     eval_h5=_DS/"phyworld_parabola.h5",
                     src_hdf5=_ROOT/"phyworld/data/parabola_eval.hdf5", ncol=2),
}
HS = 3
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pl_2col(r, v):
    r_ok = 0.7 <= abs(r) <= 1.5; v_ok = 1.0 <= abs(v) <= 4.0
    if r_ok and v_ok: return 0
    if not r_ok and v_ok: return 1
    if r_ok and not v_ok: return 2
    return 3

def pl_4col(m0, m1, v0, v1):
    m_ok = (0.7<=m0<=1.5) and (0.7<=m1<=1.5)
    v_ok = (1.0<=abs(v0)<=4.0) and (1.0<=abs(v1)<=4.0)
    if m_ok and v_ok: return 0
    if not m_ok and v_ok: return 1
    if m_ok and not v_ok: return 2
    return 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), required=True)
    ap.add_argument("--ckpt", required=True, help="trained ckpt path (the predictor/encoder we test)")
    ap.add_argument("--max-trajs", type=int, default=300)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = DOMAINS[args.domain]
    dev = torch.device('cuda')

    print(f"[ckpt] trained = {args.ckpt}")
    print(f"[frozen] target encoder = {FROZEN_WEIGHTS}")

    # action normalization
    with h5py.File(cfg["train_h5"], 'r') as f:
        a = np.nan_to_num(f['action'][:]).astype(np.float64)
        a_mean, a_std = a.mean(0), a.std(0) + 1e-8

    # --- Load trained ckpt ---
    trained = torch.load(args.ckpt, map_location='cpu', weights_only=False).to(dev).eval()

    # --- Build a frozen-encoder copy (same arch, pusht weights) ---
    # Use trained's structure, then load only encoder.* weights from pusht
    import copy
    frozen_for_target = copy.deepcopy(trained)
    pusht_sd = torch.load(FROZEN_WEIGHTS, map_location='cpu', weights_only=False)
    enc_only = {k: v for k, v in pusht_sd.items() if k.startswith('encoder.')}
    frozen_for_target.load_state_dict(enc_only, strict=False)
    frozen_for_target = frozen_for_target.to(dev).eval()
    # We only need its encoder; don't care about its predictor/projector

    # --- Load eval data ---
    f_eval = h5py.File(cfg["eval_h5"], 'r')
    pixels = f_eval['pixels']
    action = np.nan_to_num(f_eval['action'][:]).astype(np.float64)
    proprio = f_eval['proprio'][:]
    if cfg["ncol"] == 4:
        state = f_eval['state'][:]
    ep_idx = f_eval['episode_idx'][:]; step_idx = f_eval['step_idx'][:]

    # --- Partition labels per traj from src_hdf5 init_streams ---
    # (sourced from rollout_eval_id1k.py L101-107)
    with h5py.File(cfg["src_hdf5"], 'r') as f2:
        init = np.concatenate([f2['init_streams'][k][...] for k in sorted(f2['init_streams'])], 0)
    if cfg["ncol"] == 4:
        parts_arr = np.array([pl_4col(*init[i]) for i in range(len(init))], np.uint8)
    else:
        parts_arr = np.array([pl_2col(float(init[i,0]), float(init[i,1])) for i in range(len(init))], np.uint8)
    parts = {i: int(parts_arr[i]) for i in range(len(parts_arr))}

    # --- AR rollout ---
    # IMPORTANT: predictor lives in PROJECTOR space, not raw encoder space.
    # So we must use model.encode() (encoder + projector), NOT model.encoder() alone.
    def encode_frames(frames_u8, model):
        x = torch.from_numpy(frames_u8).permute(0,3,1,2).float().to(dev) / 255.0
        x = (x - IMNET_MEAN.to(dev)) / IMNET_STD.to(dev)
        info = {"pixels": x.unsqueeze(0)}     # (1, T, C, H, W)
        return model.encode(info)["emb"][0]   # (T, D) in projector space

    def ar_rollout(real_emb, act_norm):
        # Faithful copy of rollout_eval_id1k.py:ar_rollout — slide HS window correctly
        T = real_emb.size(0)
        emb_hist = real_emb[:HS].clone()
        act_emb_all = trained.action_encoder(act_norm.unsqueeze(0))[0]
        preds = []
        for k in range(HS, T):
            e_in = emb_hist[-HS:].unsqueeze(0)
            a_in = act_emb_all[k - HS:k].unsqueeze(0)
            p = trained.predict(e_in, a_in)[0, -1]
            preds.append(p)
            emb_hist = torch.cat([emb_hist, p.unsqueeze(0)], 0)
        return torch.stack(preds, 0)

    sel_eps = sorted(set(ep_idx.tolist()))[:args.max_trajs]
    t0 = time.time()
    real_E_trained, real_E_frozen, pred_E, meta = [], [], [], []
    n_done = 0
    for ep in sel_eps:
        rows = np.nonzero(ep_idx == ep)[0]
        rows = rows[np.argsort(step_idx[rows])]
        T = len(rows)
        if T <= HS + 1: continue
        if np.all(np.diff(rows) == 1):
            frames = pixels[rows[0]:rows[0]+T]
        else:
            frames = pixels[:][rows]
        real_emb_trained = encode_frames(frames, trained)
        real_emb_frozen = encode_frames(frames, frozen_for_target)

        raw_act = np.nan_to_num(action[rows]).astype(np.float64)
        act_norm = torch.from_numpy(((raw_act - a_mean) / a_std).astype(np.float32)).to(dev)
        pred = ar_rollout(real_emb_trained, act_norm)
        part = int(parts.get(ep, 0))
        re_t = real_emb_trained.cpu().numpy(); re_f = real_emb_frozen.cpu().numpy(); pe = pred.cpu().numpy()
        for j, k in enumerate(range(HS, T)):
            real_E_trained.append(re_t[k]); real_E_frozen.append(re_f[k]); pred_E.append(pe[j])
            meta.append((int(ep), k-HS+1, part))
        n_done += 1
        if n_done % 100 == 0:
            print(f"  rolled out {n_done} trajs, t={time.time()-t0:.0f}s", flush=True)
    f_eval.close()

    real_E_trained = np.array(real_E_trained)
    real_E_frozen = np.array(real_E_frozen)
    pred_E = np.array(pred_E)
    meta = np.array(meta)
    horizon = meta[:,1]; part_arr = meta[:,2]

    def cos_mean(R, P, mask):
        r = R[mask]; p = P[mask]
        c = (r*p).sum(1) / (np.linalg.norm(r,axis=1)*np.linalg.norm(p,axis=1)+1e-8)
        return c.mean()

    print(f"\n[data] {n_done} trajs, {len(pred_E)} pairs, emb dim={pred_E.shape[1]}")
    NAMES = {0:'ID', 1:'r/m-OOD', 2:'v-OOD', 3:'both-OOD'}

    print(f"\n=== cos by partition — TRAINED-target (original loop) vs FROZEN-target ===")
    print(f"  {'partition':<10} | trained-target | frozen-target |   Δ")
    print(f"  {'-'*10}-+----------------+---------------+--------")
    for p in range(4):
        m = (part_arr == p)
        if m.sum() < 50: continue
        c_t = cos_mean(real_E_trained, pred_E, m)
        c_f = cos_mean(real_E_frozen,  pred_E, m)
        print(f"  {NAMES[p]:<10} | {c_t:+.4f}        | {c_f:+.4f}      | {c_f-c_t:+.4f}")

    print(f"\n=== cos vs horizon — TRAINED-target vs FROZEN-target ===")
    print(f"  {'h':>3} | trained-target | frozen-target |   Δ")
    print(f"  {'-'*3}-+----------------+---------------+--------")
    for h in [1, 2, 4, 8, 16, 28]:
        m = (horizon == h)
        if m.sum() < 20: continue
        c_t = cos_mean(real_E_trained, pred_E, m)
        c_f = cos_mean(real_E_frozen,  pred_E, m)
        print(f"  {h:>3} | {c_t:+.4f}        | {c_f:+.4f}      | {c_f-c_t:+.4f}")

if __name__ == '__main__':
    main()
