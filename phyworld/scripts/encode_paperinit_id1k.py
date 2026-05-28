"""Encode phyworld eval h5 with ID-only-trained LeWM FT ckpts.

ID-only FT trained on PhyWorld official ID training set (1000 trajs from *_30K.hdf5).
Probing happens on the eval h5 (containing all 4 OOD partitions).
This measures TRUE ID→OOD zero-shot generalization.
"""
import argparse, os, sys, time, h5py, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path('/home/qlib/agent_memory/wm/le-wm')))
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

DOMAINS = {
    "collision": {
        "ckpt": "/home/qlib/.stable_worldmodel/collision_paperinit_id1k/lewm_collision_paperinit_id1k_epoch_20_object.ckpt",
        "emb_out": "/home/qlib/agent_memory/wm/artifacts/embeddings/lewm_collision_paperinit_id1k_collision_eval_emb_52k_noproj.npy",
        "lewm_h5": "/home/qlib/.stable_worldmodel/phyworld_collision_eval.h5",
        "src_hdf5": "/home/qlib/agent_memory/wm/phyworld/data/collision_eval.hdf5",
    },
    "uniform_motion": {
        "ckpt": "/home/qlib/.stable_worldmodel/uniform_paperinit_id1k/lewm_uniform_paperinit_id1k_epoch_20_object.ckpt",
        "emb_out": "/home/qlib/agent_memory/wm/artifacts/embeddings/lewm_uniform_paperinit_id1k_uniform_motion_emb_37k_noproj.npy",
        "lewm_h5": "/home/qlib/.stable_worldmodel/phyworld_uniform_motion.h5",
        "src_hdf5": "/home/qlib/agent_memory/wm/phyworld/data/uniform_motion_eval.hdf5",
    },
    "parabola": {
        "ckpt": "/home/qlib/.stable_worldmodel/parabola_paperinit_id1k/lewm_parabola_paperinit_id1k_epoch_20_object.ckpt",
        "emb_out": "/home/qlib/agent_memory/wm/artifacts/embeddings/lewm_parabola_paperinit_id1k_parabola_emb_noproj.npy",
        "lewm_h5": "/home/qlib/.stable_worldmodel/phyworld_parabola.h5",
        "src_hdf5": "/home/qlib/agent_memory/wm/phyworld/data/parabola_eval.hdf5",
    },
}

ap = argparse.ArgumentParser()
ap.add_argument("--domain", choices=list(DOMAINS), required=True)
args = ap.parse_args()
cfg = DOMAINS[args.domain]
t0 = time.time()

# ---- encode ----
if not os.path.exists(cfg["emb_out"]):
    print(f'[load] ckpt {cfg["ckpt"]}', flush=True)
    model = torch.load(cfg["ckpt"], map_location='cpu', weights_only=False)
    print(f'  loaded {type(model).__name__}')
    f = h5py.File(cfg["lewm_h5"], 'r')
    pix = f['pixels'][:]; f.close()
    pix_chw = torch.from_numpy(pix).permute(0, 3, 1, 2).contiguous()
    encoder = model.encoder.eval().to('cuda')
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    out = []
    bs = 128
    for i in range(0, pix_chw.shape[0], bs):
        x = pix_chw[i:i + bs].to('cuda', non_blocking=True).float() / 255.0
        x = (x - mean) / std
        with torch.no_grad():
            h = encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        out.append(h.float().cpu().numpy())
        if (i // bs) % 50 == 0:
            print(f'  encoded {i + x.shape[0]}/{pix_chw.shape[0]}', flush=True)
    emb = np.concatenate(out, 0)
    np.save(cfg["emb_out"], emb)
    print(f'  cached to {cfg["emb_out"]}, t={time.time()-t0:.1f}s')
else:
    emb = np.load(cfg["emb_out"])
    print(f'[cache] loaded {emb.shape}')

# ---- probe (Ridge K=4 mixed-fit) ----
f1 = h5py.File(cfg["lewm_h5"], 'r')
ep_idx = f1['episode_idx'][:]; step_idx = f1['step_idx'][:]
if args.domain == 'collision':
    state = f1['state'][:]   # (N, 4) -> vx1, vy1, vx2, vy2
    proprio = f1['proprio'][:]  # (N, 4) -> x1, y1, x2, y2
    pos_target = proprio[:, [0, 2]]   # pos_x1, pos_x2
    vel_target = state[:, [0, 2]]     # vx1, vx2
    pos_names = ['pos_x1', 'pos_x2']
    vel_names = ['vx1', 'vx2']
else:
    prop = f1['proprio'][:]; action = f1['action'][:]
    pos_target = prop[:, :2].astype(np.float32)
    vel_target = action[:, :2].astype(np.float32)
    pos_names = ['pos_x', 'pos_y']
    vel_names = ['vx', 'vy']
f1.close()

# Partition assignment
f2 = h5py.File(cfg["src_hdf5"], 'r')
init = np.concatenate([f2['init_streams'][k][...] for k in sorted(f2['init_streams'])], 0)
f2.close()

def pl_2col(r, v):
    r_ok = 0.7 <= abs(r) <= 1.5; v_ok = 1.0 <= abs(v) <= 4.0
    if r_ok and v_ok: return 0
    if not r_ok and v_ok: return 1
    if r_ok and not v_ok: return 2
    return 3

def pl_4col(m0, m1, v0, v1):
    """Collision: ID = m∈[0.7,1.5]^2 AND v∈[1,4]^2."""
    m_ok = (0.7 <= m0 <= 1.5) and (0.7 <= m1 <= 1.5)
    v_ok = (1.0 <= abs(v0) <= 4.0) and (1.0 <= abs(v1) <= 4.0)
    if m_ok and v_ok: return 0
    if not m_ok and v_ok: return 1
    if m_ok and not v_ok: return 2
    return 3

if args.domain == 'collision':
    parts = np.array([pl_4col(*init[i]) for i in range(len(init))], dtype=np.uint8)
else:
    parts = np.array([pl_2col(float(init[i, 0]), float(init[i, 1])) for i in range(len(init))], dtype=np.uint8)
part = parts[ep_idx]
NAMES = {0: 'ID', 1: 'r/m-OOD', 2: 'v-OOD', 3: 'both-OOD'}

def build_k4(emb, ep_idx, step_idx):
    N, D = emb.shape
    feats = np.zeros((N, 4 * D), dtype=np.float32); valid = np.zeros(N, dtype=bool)
    by_ep = {}
    for i in range(N): by_ep.setdefault(int(ep_idx[i]), []).append((int(step_idx[i]), i))
    for ep, lst in by_ep.items():
        lst.sort(); ordered = [idx for _, idx in lst]
        for k, idx in enumerate(ordered):
            if k >= 3:
                ctx = ordered[k - 3:k + 1]
                feats[idx] = np.concatenate([emb[j] for j in ctx]); valid[idx] = True
    return feats, valid

feats, valid = build_k4(emb, ep_idx, step_idx)
rng = np.random.default_rng(0)
uniq = np.unique(ep_idx); perm = rng.permutation(uniq)
n_tr = int(round(len(uniq) * 0.8))
train_eps = set(perm[:n_tr].tolist())
base_tr = np.array([e in train_eps for e in ep_idx])
mask_tr = base_tr & valid; mask_te = (~base_tr) & valid
mu, sigma = feats[mask_tr].mean(0), feats[mask_tr].std(0) + 1e-6
e_tr = (feats[mask_tr] - mu) / sigma

# Print probe-train partition mix
print(f"\n  probe train mix: {dict(zip(*np.unique(part[mask_tr], return_counts=True)))} (NAMES={NAMES})")

# train stds
pos_std = pos_target[mask_tr].std(0) + 1e-8
vel_std = vel_target[mask_tr].std(0) + 1e-8

rp = Ridge(alpha=1.0); rp.fit(e_tr, pos_target[mask_tr])
rv = Ridge(alpha=1.0); rv.fit(e_tr, vel_target[mask_tr])

print(f'\n=== LeWM {args.domain} ID-only FT (K=4 mixed-fit Ridge, true ID→OOD) ===')
hdr = f' {"partition":12s}  {"n":>5s}  '
for nm in pos_names + vel_names:
    hdr += f'{nm+" nMSE":>10s}  {nm+" ρ":>8s}  '
print(hdr)

rows = [('AGGREGATE', mask_te)]
for p in range(4):
    m = mask_te & (part == p)
    if m.sum() >= 50: rows.append((NAMES[p], m))

for name, m in rows:
    e_p = (feats[m] - mu) / sigma
    pp = rp.predict(e_p); pv = rv.predict(e_p)
    line = f' {name:12s}  {m.sum():>5d}  '
    for d in range(2):
        nmse = float((((pp[:, d] - pos_target[m, d]) / pos_std[d]) ** 2).mean())
        rho = float(pearsonr(pos_target[m, d], pp[:, d])[0])
        line += f'{nmse:>10.4f}  {rho:>+8.4f}  '
    for d in range(2):
        nmse = float((((pv[:, d] - vel_target[m, d]) / vel_std[d]) ** 2).mean())
        rho = float(pearsonr(vel_target[m, d], pv[:, d])[0])
        line += f'{nmse:>10.4f}  {rho:>+8.4f}  '
    print(line)

print(f'\nTotal {time.time()-t0:.1f}s')
