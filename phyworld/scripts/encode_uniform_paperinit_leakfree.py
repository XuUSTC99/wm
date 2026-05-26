"""Encode uniform_motion with leak-free FT ckpt + MSE/ρ probe.
Run after training completes."""
import sys, os, h5py, numpy as np, torch, time
from pathlib import Path
sys.path.insert(0, str(Path('/home/qlib/agent_memory/wm/le-wm')))
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

CKPT = '/home/qlib/.stable_worldmodel/uniform_paperinit_leakfree/lewm_uniform_paperinit_leakfree_epoch_20_object.ckpt'
EMB_CACHE = '/home/qlib/agent_memory/wm/artifacts/embeddings/lewm_uniform_paperinit_leakfree_uniform_motion_emb_37k_noproj.npy'
LEWM_DATA = '/home/qlib/.stable_worldmodel/phyworld_uniform_motion.h5'
SRC_DATA = '/home/qlib/agent_memory/wm/phyworld/data/uniform_motion_eval.hdf5'

t0 = time.time()
if not os.path.exists(EMB_CACHE):
    print(f'[load] ckpt {CKPT}', flush=True)
    model = torch.load(CKPT, map_location='cpu', weights_only=False)
    print(f'  loaded {type(model).__name__}')
    f = h5py.File(LEWM_DATA, 'r')
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
    np.save(EMB_CACHE, emb)
    print(f'  cached to {EMB_CACHE}, t={time.time()-t0:.1f}s')
else:
    emb = np.load(EMB_CACHE)
    print(f'[cache] loaded {emb.shape}')

# probe
f1 = h5py.File(LEWM_DATA, 'r')
prop = f1['proprio'][:]; action = f1['action'][:]
ep_idx = f1['episode_idx'][:]; step_idx = f1['step_idx'][:]
f1.close()
f2 = h5py.File(SRC_DATA, 'r')
init = np.concatenate([f2['init_streams'][k][...] for k in sorted(f2['init_streams'])], 0)
f2.close()

def pl(r, v):
    r_ok = 0.7 <= r <= 1.5; v_ok = 1.0 <= abs(v) <= 4.0
    if r_ok and v_ok: return 0
    if not r_ok and v_ok: return 1
    if r_ok and not v_ok: return 2
    return 3
parts = np.array([pl(float(init[i, 0]), float(init[i, 1])) for i in range(len(init))], dtype=np.uint8)
part = parts[ep_idx]
NAMES = {0: 'ID', 1: 'r-OOD', 2: 'v-OOD', 3: 'both-OOD'}

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

def mse_rho(true, pred):
    t = true.ravel(); p = pred.ravel()
    return float(((p - t) ** 2).mean()), float(pearsonr(t, p)[0])

feats, valid = build_k4(emb, ep_idx, step_idx)
rng = np.random.default_rng(0)
uniq = np.unique(ep_idx); perm = rng.permutation(uniq)
n_tr = int(round(len(uniq) * 0.8))
train_eps = set(perm[:n_tr].tolist())
base_tr = np.array([e in train_eps for e in ep_idx])
mask_tr = base_tr & valid; mask_te = (~base_tr) & valid
mu, sigma = feats[mask_tr].mean(0), feats[mask_tr].std(0) + 1e-6
e_tr = (feats[mask_tr] - mu) / sigma

# Sanity check: train_eps from probe should match FT train_eps
ft_train = set(np.load('/home/qlib/.stable_worldmodel/uniform_train_eps.npy').tolist())
overlap = train_eps & ft_train
print(f'\n[SANITY] probe train_eps ∩ FT train_eps = {len(overlap)} / {len(train_eps)} (should be {len(train_eps)})')
ft_test = set(np.load('/home/qlib/.stable_worldmodel/uniform_test_eps.npy').tolist())
test_eps = set(perm[n_tr:].tolist())
overlap_test = test_eps & ft_test
overlap_te_with_ft_train = test_eps & ft_train
print(f'[SANITY] probe test_eps ∩ FT held-out = {len(overlap_test)} / {len(test_eps)} (should be {len(test_eps)})')
print(f'[SANITY] probe test_eps ∩ FT train_eps = {len(overlap_te_with_ft_train)} (must be 0 — this is the leak check)')
print()

pos_x_1d = prop[:, 0]; vx_1d = action[:, 0]
rp = Ridge(alpha=1.0); rp.fit(e_tr, pos_x_1d[mask_tr])
rv = Ridge(alpha=1.0); rv.fit(e_tr, vx_1d[mask_tr])

print(f'\n=== LeWM uniform_paperinit FT LEAK-FREE (K=4 mixed-fit) ===')
print(f' {"partition":12s}  {"n":>5s}  {"pos_x MSE":>10s}  {"pos_x ρ":>8s}  {"vx MSE":>10s}  {"vx ρ":>8s}')
rows = [('AGGREGATE', mask_te)]
for p in range(4):
    m = mask_te & (part == p)
    if m.sum() >= 50: rows.append((NAMES[p], m))
for name, m in rows:
    e_p = (feats[m] - mu) / sigma
    pp = rp.predict(e_p); pv = rv.predict(e_p)
    mse_p, rho_p = mse_rho(pos_x_1d[m], pp)
    mse_v, rho_v = mse_rho(vx_1d[m], pv)
    print(f' {name:12s}  {m.sum():>5d}  {mse_p:>10.4f}  {rho_p:>+8.4f}  {mse_v:>10.4f}  {rho_v:>+8.4f}')

print(f'\nTotal {time.time()-t0:.1f}s')
