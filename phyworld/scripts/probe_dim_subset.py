"""Probe position from DIMENSION SUBSETS of the latent.

Tests the mechanism claim "the black-box channel redundantly encodes position,
so prediction can route around a dedicated physical slot": if position is
decodable at high rho from the NON-slot dims (or any random subset) of a
shared latent, the slot is not load-bearing -- a bypass exists.

Encodes real frames only (no rollout); ridge-probes position from:
  all 192 / slot[0:2] / blackbox[2:192] / random-2 / random-10.
Reuse the loader+partition logic of rollout_eval_id1k.py.
"""
import argparse, os, sys, time, h5py, numpy as np, torch
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
_SWM = Path(os.environ.get('STABLEWM_HOME', str(Path.home() / '.stable_worldmodel')))
_DS = _SWM / 'datasets'
sys.path.insert(0, str(_ROOT / 'le-wm'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from rollout_eval_id1k import DOMAINS, pl_2col, pl_4col, NAMES, HS, IMNET_MEAN, IMNET_STD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-trajs", type=int, default=400)
    args = ap.parse_args()
    cfg = DOMAINS[args.domain]
    ckpt_path = args.ckpt or cfg["ckpt"]
    dev = 'cuda'
    t0 = time.time()

    model = torch.load(ckpt_path, map_location='cpu', weights_only=False).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    print(f"[ckpt] {ckpt_path} tag={args.tag}", flush=True)

    f = h5py.File(cfg["eval_h5"], 'r')
    pixels = f['pixels']; ep_idx = f['episode_idx'][:]; step_idx = f['step_idx'][:]
    proprio = f['proprio'][:]
    f2 = h5py.File(cfg["src_hdf5"], 'r')
    init = np.concatenate([f2['init_streams'][k][...] for k in sorted(f2['init_streams'])], 0)
    f2.close()
    if cfg["ncol"] == 4:
        parts = np.array([pl_4col(*init[i]) for i in range(len(init))], np.uint8)
    else:
        parts = np.array([pl_2col(float(init[i, 0]), float(init[i, 1])) for i in range(len(init))], np.uint8)

    uniq_eps = np.unique(ep_idx)
    rng = np.random.default_rng(0)
    sel_eps = rng.permutation(uniq_eps)[:args.max_trajs]
    n_tr = int(round(len(sel_eps) * 0.8))
    train_eps = set(sel_eps[:n_tr].tolist())

    @torch.no_grad()
    def encode_frames(frames_u8):
        x = torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float().to(dev) / 255.0
        x = (x - IMNET_MEAN.to(dev)) / IMNET_STD.to(dev)
        return model.encode({"pixels": x.unsqueeze(0)})["emb"][0]

    real_E, pos_all, part_l, intr_l = [], [], [], []
    for ep in sel_eps:
        rows = np.nonzero(ep_idx == ep)[0]; rows = rows[np.argsort(step_idx[rows])]
        T = len(rows)
        if T <= HS + 1: continue
        frames = pixels[rows[0]:rows[0] + T] if np.all(np.diff(rows) == 1) else pixels[:][rows]
        re = encode_frames(frames).cpu().numpy()
        pos_t = proprio[rows][:, [0, 2]] if cfg["ncol"] == 4 else proprio[rows][:, :2]
        in_tr = ep in train_eps
        for k in range(HS, T):
            real_E.append(re[k]); pos_all.append(pos_t[k]); part_l.append(int(parts[ep])); intr_l.append(in_tr)
    f.close()
    real_E = np.array(real_E); pos_all = np.array(pos_all)
    part_arr = np.array(part_l); in_train = np.array(intr_l, bool)
    D = real_E.shape[1]
    print(f"[data] {len(real_E)} frames, emb dim={D}, t={time.time()-t0:.0f}s", flush=True)

    rs = np.random.default_rng(42)
    subsets = {
        f"all[{D}]": np.arange(D),
        "slot[0:2]": np.array([0, 1]),
        f"blackbox[2:{D}]": np.arange(2, D),
        "rand-2": rs.choice(D, 2, replace=False),
        "rand-10": rs.choice(D, 10, replace=False),
    }
    print(f"\n=== decoded position rho by DIM SUBSET (ridge on REAL embs, test split) ===")
    print(f"    {'subset':16s} " + " ".join(f"{NAMES[p]:>9s}" for p in range(4)))
    for name, idx in subsets.items():
        Xtr = real_E[in_train][:, idx]; Xte = real_E[:, idx]
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
        rp = Ridge(alpha=1.0).fit((Xtr - mu) / sd, pos_all[in_train])
        pred = rp.predict((Xte - mu) / sd)
        cells = []
        for p in range(4):
            m = (~in_train) & (part_arr == p)
            if m.sum() < 5: cells.append("   n/a"); continue
            rhos = [pearsonr(pos_all[m][:, d], pred[m][:, d])[0] for d in range(pos_all.shape[1])]
            cells.append(f"{np.nanmean(rhos):+.3f}")
        print(f"    {name:16s} " + " ".join(f"{c:>9s}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
