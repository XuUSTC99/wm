"""Presence probe on Physion++: decode target 3D position from the model's REAL-frame
embeddings (Ridge fit on train trajs, applied to held-out REAL embs). This is the
Physion++ analog of the phyworld "REAL ... pos rho" table (presence side), NOT the
rolled-out/PRED side. No rollout is run. Reuses the encoding logic of
rollout_eval_physionpp.py verbatim.

Usage: python probe_real_emb_physionpp.py --ckpt <pp_fr ckpt> --device cuda:N
"""
import argparse, os, sys, time
from pathlib import Path
import h5py, numpy as np, torch
ROOT = Path("/home/likun-share/junjxu/wm"); sys.path.insert(0, str(ROOT / "le-wm"))
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
    args = ap.parse_args()
    dev = args.device; t0 = time.time()
    model = torch.load(args.ckpt, map_location="cpu", weights_only=False).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    f = h5py.File(args.h5, "r")
    ep_idx = f["episode_idx"][:]; step_idx = f["step_idx"][:]
    proprio = f["proprio"][:]; scene = f["scene_idx"][:]
    scene_names = f.attrs["scene_names"].split(","); pixels = f["pixels"]
    uniq = np.unique(ep_idx)
    rng = np.random.default_rng(0); perm = rng.permutation(uniq)
    sel = perm[: args.max_trajs] if args.max_trajs else perm
    n_tr = int(round(len(sel) * 0.8)); train_eps = set(sel[:n_tr].tolist())

    @torch.no_grad()
    def encode_frames(fr):
        x = torch.from_numpy(fr).permute(0, 3, 1, 2).float().to(dev) / 255.0
        x = (x - IMNET_MEAN.to(dev)) / IMNET_STD.to(dev)
        return model.encode({"pixels": x.unsqueeze(0)})["emb"][0]

    real_E, pos_all, scn, in_train = [], [], [], []
    nd = 0
    for ep in sel:
        rows = np.nonzero(ep_idx == ep)[0]; rows = rows[np.argsort(step_idx[rows])]
        T = len(rows)
        if T <= HS + 1:
            continue
        frames = pixels[rows[0]:rows[0] + T] if np.all(np.diff(rows) == 1) else pixels[:][rows]
        re = encode_frames(frames).cpu().numpy()
        sc = int(scene[rows[0]]); tr = ep in train_eps
        for k in range(HS, T):                       # match eval: frames HS..T-1
            real_E.append(re[k]); pos_all.append(proprio[rows][k]); scn.append(sc); in_train.append(tr)
        nd += 1
        if nd % 100 == 0:
            print(f"  {nd} trajs t={time.time()-t0:.0f}s", flush=True)
    f.close()
    real_E = np.array(real_E); pos_all = np.array(pos_all)
    scn = np.array(scn); in_train = np.array(in_train, bool)
    print(f"[data] {nd} trajs, {len(real_E)} frames, dim={real_E.shape[1]}", flush=True)

    mu = real_E[in_train].mean(0); sd = real_E[in_train].std(0) + 1e-6
    z = lambda x: (x - mu) / sd
    rp = Ridge(alpha=1.0).fit(z(real_E[in_train]), pos_all[in_train])   # fit on REAL train
    te = ~in_train
    ppred = rp.predict(z(real_E[te]))                                   # APPLY TO REAL held-out (presence)
    P = pos_all.shape[1]
    def rhos(mask):
        return [pearsonr(pos_all[te][mask][:, d], ppred[mask][:, d])[0] for d in range(P)]
    overall = rhos(np.ones(te.sum(), bool))
    print("\n=== PRESENCE: decode target pos from REAL held-out embs (pos0/1/2 rho) ===")
    print("  OVERALL (all scenes): " + " / ".join(f"{r:+.3f}" for r in overall))
    tescn = scn[te]
    for s, name in enumerate(scene_names):
        m = tescn == s
        if m.sum() < 5:
            continue
        print(f"  {name:22s} n={m.sum():>4}: " + " / ".join(f"{r:+.3f}" for r in rhos(m)))
    print(f"[done] {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
