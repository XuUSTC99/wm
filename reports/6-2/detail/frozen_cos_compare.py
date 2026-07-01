#!/usr/bin/env python3
"""Compare latent cos + MSE under THREE target encoders, to break the
self-similarity loop of the original rollout metric.

For a trained ckpt (its encoder E_tr + predictor P), and a frozen pusht
encoder E_fr (properly loaded via key remap):

  pred_k      = P autoregressively rolled out in E_tr's projector space
  real_tr_k   = E_tr.encode(real_frame_k)      # trained space  (original metric)
  real_fr_k   = E_fr.encode(real_frame_k)      # frozen pusht space

Variants reported (cos & nMSE, by horizon):
  1. trained-target : cos(pred_k, real_tr_k)              # circular (original)
  2. frozen-naive   : cos(pred_k, real_fr_k)              # cross-space, trend only
  3. frozen-aligned : cos(T @ pred_k, real_fr_k)          # T: Ridge E_tr->E_fr on ID
                       (breaks circularity AND fixes space mismatch)

Usage: frozen_cos_compare.py --domain parabola --ckpt <abs.ckpt> [--max-trajs 500]
"""
import argparse, sys, os, re, h5py, numpy as np, torch
from pathlib import Path
from sklearn.linear_model import Ridge

torch.set_grad_enabled(False)
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'le-wm'))
_SWM = Path(os.environ.get('STABLEWM_HOME', str(Path.home() / '.stable_worldmodel')))
_DS = _SWM / 'datasets'
FROZEN_WEIGHTS = _SWM / 'lewm_paper_pusht' / 'weights.pt'
HS = 3
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DOMAINS = {
    "parabola":       dict(train_h5=_DS/"phyworld_parabola_id1k.h5",       eval_h5=_DS/"phyworld_parabola.h5",        ncol=2),
    "uniform_motion": dict(train_h5=_DS/"phyworld_uniform_motion_id1k.h5", eval_h5=_DS/"phyworld_uniform_motion.h5",  ncol=2),
    "collision":      dict(train_h5=_DS/"phyworld_collision_id1k.h5",      eval_h5=_DS/"phyworld_collision_eval.h5",  ncol=4),
}


def remap(sd):
    sub = [(r"\.attention\.attention\.query\.", ".attention.q_proj."),
           (r"\.attention\.attention\.key\.", ".attention.k_proj."),
           (r"\.attention\.attention\.value\.", ".attention.v_proj."),
           (r"\.attention\.output\.dense\.", ".attention.o_proj."),
           (r"\.intermediate\.dense\.", ".mlp.fc1."),
           (r"\.output\.dense\.", ".mlp.fc2.")]
    out = {}
    for k, v in sd.items():
        nk = re.sub(r"^encoder\.encoder\.layer\.(\d+)\.", r"encoder.layers.\1.", k)
        for p, r in sub:
            nk = re.sub(p, r, nk)
        out[nk] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-trajs", type=int, default=500)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = DOMAINS[args.domain]
    dev = 'cuda'
    MEAN, STD = IMNET_MEAN.to(dev), IMNET_STD.to(dev)

    # action norm stats from id1k train set
    with h5py.File(cfg["train_h5"], 'r') as f:
        a = np.nan_to_num(f['action'][:]).astype(np.float64)
        a_mean, a_std = a.mean(0), a.std(0) + 1e-8

    trained = torch.load(args.ckpt, map_location='cpu', weights_only=False).to(dev).eval()
    # frozen pusht: deepcopy + load encoder.+projector. from remapped pusht
    import copy
    frozen = copy.deepcopy(trained)
    pusht = remap(torch.load(FROZEN_WEIGHTS, map_location='cpu', weights_only=False))
    encproj = {k: v for k, v in pusht.items() if k.startswith(('encoder.', 'projector.'))}
    miss, unexp = frozen.load_state_dict(encproj, strict=False)
    n_enc = sum(1 for k in encproj if k.startswith('encoder.'))
    n_enc_loaded = n_enc - sum(1 for k in unexp if k.startswith('encoder.'))
    print(f"[frozen] loaded {len(encproj)-len(unexp)}/{len(encproj)} (encoder {n_enc_loaded}/{n_enc})", flush=True)
    assert n_enc_loaded >= 0.9 * n_enc, "frozen encoder failed to load — naming mismatch!"
    frozen = frozen.to(dev).eval()

    def enc(model, frames):
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(dev) / 255.0
        x = (x - MEAN) / STD
        return model.encode({"pixels": x.unsqueeze(0)})["emb"][0]  # (T,D)

    def ar_rollout(real_emb, act_norm):
        T = real_emb.size(0)
        emb_hist = real_emb[:HS].clone()
        act_all = trained.action_encoder(act_norm.unsqueeze(0))[0]
        preds = []
        for k in range(HS, T):
            e_in = emb_hist[-HS:].unsqueeze(0)
            a_in = act_all[k - HS:k].unsqueeze(0)
            p = trained.predict(e_in, a_in)[0, -1]
            preds.append(p); emb_hist = torch.cat([emb_hist, p.unsqueeze(0)], 0)
        return torch.stack(preds, 0)

    f_eval = h5py.File(cfg["eval_h5"], 'r')
    pixels = f_eval['pixels']; action = np.nan_to_num(f_eval['action'][:]).astype(np.float64)
    ep_idx = f_eval['episode_idx'][:]; step_idx = f_eval['step_idx'][:]
    sel = sorted(set(ep_idx.tolist()))[:args.max_trajs]
    train_eps = set(sorted(set(ep_idx.tolist()))[:int(0.5*len(sel))])  # arbitrary ID split for Ridge fit

    real_tr, real_fr, pred, hor, intr = [], [], [], [], []
    for ep in sel:
        rows = np.nonzero(ep_idx == ep)[0]; rows = rows[np.argsort(step_idx[rows])]
        T = len(rows)
        if T <= HS + 1: continue
        frames = pixels[rows[0]:rows[0]+T] if np.all(np.diff(rows) == 1) else pixels[:][rows]
        rt = enc(trained, frames); rf = enc(frozen, frames)
        raw = np.nan_to_num(action[rows]).astype(np.float64)
        an = torch.from_numpy(((raw - a_mean)/a_std).astype(np.float32)).to(dev)
        pe = ar_rollout(rt, an)
        rt = rt.cpu().numpy(); rf = rf.cpu().numpy(); pe = pe.cpu().numpy()
        for j, k in enumerate(range(HS, T)):
            real_tr.append(rt[k]); real_fr.append(rf[k]); pred.append(pe[j])
            hor.append(k-HS+1); intr.append(ep in train_eps)
    f_eval.close()
    real_tr = np.array(real_tr); real_fr = np.array(real_fr); pred = np.array(pred)
    hor = np.array(hor); intr = np.array(intr)

    # Ridge T: trained-space real emb -> frozen-space real emb, fit on ID-train frames
    T_map = Ridge(alpha=1.0).fit(real_tr[intr], real_fr[intr])
    pred_aligned = T_map.predict(pred)

    def cos_nmse(R, P, mask):
        r, p = R[mask], P[mask]
        cos = (r*p).sum(1)/(np.linalg.norm(r,axis=1)*np.linalg.norm(p,axis=1)+1e-8)
        nmse = ((p-r)**2).sum(1) / (((r-r.mean(0))**2).sum(1).mean()+1e-8)
        return cos.mean(), nmse.mean()

    print(f"\n[data] {len(pred)} pairs, dim={pred.shape[1]}, tag={args.tag}")
    print(f"\n{'h':>3} | {'trained-tgt':^16} | {'frozen-naive':^16} | {'frozen-aligned':^16}")
    print(f"{'':>3} | {'cos':>7} {'nMSE':>8} | {'cos':>7} {'nMSE':>8} | {'cos':>7} {'nMSE':>8}")
    print('-'*60)
    for h in [1, 2, 4, 8, 16, 28]:
        m = (hor == h)
        if m.sum() < 20: continue
        ct, nt = cos_nmse(real_tr, pred, m)
        cn, nn = cos_nmse(real_fr, pred, m)
        ca, na = cos_nmse(real_fr, pred_aligned, m)
        print(f"{h:>3} | {ct:+.4f} {nt:7.3f} | {cn:+.4f} {nn:7.3f} | {ca:+.4f} {na:7.3f}")


if __name__ == '__main__':
    main()
