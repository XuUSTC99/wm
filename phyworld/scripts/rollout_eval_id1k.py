"""AR rollout eval for LeWM ID-only FT models.

Tests the FORWARD DYNAMICS (encoder + ARPredictor), not just state encoding:
  - encode first HS=3 real frames as context
  - autoregressively roll out predictor with TRUE (normalized) actions
  - measure (a) latent prediction quality vs real embeddings per horizon,
            (b) decoded pos/vel ρ from rolled-out latents per partition.

Action normalization MUST match FT (mean/std from the ID-only id1k training h5).
Predictor operates in projector(encoder(x)) space — probe is trained there too.
"""
import argparse, os, sys, time, h5py, numpy as np, torch
from pathlib import Path

# Resolve project root: this file is at <ROOT>/phyworld/scripts/rollout_eval_id1k.py
_ROOT = Path(__file__).resolve().parents[2]
_SWM = Path(os.environ.get('STABLEWM_HOME', str(Path.home() / '.stable_worldmodel')))
# Datasets in new stable_worldmodel layout live under <SWM>/datasets/<name>.h5
_DS = _SWM / 'datasets'
_RAW = Path(os.environ.get('PHYWORLD_RAW', str(_SWM.parent / 'phyworld_raw')))
sys.path.insert(0, str(_ROOT / 'le-wm'))
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

DOMAINS = {
    "collision": {
        "ckpt": str(_SWM / "collision_paperinit_id1k/lewm_collision_paperinit_id1k_epoch_20_object.ckpt"),
        "train_h5": str(_DS / "phyworld_collision_id1k.h5"),
        "eval_h5": str(_DS / "phyworld_collision_eval.h5"),
        "src_hdf5": str(_RAW / "collision_eval.hdf5"),
        "ncol": 4,
    },
    "uniform_motion": {
        "ckpt": str(_SWM / "uniform_paperinit_id1k/lewm_uniform_paperinit_id1k_epoch_20_object.ckpt"),
        "train_h5": str(_DS / "phyworld_uniform_motion_id1k.h5"),
        "eval_h5": str(_DS / "phyworld_uniform_motion.h5"),
        "src_hdf5": str(_RAW / "uniform_motion_eval.hdf5"),
        "ncol": 2,
    },
    "parabola": {
        "ckpt": str(_SWM / "parabola_paperinit_id1k/lewm_parabola_paperinit_id1k_epoch_20_object.ckpt"),
        "train_h5": str(_DS / "phyworld_parabola_id1k.h5"),
        "eval_h5": str(_DS / "phyworld_parabola.h5"),
        "src_hdf5": str(_RAW / "parabola_eval.hdf5"),
        "ncol": 2,
    },
}
HS = 3            # history_size (matches wm.history_size)
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pl_2col(r, v):
    r_ok = 0.7 <= abs(r) <= 1.5; v_ok = 1.0 <= abs(v) <= 4.0
    if r_ok and v_ok: return 0
    if not r_ok and v_ok: return 1
    if r_ok and not v_ok: return 2
    return 3

def pl_4col(m0, m1, v0, v1):
    m_ok = (0.7 <= m0 <= 1.5) and (0.7 <= m1 <= 1.5)
    v_ok = (1.0 <= abs(v0) <= 4.0) and (1.0 <= abs(v1) <= 4.0)
    if m_ok and v_ok: return 0
    if not m_ok and v_ok: return 1
    if m_ok and not v_ok: return 2
    return 3

NAMES = {0: 'ID', 1: 'r/m-OOD', 2: 'v-OOD', 3: 'both-OOD'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), required=True)
    ap.add_argument("--max-trajs", type=int, default=400, help="cap eval trajs for speed")
    ap.add_argument("--ckpt", default=None, help="override ckpt (e.g. +probe model); norm/data still per-domain")
    ap.add_argument("--tag", default="", help="label for this run in the header")
    ap.add_argument("--use-action", action="store_true",
                    help="privileged legacy upper bound; default is leak-free action-free rollout")
    args = ap.parse_args()
    cfg = DOMAINS[args.domain]
    ckpt_path = args.ckpt or cfg["ckpt"]
    dev = 'cuda'
    t0 = time.time()
    gate_scores = []
    gate_horizons = []

    # ---- action norm stats from ID-only TRAIN h5 (must match FT) ----
    with h5py.File(cfg["train_h5"], 'r') as f:
        tr_act = f['action'][:].astype(np.float64)
    a_mean = np.nan_to_num(tr_act).mean(0)
    a_std = np.nan_to_num(tr_act).std(0) + 1e-8
    print(f"[norm] action mean={a_mean}, std={a_std}  (from {cfg['train_h5'].split('/')[-1]})", flush=True)
    print(f"[ckpt] {ckpt_path}  {('tag='+args.tag) if args.tag else ''}", flush=True)

    # ---- load model ----
    model = torch.load(ckpt_path, map_location='cpu', weights_only=False).to(dev).eval()
    if hasattr(model, "use_action"):
        model.use_action = bool(args.use_action)
    for p in model.parameters(): p.requires_grad_(False)

    # ---- load eval data ----
    f = h5py.File(cfg["eval_h5"], 'r')
    pixels = f['pixels']; ep_idx = f['episode_idx'][:]; step_idx = f['step_idx'][:]
    proprio = f['proprio'][:]; action = f['action'][:]
    state = f['state'][:] if 'state' in f else None

    # partition per episode
    f2 = h5py.File(cfg["src_hdf5"], 'r')
    init = np.concatenate([f2['init_streams'][k][...] for k in sorted(f2['init_streams'])], 0)
    f2.close()
    if cfg["ncol"] == 4:
        parts = np.array([pl_4col(*init[i]) for i in range(len(init))], np.uint8)
    else:
        parts = np.array([pl_2col(float(init[i, 0]), float(init[i, 1])) for i in range(len(init))], np.uint8)

    # group rows by episode
    uniq_eps = np.unique(ep_idx)
    rng = np.random.default_rng(0)
    perm = rng.permutation(uniq_eps)
    # cap eval trajs first, THEN split those into probe-train/test (80/20)
    sel_eps = perm[:args.max_trajs] if args.max_trajs else perm
    n_tr = int(round(len(sel_eps) * 0.8))
    train_eps = set(sel_eps[:n_tr].tolist())

    @torch.no_grad()
    def encode_frames(frames_u8):
        # frames_u8: (T, H, W, 3) uint8  -> projector-space embs (T, D)
        x = torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float().to(dev) / 255.0
        x = (x - IMNET_MEAN.to(dev)) / IMNET_STD.to(dev)
        info = {"pixels": x.unsqueeze(0)}  # (1, T, C, H, W)
        return model.encode(info)["emb"][0]  # (T, D)

    @torch.no_grad()
    def ar_rollout(real_emb, act_norm):
        # real_emb: (T, D) tensor; act_norm: (T, A) tensor (normalized)
        T = real_emb.size(0)
        emb_hist = real_emb[:HS].clone()           # (HS, D)
        if args.use_action:
            act_emb_all = model.action_encoder(act_norm.unsqueeze(0))[0]
        else:
            act_emb_all = torch.zeros(T, real_emb.size(-1), device=real_emb.device)
        preds = []
        for k in range(HS, T):
            e_in = emb_hist[-HS:].unsqueeze(0)
            a_in = act_emb_all[k - HS:k].unsqueeze(0)
            step = k - HS + 1
            gipp = getattr(model, "gipp", None)
            shadow = bool(gipp is not None and getattr(gipp, "shadow", False))
            p = model.predict(e_in, a_in, rollout_step=step, apply_gipp=not shadow)[0, -1]
            p_memory = gipp(p, e_in, rollout_step=step).squeeze(0) if shadow else p
            if gipp is not None and hasattr(gipp, "last_gate_score"):
                gate_scores.extend(gipp.last_gate_score.float().cpu().reshape(-1).tolist())
                gate_horizons.extend([step] * gipp.last_gate_score.numel())
            preds.append(p)
            emb_hist = torch.cat([emb_hist, p_memory.unsqueeze(0)], 0)
        return torch.stack(preds, 0)

    # ---- pass 1: encode all selected trajs, collect real & predicted embs ----
    real_E, pred_E, meta = [], [], []  # meta: (ep, frame_k, partition, in_train)
    pos_all, vel_all = [], []
    n_done = 0
    for ep in sel_eps:
        rows = np.nonzero(ep_idx == ep)[0]
        rows = rows[np.argsort(step_idx[rows])]
        T = len(rows)
        if T <= HS + 1:
            continue
        frames = pixels[rows[0]:rows[0] + T] if np.all(np.diff(rows) == 1) else pixels[:][rows]
        real_emb = encode_frames(frames)  # (T, D)
        raw_act = np.nan_to_num(action[rows]).astype(np.float64)
        if args.use_action:
            act_np = ((raw_act - a_mean) / a_std).astype(np.float32)
        else:
            act_np = np.zeros_like(raw_act, dtype=np.float32)
        act_norm = torch.from_numpy(act_np).to(dev)
        pred = ar_rollout(real_emb, act_norm)  # (T-HS, D)

        part = int(parts[ep])
        in_tr = ep in train_eps
        re = real_emb.cpu().numpy(); pe = pred.cpu().numpy()
        # pos/vel targets
        if cfg["ncol"] == 4:
            pos_t = proprio[rows][:, [0, 2]]; vel_t = state[rows][:, [0, 2]]
        else:
            pos_t = proprio[rows][:, :2]
            vel_t = np.empty_like(pos_t)
            vel_t[:-1] = pos_t[1:] - pos_t[:-1]
            vel_t[-1] = vel_t[-2]
        for j, k in enumerate(range(HS, T)):
            real_E.append(re[k]); pred_E.append(pe[j])
            pos_all.append(pos_t[k]); vel_all.append(vel_t[k])
            meta.append((int(ep), k - HS + 1, part, in_tr))  # horizon h = k-HS+1
        n_done += 1
        if n_done % 100 == 0:
            print(f"  rolled out {n_done} trajs, t={time.time()-t0:.0f}s", flush=True)
    f.close()

    real_E = np.array(real_E); pred_E = np.array(pred_E)
    pos_all = np.array(pos_all); vel_all = np.array(vel_all)
    meta = np.array(meta)  # (N, 4): ep, horizon, part, in_train
    horizon = meta[:, 1]; part_arr = meta[:, 2]; in_train = meta[:, 3].astype(bool)
    print(f"\n[data] {n_done} trajs, {len(real_E)} (frame,pred) pairs, emb dim={real_E.shape[1]}", flush=True)

    # ---- train probe on REAL embs (train trajs), apply to predicted embs ----
    # K=1 single-frame Ridge in projector space
    mu = real_E[in_train].mean(0); sd = real_E[in_train].std(0) + 1e-6
    def z(x): return (x - mu) / sd
    rp = Ridge(alpha=1.0).fit(z(real_E[in_train]), pos_all[in_train])
    rv = Ridge(alpha=1.0).fit(z(real_E[in_train]), vel_all[in_train])

    pos_std = pos_all[in_train].std(0) + 1e-8
    vel_std = vel_all[in_train].std(0) + 1e-8

    def report_decoded(tag, X, mask):
        pp = rp.predict(z(X[mask])); pv = rv.predict(z(X[mask]))
        line = f"  {tag:24s}"
        for d in range(pos_all.shape[1]):
            rho = pearsonr(pos_all[mask][:, d], pp[:, d])[0]
            line += f"  pos{d}ρ={rho:+.3f}"
        for d in range(vel_all.shape[1]):
            rho = pearsonr(vel_all[mask][:, d], pv[:, d])[0]
            line += f"  vel{d}ρ={rho:+.3f}"
        print(line)

    te = ~in_train  # report on held-out trajs only

    # latent fidelity: cosine sim + nMSE between pred & real emb
    def latent_stats(mask):
        r = real_E[mask]; p = pred_E[mask]
        cos = (r * p).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(p, axis=1) + 1e-8)
        nmse = ((p - r) ** 2).sum(1) / ((r - r.mean(0)) ** 2).sum(1).mean()
        return cos.mean(), nmse.mean()

    protocol = "privileged-action" if args.use_action else "action-free"
    print(f"\n=== {args.domain} AR ROLLOUT ({protocol}, history_size={HS}) ===")
    print(f"--- latent fidelity (pred vs real emb), test trajs, by partition ---")
    for p in range(4):
        m = te & (part_arr == p)
        if m.sum() < 50: continue
        cos, nmse = latent_stats(m)
        print(f"  {NAMES[p]:10s} n={m.sum():5d}  cos={cos:+.4f}  nMSE={nmse:.4f}")

    print(f"\n--- latent fidelity vs horizon (test, aggregate) ---")
    for h in [1, 2, 4, 8, 16, 28]:
        m = te & (horizon == h)
        if m.sum() < 20: continue
        cos, nmse = latent_stats(m)
        print(f"  h={h:3d}  n={m.sum():5d}  cos={cos:+.4f}  nMSE={nmse:.4f}")

    print(f"\n--- decoded pos/vel ρ from ROLLED-OUT latents (test), by partition ---")
    print(f"  [baseline] probe applied to REAL embs:")
    for p in range(4):
        m = te & (part_arr == p)
        if m.sum() < 50: continue
        report_decoded(f"REAL {NAMES[p]}", real_E, m)
    print(f"  [rollout] probe applied to PREDICTED embs:")
    for p in range(4):
        m = te & (part_arr == p)
        if m.sum() < 50: continue
        report_decoded(f"PRED {NAMES[p]}", pred_E, m)

    print(f"\n--- decoded vel ρ from PRED latents vs horizon (test, aggregate) ---")
    for h in [1, 2, 4, 8, 16, 28]:
        m = te & (horizon == h)
        if m.sum() < 20: continue
        pv = rv.predict(z(pred_E[m]))
        rhos = [pearsonr(vel_all[m][:, d], pv[:, d])[0] for d in range(vel_all.shape[1])]
        print(f"  h={h:3d}  n={m.sum():5d}  " + "  ".join(f"vel{d}ρ={r:+.3f}" for d, r in enumerate(rhos)))

    # ==================== K=4 decode (stack 4 consecutive rolled-out latents) ====================
    # Rows are appended per-episode in frame order, so within an episode consecutive
    # rows = consecutive frames. K=4 feature = concat of the row's emb + its 3 predecessors
    # in the SAME episode. Real K=4 trains the probe; predicted K=4 is decoded.
    KW = 4
    ep_arr = meta[:, 0]
    order = np.lexsort((horizon, ep_arr))  # sort by (ep, horizon)
    def build_k4(E):
        F = np.zeros((len(E), KW * E.shape[1]), dtype=np.float32)
        valid = np.zeros(len(E), dtype=bool)
        i = 0
        while i < len(order):
            j = i
            while j < len(order) and ep_arr[order[j]] == ep_arr[order[i]]:
                j += 1
            seq = order[i:j]  # rows of one episode, ordered by horizon
            for p in range(KW - 1, len(seq)):
                idx = seq[p]
                ctx = seq[p - KW + 1:p + 1]
                F[idx] = np.concatenate([E[c] for c in ctx])
                valid[idx] = True
            i = j
        return F, valid

    realK, validK = build_k4(real_E)
    predK, _ = build_k4(pred_E)
    trK = in_train & validK
    muK = realK[trK].mean(0); sdK = realK[trK].std(0) + 1e-6
    def zK(x): return (x - muK) / sdK
    rpK = Ridge(alpha=1.0).fit(zK(realK[trK]), pos_all[trK])
    rvK = Ridge(alpha=1.0).fit(zK(realK[trK]), vel_all[trK])
    teK = te & validK

    print(f"\n--- [K=4] decoded pos/vel ρ from ROLLED-OUT latents (test), by partition ---")
    print(f"  [K=4] probe applied to PREDICTED embs (stack 4 consecutive predicted latents):")
    for p in range(4):
        m = teK & (part_arr == p)
        if m.sum() < 50: continue
        pp = rpK.predict(zK(predK[m])); pv = rvK.predict(zK(predK[m]))
        line = f"  PRED-K4 {NAMES[p]:10s}"
        for d in range(pos_all.shape[1]):
            line += f"  pos{d}ρ={pearsonr(pos_all[m][:, d], pp[:, d])[0]:+.3f}"
        for d in range(vel_all.shape[1]):
            line += f"  vel{d}ρ={pearsonr(vel_all[m][:, d], pv[:, d])[0]:+.3f}"
        print(line)

    print(f"\n--- [K=4] decoded vel ρ from PRED latents vs horizon (test, aggregate) ---")
    for h in [4, 8, 16, 28]:
        m = teK & (horizon == h)
        if m.sum() < 20: continue
        pv = rvK.predict(zK(predK[m]))
        rhos = [pearsonr(vel_all[m][:, d], pv[:, d])[0] for d in range(vel_all.shape[1])]
        print(f"  h={h:3d}  n={m.sum():5d}  " + "  ".join(f"vel{d}ρ={r:+.3f}" for d, r in enumerate(rhos)))

    if gate_scores:
        q = np.percentile(np.asarray(gate_scores), [10, 25, 50, 75, 90, 95, 99])
        print("\n--- GIPP innovation score percentiles ---")
        print("  " + "  ".join(f"p{p}={v:.4f}" for p, v in zip([10,25,50,75,90,95,99], q)))
        gh = np.asarray(gate_horizons)
        gs = np.asarray(gate_scores)
        for h in [1, 2, 4, 8, 16, 28]:
            if np.any(gh == h):
                qh = np.percentile(gs[gh == h], [50, 75, 90])
                print(f"  h={h}: p50={qh[0]:.4f} p75={qh[1]:.4f} p90={qh[2]:.4f}")
    print(f"\nTotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
