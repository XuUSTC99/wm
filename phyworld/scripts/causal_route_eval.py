"""CAUSAL routing test: does the predictor read position from the slot or the black box?

The mechanism section currently argues *bypass* from decodability (the 190 non-slot
dims decode position as well as all 192). Decodability is not use (Elazar et al.).
This script intervenes on a trained model at rollout time and asks the causal
question directly: when we move the position that the slot reports, does the
prediction follow? When we move the position that the black box reports, does it?

Three families of intervention, all applied inside the AR rollout loop:

  steer   : add an offset to the latent so that ONE channel's decoded position
            shifts by exactly delta (position units), leaving the other channel
            untouched. Readout = position decoded from the model's own PREDICTED
            latent (recorded before any clamping), so the readout is downstream of
            the intervention, never directly damaged by it.
              g_slot = d(pred pos)/d(delta) via the slot
              g_bb   = d(pred pos)/d(delta) via the black box
              g_joint= both channels moved consistently  (sanity: should be ~1)
            Bypass predicts g_bb >> g_slot with g_joint ~ g_bb.
            Controls: a norm-matched RANDOM direction in the black box (must give
            ~0), and a delta sweep with both signs (linearity / off-manifold check).

  patch   : counterfactual patching. Replace the slot with a donor trajectory's
            slot (same timestep) and regress the rollout's decoded position on
            [own position, donor position]. beta_donor ~ 0 => the slot is a
            passenger. Also zero-slot and shuffle-slot ablations.

  amnesic : INLP nullspace projection that removes the linearly decodable position
            subspace from the black box at every rollout step, leaving the slot
            intact. If prediction still tracks position, the slot carries it; if it
            collapses, the black box was the route. Rank-matched RANDOM projection
            and a control-task projection bound how much damage removing *any* r
            directions does.

Usage:
  python causal_route_eval.py --domain uniform_motion --mode steer \
      --ckpt ~/.stable_worldmodel/uniform_motion_structpos_fr_pw30_id1k/..._epoch_20_object.ckpt \
      --tag structpos_pw30
"""
import argparse, os, sys, time, h5py, numpy as np, torch
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'le-wm'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from rollout_eval_id1k import DOMAINS, pl_2col, pl_4col, NAMES, HS, IMNET_MEAN, IMNET_STD

SLOT = slice(0, 2)          # structured physical slot (structured.start_dim=0, pos_dim=2)
PART_FOCUS = 3              # both-OOD is the paper's reading partition


# --------------------------------------------------------------------------- #
# probes                                                                       #
# --------------------------------------------------------------------------- #
class LinProbe:
    """Ridge probe on a dimension subset, standardized on the train split.

    Also exposes the *steering* direction: the minimum-norm latent offset (over
    this subset only) that moves the probe's decoded position by a given delta.
    """

    def __init__(self, X, Y, idx, alpha=1.0):
        self.idx = idx
        Xs = X[:, idx]
        self.mu = Xs.mean(0)
        self.sd = Xs.std(0) + 1e-6
        self.r = Ridge(alpha=alpha).fit((Xs - self.mu) / self.sd, Y)
        # min-norm steering: standardized offset u with W u = delta  ->  u = W^+ delta
        self.Wpinv = np.linalg.pinv(self.r.coef_)          # (d, k)

    def decode(self, X):
        return self.r.predict((X[:, self.idx] - self.mu) / self.sd)

    def steer_vec(self, delta):
        """delta: (k,) position offset -> raw-latent offset restricted to self.idx."""
        u_std = self.Wpinv @ delta                          # (d,)
        return u_std * self.sd                              # back to raw latent units


def inlp(X, Y, max_iter=80, tol=0.05, alpha=1.0):
    """Iterative nullspace projection. X standardized (N,d), Y (N,k).

    Returns (P, rank_removed, r2_trace). x' = x @ P kills the linearly decodable
    part of Y. Guarding on tol so we stop as soon as position is gone rather than
    deleting more of the representation than the claim needs.
    """
    d = X.shape[1]
    P = np.eye(d)
    trace = []
    for _ in range(max_iter):
        Xp = X @ P
        rg = Ridge(alpha=alpha).fit(Xp, Y)
        r2 = rg.score(Xp, Y)
        trace.append(float(r2))
        if r2 < tol:
            break
        W = rg.coef_                                        # (k, d)
        _, S, Vt = np.linalg.svd(W, full_matrices=False)
        B = Vt[S > 1e-8]                                    # orthonormal rowspace basis
        P = P @ (np.eye(d) - B.T @ B)
    rank = int(round(d - np.trace(P)))
    return P, rank, trace


def random_proj(d, rank, rng):
    """Project out `rank` random orthonormal directions (rank-matched control)."""
    A = rng.standard_normal((rank, d))
    Q, _ = np.linalg.qr(A.T)                                # (d, rank)
    return np.eye(d) - Q @ Q.T


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--mode", default="steer",
                    choices=["steer", "patch", "amnesic", "subset", "jacobian", "all"])
    ap.add_argument("--max-trajs", type=int, default=150)
    ap.add_argument("--deltas", type=float, nargs="+", default=[-1.0, -0.5, 0.5, 1.0],
                    help="steering magnitudes in units of position std")
    ap.add_argument("--clamp", default="every-step", choices=["every-step", "ctx-only"],
                    help="re-apply the intervention to fed-back predictions, or only to context")
    ap.add_argument("--slot-direct", action="store_true",
                    help="steer the slot by adding delta to it RAW rather than through the "
                         "probe pseudo-inverse. Correct for structured-slot models, where the "
                         "slot is supervised to equal proprio position, and far better "
                         "conditioned when the slot only partly absorbed position (collision).")
    args = ap.parse_args()
    cfg = DOMAINS[args.domain]
    ckpt_path = args.ckpt or cfg["ckpt"]
    dev = 'cuda'
    t0 = time.time()

    with h5py.File(cfg["train_h5"], 'r') as f:
        tr_act = f['action'][:].astype(np.float64)
    a_mean = np.nan_to_num(tr_act).mean(0)
    a_std = np.nan_to_num(tr_act).std(0) + 1e-8

    model = torch.load(ckpt_path, map_location='cpu', weights_only=False).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[ckpt] {ckpt_path}  tag={args.tag}  mode={args.mode}  clamp={args.clamp}", flush=True)

    f = h5py.File(cfg["eval_h5"], 'r')
    pixels = f['pixels']; ep_idx = f['episode_idx'][:]; step_idx = f['step_idx'][:]
    proprio = f['proprio'][:]; action = f['action'][:]
    with h5py.File(cfg["src_hdf5"], 'r') as f2:
        init = np.concatenate([f2['init_streams'][k][...] for k in sorted(f2['init_streams'])], 0)
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

    # ---- pass 1: encode once, cache everything (all conditions reuse this) ----
    trajs = []
    for ep in sel_eps:
        rows = np.nonzero(ep_idx == ep)[0]
        rows = rows[np.argsort(step_idx[rows])]
        T = len(rows)
        if T <= HS + 1:
            continue
        frames = pixels[rows[0]:rows[0] + T] if np.all(np.diff(rows) == 1) else pixels[:][rows]
        real_emb = encode_frames(frames)
        raw_act = np.nan_to_num(action[rows]).astype(np.float64)
        act_norm = torch.from_numpy(((raw_act - a_mean) / a_std).astype(np.float32)).to(dev)
        act_emb = model.action_encoder(act_norm.unsqueeze(0))[0]
        pos_t = proprio[rows][:, [0, 2]] if cfg["ncol"] == 4 else proprio[rows][:, :2]
        trajs.append(dict(ep=int(ep), real=real_emb, act=act_emb, pos=pos_t,
                          part=int(parts[ep]), in_train=ep in train_eps, T=T))
        if len(trajs) % 50 == 0:
            print(f"  encoded {len(trajs)} trajs, t={time.time()-t0:.0f}s", flush=True)
    f.close()
    D = trajs[0]["real"].size(1)
    print(f"[data] {len(trajs)} trajs, emb dim={D}, t={time.time()-t0:.0f}s", flush=True)

    # ---- probes on REAL embeddings, train split ----
    Xtr = np.concatenate([t["real"][HS:].cpu().numpy() for t in trajs if t["in_train"]], 0)
    Ytr = np.concatenate([t["pos"][HS:] for t in trajs if t["in_train"]], 0)
    idx_slot = np.arange(D)[SLOT]
    idx_bb = np.arange(2, D)
    pr_full = LinProbe(Xtr, Ytr, np.arange(D))
    pr_slot = LinProbe(Xtr, Ytr, idx_slot)
    pr_bb = LinProbe(Xtr, Ytr, idx_bb)
    pos_std = Ytr.std(0)
    print(f"[probe R2 on train]  full={pr_full.r.score((Xtr-pr_full.mu)/pr_full.sd, Ytr):.3f}"
          f"  slot={pr_slot.r.score((Xtr[:, idx_slot]-pr_slot.mu)/pr_slot.sd, Ytr):.3f}"
          f"  bb={pr_bb.r.score((Xtr[:, idx_bb]-pr_bb.mu)/pr_bb.sd, Ytr):.3f}", flush=True)

    # ===================================================================== #
    # MODE: jacobian -- per-dimension sensitivity, no intervention designed  #
    # ===================================================================== #
    if args.mode in ("jacobian", "all"):
        print("\n=== JACOBIAN: d(predicted position) / d(input latent dim) ===")
        print("    Every interventional measure of the black box here had to pick a")
        print("    direction first (the probe pseudo-inverse), and a redundant")
        print("    distributed code barely moves along any one direction -- which is")
        print("    why the min-norm patch reads ~0.09 on a channel that reads ~0.96")
        print("    when replaced wholesale. Differentiating picks no direction.")
        Wp = torch.from_numpy(pr_full.r.coef_.astype(np.float32)).to(dev)     # (2, D)
        mu_p = torch.from_numpy(pr_full.mu.astype(np.float32)).to(dev)
        sd_p = torch.from_numpy(pr_full.sd.astype(np.float32)).to(dev)
        rsj = np.random.default_rng(0)
        picks = [(i, k) for i, tr in enumerate(trajs) for k in range(HS, tr["T"])]
        rsj.shuffle(picks)
        picks = picks[:400]
        per_dim = torch.zeros(D, device=dev)
        for cnt, (i, k) in enumerate(picks, 1):
            tr = trajs[i]
            hist = tr["real"][k - HS:k].clone().unsqueeze(0).requires_grad_(True)
            pred = model.predict(hist, tr["act"][k - HS:k].unsqueeze(0))[0, -1]
            pos_hat = ((pred - mu_p) / sd_p) @ Wp.T                            # (2,)
            for c in range(pos_hat.numel()):
                gr = torch.autograd.grad(pos_hat[c], hist, retain_graph=True)[0]
                # sensitivity to the LAST history frame only -- that is the one the
                # patch/steer interventions clamp, so this stays comparable to them.
                per_dim += gr[0, -1].abs().detach()
            if cnt % 100 == 0:
                print(f"    {cnt}/{len(picks)}  t={time.time()-t0:.0f}s", flush=True)
        per_dim = (per_dim / len(picks)).cpu().numpy()
        s_mean, b_mean = per_dim[:2].mean(), per_dim[2:].mean()
        order = np.argsort(-per_dim)
        print(f"    slot [0:2]    mean |dpos/dz| per dim = {s_mean:.6f}")
        print(f"    black box     mean |dpos/dz| per dim = {b_mean:.6f}")
        print(f"    RATIO slot/bb per dim = {s_mean / (b_mean + 1e-12):.2f}")
        print(f"    totals: slot {per_dim[:2].sum():.6f} vs black box "
              f"{per_dim[2:].sum():.6f}  (slot share {per_dim[:2].sum() / (per_dim.sum() + 1e-12):.3f})")
        print(f"    slot dims rank {[int(np.where(order == j)[0][0]) + 1 for j in range(2)]} "
              f"of {D} by sensitivity")

    # ---- rollout with an arbitrary per-step latent intervention ----
    @torch.no_grad()
    def rollout(tr, fn=None, fn_ctx=None):
        """fn(e, k0): (n,D) -> (n,D), applied to latents FED INTO the predictor;
        k0 is the frame index of e's first row, so time-varying interventions
        (donor patches) can index their own targets.
        Predictions are recorded BEFORE fn, so the readout stays downstream-only."""
        real, act, T = tr["real"], tr["act"], tr["T"]
        g_ctx = fn_ctx or fn
        hist = real[:HS].clone()
        if g_ctx is not None:
            hist = g_ctx(hist, 0)
        preds = []
        for k in range(HS, T):
            p = model.predict(hist[-HS:].unsqueeze(0), act[k - HS:k].unsqueeze(0))[0, -1]
            preds.append(p)
            nxt = p.unsqueeze(0)
            if fn is not None and args.clamp == "every-step":
                nxt = fn(nxt, k)
            hist = torch.cat([hist, nxt], 0)
        return torch.stack(preds, 0)

    def run_all(fn=None, fn_ctx=None):
        """-> (pred latents (N,D), per-frame pos targets (N,2), part (N,), in_train (N,))"""
        P, Y, PA, IT = [], [], [], []
        for tr in trajs:
            pe = rollout(tr, fn, fn_ctx).cpu().numpy()
            P.append(pe); Y.append(tr["pos"][HS:tr["T"]])
            PA.append(np.full(len(pe), tr["part"])); IT.append(np.full(len(pe), tr["in_train"]))
        return (np.concatenate(P), np.concatenate(Y), np.concatenate(PA),
                np.concatenate(IT).astype(bool))

    def nmse(pred, real):
        return float(((pred - real) ** 2).sum(1).mean() / ((real - real.mean(0)) ** 2).sum(1).mean())

    print(f"\n[clean rollout] t={time.time()-t0:.0f}s", flush=True)
    Pc, Ypos, part_arr, in_train = run_all(None)
    te = ~in_train
    m_focus = te & (part_arr == PART_FOCUS)
    if m_focus.sum() < 50:
        m_focus = te
        print("  (both-OOD too small on this subsample -> reporting over all test frames)")
    dec_clean = pr_full.decode(Pc)
    rho_clean = [pearsonr(Ypos[m_focus][:, d], dec_clean[m_focus][:, d])[0] for d in range(2)]
    print(f"  clean decoded rho from PRED latents ({NAMES[PART_FOCUS]}, test): "
          f"{rho_clean[0]:+.3f} / {rho_clean[1]:+.3f}   n={m_focus.sum()}")

    Rreal = np.concatenate([t["real"][HS:t["T"]].cpu().numpy() for t in trajs])
    print(f"  clean latent nMSE (pred vs real, test focus): {nmse(Pc[m_focus], Rreal[m_focus]):.4f}")

    rs = np.random.default_rng(42)

    # ===================================================================== #
    # MODE: subset -- the bypass table, but on ROLLED-OUT latents            #
    # ===================================================================== #
    if args.mode in ("subset", "all"):
        print(f"\n=== DIM-SUBSET decodability on ROLLED-OUT latents ===")
        print(f"    (the published table probes frame embeddings; this is the same")
        print(f"     table applied to what the predictor actually produces)")
        horiz = np.concatenate([np.arange(1, t["T"] - HS + 1) for t in trajs])
        subsets = {"all[192]": np.arange(D), "slot[0:2]": idx_slot,
                   "blackbox[2:192]": idx_bb,
                   "rand-2": rs.choice(D, 2, replace=False),
                   "rand-10": rs.choice(D, 10, replace=False)}
        hs_report = [1, 2, 4, 8, 16, 28]
        print(f"    {'subset':16s} {'REAL':>8s} {'PRED':>8s} | "
              + " ".join(f"{'h'+str(h):>7s}" for h in hs_report))
        for name, idx in subsets.items():
            pr = LinProbe(Xtr, Ytr, idx)
            def rho_on(X, m):
                dd = pr.decode(X)
                return np.nanmean([pearsonr(Ypos[m][:, d], dd[m][:, d])[0] for d in range(2)])
            cells = []
            for h in hs_report:
                m = m_focus & (horiz == h)
                cells.append(f"{rho_on(Pc, m):+.3f}" if m.sum() >= 20 else "    n/a")
            print(f"    {name:16s} {rho_on(Rreal, m_focus):+8.3f} {rho_on(Pc, m_focus):+8.3f} | "
                  + " ".join(f"{c:>7s}" for c in cells), flush=True)

    # ===================================================================== #
    # MODE: steer                                                           #
    # ===================================================================== #
    if args.mode in ("steer", "all"):
        print(f"\n=== STEERING: causal gain d(predicted position)/d(delta) ===")
        print(f"    delta is applied so that the *named channel's* decoded position "
              f"moves by delta; readout is the full-latent probe on the PREDICTED latent.")

        def mk_fn(vec_np):
            v = torch.from_numpy(vec_np.astype(np.float32)).to(dev)
            return lambda e, k0: e + v

        def gain_of(vec_np, delta_vec):
            """Returns gains under THREE readouts of the predicted latent.

            The full-latent readout alone is confounded: if the predictor merely
            copies its input slot forward, perturbing the slot moves the full
            readout without the model's scene state having moved at all. The
            black-box readout is the one that reflects computation (it is also
            what drives the pixel decoder), so g[*][bb] is the load-bearing test.
            """
            fn = mk_fn(vec_np)
            Pp, _, _, _ = run_all(fn)
            out = {}
            for nm_, pr_ in [("full", pr_full), ("bb", pr_bb), ("slot", pr_slot)]:
                d = pr_.decode(Pp)[m_focus] - pr_.decode(Pc)[m_focus]
                out[nm_] = d.mean(0) / (delta_vec + 1e-12)
            return out, nmse(Pp[m_focus], Rreal[m_focus]), float(np.linalg.norm(vec_np))

        rows = []
        for s in args.deltas:
            dvec = s * pos_std                                   # (2,) position offset
            v_slot = np.zeros(D)
            v_slot[idx_slot] = dvec if args.slot_direct else pr_slot.steer_vec(dvec)
            v_bb = np.zeros(D);   v_bb[idx_bb] = pr_bb.steer_vec(dvec)
            v_joint = v_slot + v_bb
            # norm-matched random control inside the black box
            rv = rs.standard_normal(len(idx_bb))
            rv *= np.linalg.norm(v_bb[idx_bb]) / (np.linalg.norm(rv) + 1e-12)
            v_rand = np.zeros(D); v_rand[idx_bb] = rv
            for name, v in [("slot", v_slot), ("blackbox", v_bb),
                            ("joint", v_joint), ("rand-bb(norm-matched)", v_rand)]:
                g, nm, nv = gain_of(v, dvec)
                rows.append((s, name, g, nm, nv))
                print(f"  delta={s:+.2f}sd  {name:22s} "
                      + "  ".join(f"g[{k}]=({g[k][0]:+.3f},{g[k][1]:+.3f})" for k in ("full", "bb", "slot"))
                      + f"  |v|={nv:6.2f}  nMSE={nm:.4f}", flush=True)

        print(f"\n  --- summary (mean over |delta| and signs; per coordinate) ---")
        agg = {}
        for s, name, g, nm, nv in rows:
            agg.setdefault(name, []).append(g)
        for name, gl in agg.items():
            line = f"    {name:22s}"
            for k in ("full", "bb", "slot"):
                m = np.mean([x[k] for x in gl], 0)
                line += f"  read-{k}=({m[0]:+.3f},{m[1]:+.3f})"
            print(line)

        print(f"\n  --- ROUTING INDEX, per readout and per coordinate ---")
        print(f"      R = g_bb / (g_bb + g_slot);  1.0 = position taken entirely from the")
        print(f"      black box (bypass), 0.0 = entirely from the slot (load-bearing).")
        print(f"      read-bb is the load-bearing test: read-full and read-slot can be")
        print(f"      inflated by the predictor simply copying its input slot forward.")
        for k in ("full", "bb", "slot"):
            gs_ = np.mean([x[k] for x in agg["slot"]], 0)
            gb_ = np.mean([x[k] for x in agg["blackbox"]], 0)
            gj_ = np.mean([x[k] for x in agg["joint"]], 0)
            R = gb_ / (gb_ + gs_ + 1e-12)
            print(f"      read-{k:5s}  R=({R[0]:+.3f},{R[1]:+.3f})   "
                  f"additivity g_slot+g_bb=({gs_[0]+gb_[0]:+.3f},{gs_[1]+gb_[1]:+.3f}) "
                  f"vs g_joint=({gj_[0]:+.3f},{gj_[1]:+.3f})")

    # ===================================================================== #
    # MODE: patch                                                           #
    # ===================================================================== #
    if args.mode in ("patch", "all"):
        print(f"\n=== COUNTERFACTUAL PATCHING of the slot ===")
        donor = rs.permutation(len(trajs))
        donor_pos, own_pos = [], []
        for i, tr in enumerate(trajs):
            dtr = trajs[donor[i]]
            L = tr["T"] - HS
            dp = dtr["pos"][HS:HS + L]
            if len(dp) < L:                                     # pad short donors by repeat
                dp = np.concatenate([dp, np.repeat(dp[-1:], L - len(dp), 0)], 0)
            donor_pos.append(dp); own_pos.append(tr["pos"][HS:tr["T"]])
        donor_pos = np.concatenate(donor_pos); own_pos = np.concatenate(own_pos)

        bbW = torch.from_numpy(pr_bb.Wpinv.astype(np.float32)).to(dev)      # (190, 2)
        bb_sd = torch.from_numpy(pr_bb.sd.astype(np.float32)).to(dev)
        idx_r2 = torch.from_numpy(rs.choice(idx_bb, 2, replace=False)).to(dev)
        # Dose-response subsets of the black box, fixed once so every trajectory
        # and every donor sees the same dimensions. The min-norm "donor-bb" arm
        # systematically under-moves a redundant distributed code -- in a
        # baseline model, where the black box is the ONLY route position can
        # take, it still yields a follow-fraction of ~0.05. That is the method's
        # floor, not evidence about the channel. Replacing k dimensions outright
        # sidesteps it and turns the question into a measurable one: how many
        # black-box dimensions must move before the rollout follows as far as
        # moving the 2-d slot does?
        BB_DOSES = [2, 10, 40, 100, len(idx_bb)]
        idx_dose = {k: torch.from_numpy(
                        np.sort(rs.choice(idx_bb, min(k, len(idx_bb)), replace=False))
                    ).to(dev) for k in BB_DOSES}

        def patch_run(kind):
            P = []
            for i, tr in enumerate(trajs):
                dtr = trajs[donor[i]]
                if kind == "zero":
                    fn = lambda e, k0: torch.cat([torch.zeros_like(e[..., SLOT]), e[..., 2:]], -1)
                elif kind == "donor-slot":
                    ds = dtr["real"][:, SLOT]
                    def fn(e, k0, ds=ds):
                        j = torch.clamp(torch.arange(k0, k0 + e.size(0), device=dev),
                                        max=ds.size(0) - 1)
                        return torch.cat([ds[j], e[..., 2:]], -1)
                elif kind == "freeze":                          # hold the slot at its t=HS value
                    s0 = tr["real"][HS - 1:HS, SLOT]
                    fn = lambda e, k0, s0=s0: torch.cat([s0.expand(e.size(0), -1), e[..., 2:]], -1)
                elif kind == "donor-rand2":
                    # Control for "patching the slot just knocks the latent off-manifold":
                    # patch 2 RANDOM black-box dims from the same donor. Same kind and
                    # roughly the same amount of inconsistency, no physical slot involved.
                    ds = dtr["real"][:, idx_r2]
                    def fn(e, k0, ds=ds):
                        j = torch.clamp(torch.arange(k0, k0 + e.size(0), device=dev),
                                        max=ds.size(0) - 1)
                        out = e.clone()
                        out[..., idx_r2] = ds[j]
                        return out
                elif kind == "donor-bb":
                    # Symmetric counterpart: move ONLY the position-decodable component
                    # of the black box onto the donor's value (min-norm offset), leaving
                    # the rest of those 190 dims and the whole slot untouched. This is
                    # the mediation test -- fix one channel, counterfactually move the
                    # other, and see which one the rollout follows.
                    dp = torch.from_numpy(
                        pr_bb.decode(dtr["real"].cpu().numpy()).astype(np.float32)).to(dev)
                    op = torch.from_numpy(
                        pr_bb.decode(tr["real"].cpu().numpy()).astype(np.float32)).to(dev)

                    def fn(e, k0, dp=dp, op=op):
                        j = torch.clamp(torch.arange(k0, k0 + e.size(0), device=dev),
                                        max=min(dp.size(0), op.size(0)) - 1)
                        off = ((dp[j] - op[j]) @ bbW.T) * bb_sd   # (n, 190)
                        return torch.cat([e[..., :2], e[..., 2:] + off], -1)
                elif kind.startswith("donor-bbk"):
                    # Wholesale replacement of k black-box dimensions by the
                    # donor's, slot left at this trajectory's own value. Sweeping
                    # k gives the dose-response the min-norm arm cannot: at
                    # k=len(idx_bb) the entire black box is the donor's, which
                    # also serves as the positive control the min-norm arm fails
                    # (a baseline model must follow the donor there).
                    sel = idx_dose[int(kind[9:])]
                    ds = dtr["real"][:, sel]

                    def fn(e, k0, ds=ds, sel=sel):
                        j = torch.clamp(torch.arange(k0, k0 + e.size(0), device=dev),
                                        max=ds.size(0) - 1)
                        out = e.clone()
                        out[..., sel] = ds[j]
                        return out
                P.append(rollout(tr, fn).cpu().numpy())
            return np.concatenate(P)

        rho_clean_bb = [pearsonr(Ypos[m_focus][:, d], pr_bb.decode(Pc)[m_focus][:, d])[0]
                        for d in range(2)]
        rho_clean_slot = [pearsonr(Ypos[m_focus][:, d], pr_slot.decode(Pc)[m_focus][:, d])[0]
                          for d in range(2)]
        for kind in (["zero", "donor-slot", "donor-rand2", "freeze", "donor-bb"]
                     + [f"donor-bbk{k}" for k in BB_DOSES]):
            Pp = patch_run(kind)
            print(f"  {kind:11s}  nMSE={nmse(Pp[m_focus], Rreal[m_focus]):.4f}", flush=True)
            # The CROSS terms are the confound-free ones: clamping a channel and then
            # reading that same channel back is partly trivial (the predictor copies
            # its input forward), but "patch the slot, read the black box" and
            # "patch the black box, read the slot" cannot be explained that way.
            for rname, pr_, base in [("full", pr_full, rho_clean), ("bb", pr_bb, rho_clean_bb),
                                     ("slot", pr_slot, rho_clean_slot)]:
                dec = pr_.decode(Pp)
                rho = [pearsonr(Ypos[m_focus][:, d], dec[m_focus][:, d])[0] for d in range(2)]
                print(f"      read-{rname:4s} rho vs OWN pos = {rho[0]:+.3f}/{rho[1]:+.3f}"
                      f"   (clean {base[0]:+.3f}/{base[1]:+.3f})")
                if kind.startswith("donor"):
                    for d in range(2):
                        A = np.stack([own_pos[m_focus][:, d], donor_pos[m_focus][:, d],
                                      np.ones(m_focus.sum())], 1)
                        beta, *_ = np.linalg.lstsq(A, dec[m_focus][:, d], rcond=None)
                        print(f"        coord{d}: beta_own={beta[0]:+.3f} beta_donor={beta[1]:+.3f}"
                              f"  -> follow-fraction(donor) = "
                              f"{abs(beta[1])/(abs(beta[0])+abs(beta[1])+1e-12):.3f}")

    # ===================================================================== #
    # MODE: amnesic                                                         #
    # ===================================================================== #
    if args.mode in ("amnesic", "all"):
        print(f"\n=== AMNESIC PROJECTION on the black box (slot left intact) ===")
        Xbb = (Xtr[:, idx_bb] - pr_bb.mu) / pr_bb.sd
        P_am, rank, trace = inlp(Xbb, Ytr)
        # How many linear directions it takes to erase position is itself the
        # quantitative form of "the copy is redundant and distributed".
        print(f"  INLP removed rank {rank} of {len(idx_bb)} to drive position R2 "
              f"{trace[0]:.3f} -> {trace[-1]:.3f}"
              + ("  [HIT ITERATION CAP -- position NOT erased]" if trace[-1] > 0.05 else ""))
        print(f"    R2 trace every 5 removals: "
              + " -> ".join(f"{r:.3f}" for r in trace[::5]))
        P_rand = random_proj(len(idx_bb), rank, rs)

        mu_t = torch.from_numpy(pr_bb.mu.astype(np.float32)).to(dev)
        sd_t = torch.from_numpy(pr_bb.sd.astype(np.float32)).to(dev)

        def mk_proj(Pm):
            Pt = torch.from_numpy(Pm.astype(np.float32)).to(dev)
            def fn(e, k0):
                x = (e[..., 2:] - mu_t) / sd_t
                return torch.cat([e[..., :2], (x @ Pt) * sd_t + mu_t], -1)
            return fn

        # slot-side counterpart: kill the position-decodable part of the slot only
        Xs = (Xtr[:, idx_slot] - pr_slot.mu) / pr_slot.sd
        P_slot_am, rank_s, trace_s = inlp(Xs, Ytr, max_iter=3)
        print(f"  (slot-side INLP removed rank {rank_s} of 2; trace "
              + " -> ".join(f"{r:.3f}" for r in trace_s) + ")")
        mu_s = torch.from_numpy(pr_slot.mu.astype(np.float32)).to(dev)
        sd_s = torch.from_numpy(pr_slot.sd.astype(np.float32)).to(dev)
        Ps_t = torch.from_numpy(P_slot_am.astype(np.float32)).to(dev)

        def fn_slot_am(e, k0):
            x = (e[..., :2] - mu_s) / sd_s
            return torch.cat([(x @ Ps_t) * sd_s + mu_s, e[..., 2:]], -1)

        for name, fn in [("bb: position removed (INLP)", mk_proj(P_am)),
                         (f"bb: random rank-{rank} (control)", mk_proj(P_rand)),
                         ("slot: position removed", fn_slot_am)]:
            Pp, _, _, _ = run_all(fn)
            line = f"  {name:32s} nMSE={nmse(Pp[m_focus], Rreal[m_focus]):.4f}"
            for rname, pr_ in [("full", pr_full), ("bb", pr_bb), ("slot", pr_slot)]:
                dec = pr_.decode(Pp)
                base = [pearsonr(Ypos[m_focus][:, d], pr_.decode(Pc)[m_focus][:, d])[0]
                        for d in range(2)]
                rho = [pearsonr(Ypos[m_focus][:, d], dec[m_focus][:, d])[0] for d in range(2)]
                line += (f"\n      read-{rname:5s} rho={rho[0]:+.3f}/{rho[1]:+.3f}"
                         f"  (clean {base[0]:+.3f}/{base[1]:+.3f})")
            print(line, flush=True)

    print(f"\n[done] t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
