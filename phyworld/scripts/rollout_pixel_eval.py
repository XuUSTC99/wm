"""Pixel-space AR rollout eval — the MODEL-INDEPENDENT world-model metric.

Latent-space metrics (validate/pred_loss raw MSE, rollout nMSE, decoder PSNR,
probe rho) are all measured in a latent space that the probe-weight lambda
itself reshapes, so none cleanly compares forward-dynamics quality ACROSS lambda.

This script grounds everything in PIXELS:
  encode HS real frames -> AR-roll predicted proj-latents with TRUE actions
  -> decode each predicted latent with a FIXED universal decoder (trained on
     ID+OOD, this model's proj space) -> predicted future FRAME
  -> PSNR(predicted_frame, true_future_frame), per partition, per horizon.

Pixel PSNR is in model-independent units, so cross-lambda comparison is honest.
We also decode the REAL latent ("ceiling": pure decoder/info content) to
separate "how much info is in the latent" from "how well the predictor rolls".

Usage:
  python rollout_pixel_eval.py --domain uniform_motion \
    --ckpt <encoder_object.ckpt> --decoder <proj_universal_decoder.pt> \
    --tag lam1 --max-trajs 800 --out <dir>
"""
import argparse, json, os, sys, time
from pathlib import Path

import h5py, numpy as np, torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
_SWM = Path(os.environ.get("STABLEWM_HOME", str(Path.home() / ".stable_worldmodel")))
_DS = _SWM / "datasets"
sys.path.insert(0, str(_ROOT / "le-wm"))
sys.path.insert(0, str(_ROOT / "le-wm" / "decode_viz"))
from decoder import LatentDecoder  # noqa: E402

DOMAINS = {
    "uniform_motion": {
        "train_h5": str(_DS / "phyworld_uniform_motion_id1k.h5"),
        "eval_h5": str(_DS / "phyworld_uniform_motion.h5"),
        "src_hdf5": str(_ROOT / "phyworld/data/uniform_motion_eval.hdf5"),
        "ncol": 2,
    },
    "collision": {
        "train_h5": str(_DS / "phyworld_collision_id1k.h5"),
        "eval_h5": str(_DS / "phyworld_collision_eval.h5"),
        "src_hdf5": str(_ROOT / "phyworld/data/collision_eval.hdf5"),
        "ncol": 4,
    },
    "parabola": {
        "train_h5": str(_DS / "phyworld_parabola_id1k.h5"),
        "eval_h5": str(_DS / "phyworld_parabola.h5"),
        "src_hdf5": str(_ROOT / "phyworld/data/parabola_eval.hdf5"),
        "ncol": 2,
    },
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

NAMES = {0: "ID", 1: "r/m-OOD", 2: "v-OOD", 3: "both-OOD"}
HBUCKETS = [1, 2, 4, 8, 16, 28]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), default="uniform_motion")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--decoder", required=True, help="proj-space universal decoder .pt")
    ap.add_argument("--max-trajs", type=int, default=800)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = DOMAINS[args.domain]
    dev = "cuda"
    t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # action norm (must match FT)
    with h5py.File(cfg["train_h5"], "r") as f:
        tr_act = f["action"][:].astype(np.float64)
    a_mean = np.nan_to_num(tr_act).mean(0); a_std = np.nan_to_num(tr_act).std(0) + 1e-8
    print(f"[norm] action mean={a_mean} std={a_std}", flush=True)

    # model
    model = torch.load(args.ckpt, map_location="cpu", weights_only=False).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)

    # proj-space universal decoder
    sd = torch.load(args.decoder, map_location=dev)
    emb_dim = sd["emb_dim"]
    dec = LatentDecoder(emb_dim=emb_dim).to(dev).eval()
    dec.load_state_dict(sd["decoder"])
    print(f"[decoder] {args.decoder}  emb_dim={emb_dim}", flush=True)

    # eval data + partition labels
    f = h5py.File(cfg["eval_h5"], "r")
    pixels = f["pixels"]; ep_idx = f["episode_idx"][:]; step_idx = f["step_idx"][:]
    action = f["action"][:]
    f2 = h5py.File(cfg["src_hdf5"], "r")
    init = np.concatenate([f2["init_streams"][k][...] for k in sorted(f2["init_streams"])], 0)
    f2.close()
    parts = np.array([pl_2col(float(init[i, 0]), float(init[i, 1])) for i in range(len(init))], np.uint8)

    uniq_eps = np.unique(ep_idx)
    rng = np.random.default_rng(0)
    sel_eps = rng.permutation(uniq_eps)
    if args.max_trajs:
        sel_eps = sel_eps[:args.max_trajs]

    mean_d = IMNET_MEAN.to(dev); std_d = IMNET_STD.to(dev)

    @torch.no_grad()
    def encode_frames(frames_u8):
        x = torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float().to(dev) / 255.0
        x = (x - mean_d) / std_d
        return model.encode({"pixels": x.unsqueeze(0)})["emb"][0]  # (T, D) proj space

    @torch.no_grad()
    def ar_rollout(real_emb, act_norm):
        T = real_emb.size(0)
        emb_hist = real_emb[:HS].clone()
        act_emb_all = model.action_encoder(act_norm.unsqueeze(0))[0]
        preds = []
        for k in range(HS, T):
            e_in = emb_hist[-HS:].unsqueeze(0)
            a_in = act_emb_all[k - HS:k].unsqueeze(0)
            p = model.predict(e_in, a_in)[0, -1]
            preds.append(p)
            emb_hist = torch.cat([emb_hist, p.unsqueeze(0)], 0)
        return torch.stack(preds, 0)  # (T-HS, D) aligned to frames HS..T-1

    # accumulators: sum of MSE + count, keyed by (partition) and (horizon)
    acc = {"pred": {}, "real": {}}  # acc[kind][("part",p)] = [sum_mse, n]
    def add(kind, key, mse, n):
        d = acc[kind].setdefault(key, [0.0, 0])
        d[0] += mse * n; d[1] += n

    n_done = 0
    for ep in sel_eps:
        rows = np.nonzero(ep_idx == ep)[0]
        rows = rows[np.argsort(step_idx[rows])]
        T = len(rows)
        if T <= HS + 1: continue
        frames = pixels[rows[0]:rows[0] + T] if np.all(np.diff(rows) == 1) else pixels[:][rows]
        real_emb = encode_frames(frames)                       # (T, D)
        raw_act = np.nan_to_num(action[rows]).astype(np.float64)
        act_norm = torch.from_numpy(((raw_act - a_mean) / a_std).astype(np.float32)).to(dev)
        pred = ar_rollout(real_emb, act_norm)                  # (T-HS, D)

        part = int(parts[ep])
        # decode predicted + real latents for frames HS..T-1
        gt = torch.from_numpy(frames[HS:T]).permute(0, 3, 1, 2).float().to(dev) / 255.0  # (T-HS,3,H,W)
        with torch.no_grad():
            pred_img = dec(pred).clamp(0, 1)
            real_img = dec(real_emb[HS:T]).clamp(0, 1)
        mse_pred = ((pred_img - gt) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()  # per-frame
        mse_real = ((real_img - gt) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()

        for j in range(T - HS):
            h = j + 1  # horizon
            add("pred", ("part", part), float(mse_pred[j]), 1)
            add("real", ("part", part), float(mse_real[j]), 1)
            if h in HBUCKETS:
                add("pred", ("h", h), float(mse_pred[j]), 1)
                add("real", ("h", h), float(mse_real[j]), 1)
        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done} trajs, t={time.time()-t0:.0f}s", flush=True)
    f.close()

    def psnr(kind, key):
        s, n = acc[kind].get(key, [0.0, 0])
        if n == 0: return None, 0
        m = s / n
        return float(-10.0 * np.log10(max(m, 1e-10))), n

    res = {"by_partition": {}, "by_horizon": {}}
    print(f"\n=== PIXEL ROLLOUT | {args.domain} | tag={args.tag} | enc={Path(args.ckpt).name} ===")
    print(f"--- predicted-future-frame PSNR vs true frame, by partition ---")
    print(f"{'partition':<10} {'PRED PSNR':>10} {'REAL(ceil)':>11} {'gap':>7} {'n':>7}")
    for p in range(4):
        pp, n = psnr("pred", ("part", p)); rp, _ = psnr("real", ("part", p))
        if pp is None: continue
        res["by_partition"][NAMES[p]] = {"pred_psnr": pp, "real_psnr": rp, "n": n}
        print(f"{NAMES[p]:<10} {pp:>10.2f} {rp:>11.2f} {rp-pp:>7.2f} {n:>7}")

    print(f"\n--- predicted-future-frame PSNR vs horizon (aggregate) ---")
    print(f"{'h':>4} {'PRED PSNR':>10} {'REAL(ceil)':>11} {'gap':>7} {'n':>7}")
    for h in HBUCKETS:
        pp, n = psnr("pred", ("h", h)); rp, _ = psnr("real", ("h", h))
        if pp is None: continue
        res["by_horizon"][str(h)] = {"pred_psnr": pp, "real_psnr": rp, "n": n}
        print(f"{h:>4} {pp:>10.2f} {rp:>11.2f} {rp-pp:>7.2f} {n:>7}")

    with open(out / f"pxroll_{args.tag}.json", "w") as fo:
        json.dump({"domain": args.domain, "tag": args.tag, "ckpt": args.ckpt,
                   "decoder": args.decoder, "n_trajs": n_done, "results": res}, fo, indent=2)
    print(f"\n[saved] pxroll_{args.tag}.json  ({n_done} trajs, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
