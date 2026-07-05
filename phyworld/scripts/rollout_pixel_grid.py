"""Visual AR-rollout comparison for ONE trajectory.

rollout_pixel_eval.py reports PRED PSNR numbers; this script makes the picture:
for a single trajectory, AR-roll the predictor with TRUE actions and, at a set of
horizons, show three rows
  row 1 = real future frame (ground truth)
  row 2 = decode(REAL latent)      -> "ceiling": what the decoder can do with a
                                      perfect latent (isolates decoder limits)
  row 3 = decode(PREDICTED latent) -> the world model's actual prediction
Comparing row1 vs row2 = decoder fidelity; row2 vs row3 = prediction drift.
Columns are increasing horizon, so the predicted ball visibly drifts off the
real one as the rollout gets longer.

Reuses the model loading + ar_rollout of rollout_pixel_eval.py.
"""
import argparse, os, sys
from pathlib import Path

import h5py, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parents[2]
_SWM = Path(os.environ.get("STABLEWM_HOME", str(Path.home() / ".stable_worldmodel")))
_DS = _SWM / "datasets"
sys.path.insert(0, str(_ROOT / "le-wm"))
sys.path.insert(0, str(_ROOT / "le-wm" / "decode_viz"))
from decoder import LatentDecoder  # noqa: E402

HS = 3
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pl_2col(r, v):
    r_ok = 0.7 <= abs(r) <= 1.5; v_ok = 1.0 <= abs(v) <= 4.0
    if r_ok and v_ok: return 0
    if not r_ok and v_ok: return 1
    if r_ok and not v_ok: return 2
    return 3


def ball_visible(frame_u8):
    r = ((frame_u8[..., 0].astype(int) > 120)
         & (frame_u8[..., 1].astype(int) < 100)
         & (frame_u8[..., 2].astype(int) < 100))
    return int(r.sum()) > 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="uniform_motion")
    ap.add_argument("--ckpt", required=True, help="JEPA object ckpt (has predictor)")
    ap.add_argument("--decoder", required=True, help="proj-space universal decoder .pt")
    ap.add_argument("--episode", type=int, default=-1, help="-1 = auto-pick ID traj, ball in-frame at end")
    ap.add_argument("--horizons", default="1,4,8,12,16,20,24,28")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"

    train_h5 = str(_DS / f"phyworld_{args.domain}_id1k.h5")
    eval_h5 = str(_DS / f"phyworld_{args.domain}.h5")
    src_hdf5 = str(_ROOT / f"phyworld/data/{args.domain}_eval.hdf5")

    with h5py.File(train_h5, "r") as f:
        tr_act = f["action"][:].astype(np.float64)
    a_mean = np.nan_to_num(tr_act).mean(0); a_std = np.nan_to_num(tr_act).std(0) + 1e-8

    model = torch.load(args.ckpt, map_location="cpu", weights_only=False).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    sd = torch.load(args.decoder, map_location=dev)
    dec = LatentDecoder(emb_dim=sd["emb_dim"]).to(dev).eval()
    dec.load_state_dict(sd["decoder"])

    f = h5py.File(eval_h5, "r")
    ep_idx = f["episode_idx"][:]; step_idx = f["step_idx"][:]; action = f["action"][:]
    proprio = f["proprio"][:]
    f2 = h5py.File(src_hdf5, "r")
    init = np.concatenate([f2["init_streams"][k][...] for k in sorted(f2["init_streams"])], 0)
    f2.close()
    parts = np.array([pl_2col(float(init[i, 0]), float(init[i, 1])) for i in range(len(init))], np.uint8)

    # choose episode: ID partition, FASTEST (max horizontal travel) whose ball is
    # still in-frame at the end — so the predictor's drift is actually visible.
    if args.episode < 0:
        cands = []
        for e in np.unique(ep_idx):
            if parts[e] != 0:
                continue
            rows = np.where(ep_idx == e)[0]; rows = rows[np.argsort(step_idx[rows])]
            if ball_visible(f["pixels"][int(rows[-1])]):
                travel = abs(proprio[rows[-1], 0] - proprio[rows[0], 0])
                cands.append((travel, int(e)))
        cands.sort(reverse=True)
        ep = cands[0][1]
        print(f"[auto] episode {ep} (ID, max travel={cands[0][0]:.2f}, ball in-frame at end)")
    else:
        ep = args.episode
    rows = np.where(ep_idx == ep)[0]; rows = rows[np.argsort(step_idx[rows])]
    T = len(rows)
    frames = f["pixels"][rows[0]:rows[0] + T] if np.all(np.diff(rows) == 1) else f["pixels"][:][rows]
    f.close()

    mean_d = IMNET_MEAN.to(dev); std_d = IMNET_STD.to(dev)

    @torch.no_grad()
    def encode_frames(frames_u8):
        x = torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float().to(dev) / 255.0
        x = (x - mean_d) / std_d
        return model.encode({"pixels": x.unsqueeze(0)})["emb"][0]

    @torch.no_grad()
    def ar_rollout(real_emb, act_norm):
        emb_hist = real_emb[:HS].clone()
        act_emb_all = model.action_encoder(act_norm.unsqueeze(0))[0]
        preds = []
        for k in range(HS, real_emb.size(0)):
            p = model.predict(emb_hist[-HS:].unsqueeze(0), act_emb_all[k - HS:k].unsqueeze(0))[0, -1]
            preds.append(p); emb_hist = torch.cat([emb_hist, p.unsqueeze(0)], 0)
        return torch.stack(preds, 0)  # (T-HS, D) aligned to frames HS..T-1

    real_emb = encode_frames(frames)
    raw_act = np.nan_to_num(action[rows]).astype(np.float64)
    act_norm = torch.from_numpy(((raw_act - a_mean) / a_std).astype(np.float32)).to(dev)
    pred = ar_rollout(real_emb, act_norm)

    horizons = [h for h in (int(x) for x in args.horizons.split(",")) if h <= T - HS]
    cols = []
    with torch.no_grad():
        for h in horizons:
            fi = HS + h - 1                       # frame index for horizon h
            gt = torch.from_numpy(frames[fi]).permute(2, 0, 1).float().to(dev) / 255.0
            ceil = dec(real_emb[fi:fi + 1]).clamp(0, 1)[0]
            prd = dec(pred[h - 1:h]).clamp(0, 1)[0]
            cols.append((h, gt.cpu(), ceil.cpu(), prd.cpu()))

    # compose 3-row labelled canvas
    C, H, W = 3, frames.shape[1], frames.shape[2]
    pad, lab_w, lab_h = 4, 70, 18
    n = len(cols)
    cw = n * W + (n - 1) * pad
    canvas = np.ones((lab_h + 3 * H + 2 * pad, lab_w + cw, 3), np.float32)
    rows_t = [c[1:] for c in cols]
    for r in range(3):
        for ci, (h, *imgs) in enumerate(cols):
            img = imgs[r].permute(1, 2, 0).numpy()
            y0 = lab_h + r * (H + pad); x0 = lab_w + ci * (W + pad)
            canvas[y0:y0 + H, x0:x0 + W] = img
    im = Image.fromarray((canvas * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    try: font = ImageFont.load_default()
    except Exception: font = None
    for ci, (h, *_), in enumerate(cols):
        dr.text((lab_w + ci * (W + pad) + W // 2 - 10, 4), f"h={h}", fill=(0, 0, 0), font=font)
    for r, lab in enumerate(["GT", "ceil", "pred"]):
        dr.text((4, lab_h + r * (H + pad) + H // 2), lab, fill=(0, 0, 0), font=font)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    print(f"[saved] {out}  episode={ep}  horizons={horizons}")


if __name__ == "__main__":
    main()
