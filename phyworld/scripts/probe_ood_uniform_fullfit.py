"""ID+OOD-mixed probe on phyworld uniform_motion, per-partition evaluation.

Mirror of probe_ood_fullfit.py but for uniform_motion instead of collision.

Protocol (matching collision §2 of negtive_result_report.md):
  - Take phyworld_uniform_motion.h5 (1152 trajs, mixed ID + r-OOD + v-OOD + both-OOD)
  - Pull partition labels from source uniform_motion_eval.hdf5 init_streams
  - 80/20 split BY TRAJ (mixing all partitions in both train and test)
  - Fit Ridge on ALL train data (probe sees ID + OOD)
  - Evaluate on the 20% test set, broken down per partition

Targets (uniform_motion is 1D — vy ≡ 0, y constant per traj):
  - pos_x: proprio[:, 0]
  - vel_x: action[:, 0]   (action = synthetic velocity from pos diff)

Compares two frozen encoders:
  - lewm-pusht (5.5M, frozen, official PushT weights)        [no_projector cls token]
  - DiT-XL zero-shot (749M, frozen, ImageNet pretrained)     [last-block pooled]

Usage:
    python scripts/probe_ood_uniform_fullfit.py --encoder lewm_pusht
    python scripts/probe_ood_uniform_fullfit.py --encoder dit_xl
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

LEWM_ROOT = Path(__file__).resolve().parent.parent.parent / "le-wm"
sys.path.insert(0, str(LEWM_ROOT))

PART_NAMES = {0: "ID", 1: "r-OOD", 2: "v-OOD", 3: "both-OOD"}


def partition_label(r: float, v: float, r_id=(0.7, 1.5), v_id=(1.0, 4.0)) -> int:
    r_ok = r_id[0] <= r <= r_id[1]
    v_ok = v_id[0] <= abs(v) <= v_id[1]
    if r_ok and v_ok: return 0
    if not r_ok and v_ok: return 1
    if r_ok and not v_ok: return 2
    return 3


def build_lewm_pusht_model():
    import stable_pretraining as spt
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False)
    hidden = encoder.config.hidden_size
    predictor = ARPredictor(
        num_frames=3, input_dim=hidden, hidden_dim=hidden, output_dim=hidden,
        depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0)
    action_encoder = Embedder(input_dim=10, emb_dim=hidden)
    projector = MLP(input_dim=hidden, output_dim=hidden, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden, output_dim=hidden, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    return JEPA(encoder=encoder, predictor=predictor, action_encoder=action_encoder,
                projector=projector, pred_proj=pred_proj)


@torch.no_grad()
def extract_lewm_embeddings(pix_uint8_chw, weights_path, *, batch_size=128, device="cuda"):
    model = build_lewm_pusht_model()
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded keys: {len(sd) - len(unexpected)} / {len(sd)} "
          f"(unexpected={len(unexpected)}, missing={len(missing)})")

    encoder = model.encoder.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    N = pix_uint8_chw.shape[0]
    out = []
    for i in range(0, N, batch_size):
        x = pix_uint8_chw[i:i + batch_size].to(device, non_blocking=True).float() / 255.0
        x = (x - mean) / std
        h = encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        out.append(h.float().cpu().numpy())
        if (i // batch_size) % 50 == 0:
            print(f"    encoded {i + x.shape[0]}/{N}", flush=True)
    return np.concatenate(out, axis=0)


@torch.no_grad()
def extract_dit_embeddings(pix_uint8_chw, dit_dir, *, batch_size=8, device="cuda",
                            img_size=256, dtype=torch.float16):
    from diffusers import AutoencoderKL, DiTTransformer2DModel
    vae = AutoencoderKL.from_pretrained(dit_dir, subfolder="vae",
                                         torch_dtype=dtype).to(device).eval()
    transformer = DiTTransformer2DModel.from_pretrained(dit_dir, subfolder="transformer",
                                                        torch_dtype=dtype).to(device).eval()
    hidden = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    N = pix_uint8_chw.shape[0]
    embs = np.empty((N, hidden), dtype=np.float32)
    for i in range(0, N, batch_size):
        x_u = pix_uint8_chw[i:i + batch_size].to(device, non_blocking=True).to(dtype)
        x = (x_u / 127.5 - 1.0)
        if x.shape[-1] != img_size:
            x = torch.nn.functional.interpolate(x, size=img_size, mode="bilinear",
                                                 align_corners=False)
        latent = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
        B = latent.shape[0]
        timestep = torch.zeros(B, dtype=torch.long, device=device)
        class_labels = torch.full((B,), 1000, dtype=torch.long, device=device)
        feats = {}
        def hook(_m, _in, out):
            feats["last"] = out if isinstance(out, torch.Tensor) else out[0]
        h_handle = transformer.transformer_blocks[-1].register_forward_hook(hook)
        try:
            transformer(latent, timestep=timestep, class_labels=class_labels,
                       return_dict=False)
        finally:
            h_handle.remove()
        pooled = feats["last"].mean(dim=1)
        embs[i:i + B] = pooled.float().cpu().numpy()
        if (i // batch_size) % 100 == 0:
            print(f"    DiT encoded {i + B}/{N}", flush=True)
    return embs


def build_multiframe_feats(emb_per_frame, ep_idx, step_idx, K):
    N, D = emb_per_frame.shape
    feats = np.zeros((N, K * D), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)
    by_ep = {}
    for i in range(N):
        by_ep.setdefault(int(ep_idx[i]), []).append((int(step_idx[i]), i))
    for ep, lst in by_ep.items():
        lst.sort()
        ordered = [idx for _, idx in lst]
        for k, idx in enumerate(ordered):
            if k >= K - 1:
                ctx = ordered[k - K + 1:k + 1]
                feats[idx] = np.concatenate([emb_per_frame[j] for j in ctx])
                valid[idx] = True
    return feats, valid


def fit_and_eval(emb, label, prop, action, ep_idx, part, *,
                 K=1, step_idx=None, train_frac=0.8, seed=0):
    """Per-partition Ridge probe. Accepts 1D or 2D pos/vel targets.

    Auto-detects which dims have signal (train std > 1e-6) and reports
    per-dim R² + Pearson ρ + normalized MSE for those dims only.
    """
    if K > 1:
        feats, valid = build_multiframe_feats(emb, ep_idx, step_idx, K)
    else:
        feats = emb
        valid = np.ones(len(emb), dtype=bool)

    rng = np.random.default_rng(seed)
    uniq_eps = np.unique(ep_idx)
    perm = rng.permutation(uniq_eps)
    n_tr = int(round(len(uniq_eps) * train_frac))
    train_eps = set(perm[:n_tr].tolist())
    base_tr = np.array([e in train_eps for e in ep_idx])
    base_te = ~base_tr
    mask_tr = base_tr & valid
    mask_te = base_te & valid

    mu, sigma = feats[mask_tr].mean(0), feats[mask_tr].std(0) + 1e-6
    e_tr = (feats[mask_tr] - mu) / sigma

    print(f"\n=== {label}  [K={K}] ===  feat dim={feats.shape[1]}")
    print(f"  train: {mask_tr.sum()} frames (mixed partitions)")
    print(f"  test : {mask_te.sum()} frames")
    print(f"  train partition mix: " +
          ", ".join(f"{PART_NAMES[p]}={int(((part[mask_tr]==p)).sum())}"
                    for p in sorted(np.unique(part))))

    # auto-detect which target dims have signal
    pos_dims = [d for d in range(prop.shape[1]) if prop[mask_tr][:, d].std() > 1e-6]
    vel_dims = [d for d in range(action.shape[1]) if action[mask_tr][:, d].std() > 1e-6]
    pos_names = ["pos_x", "pos_y"][:prop.shape[1]]
    vel_names = ["vx", "vy"][:action.shape[1]]
    print(f"  active target dims: pos={[pos_names[d] for d in pos_dims]}, "
          f"vel={[vel_names[d] for d in vel_dims]}")

    pos_train_std = prop[mask_tr].std(0)
    vel_train_std = action[mask_tr].std(0)

    ridge_pos = Ridge(alpha=1.0); ridge_pos.fit(e_tr, prop[mask_tr])
    ridge_vel = Ridge(alpha=1.0); ridge_vel.fit(e_tr, action[mask_tr])

    # build header: aggregate + per-partition rows, per-dim columns
    header = f"  {'partition':12s} {'n':>5s}"
    for d in pos_dims:
        header += f"  {pos_names[d]+' nMSE':>10s} {pos_names[d]+' ρ':>9s}"
    for d in vel_dims:
        header += f"  {vel_names[d]+' nMSE':>10s} {vel_names[d]+' ρ':>9s}"
    print(header)

    def per_dim_metrics(true_full, pred_full, dims, train_std):
        out = []
        for d in dims:
            t = true_full[:, d]; p = pred_full[:, d]
            sd = train_std[d]
            nmse = float((((p - t) / sd) ** 2).mean())
            rho = float(np.corrcoef(t, p)[0, 1]) if t.std() > 1e-9 else float('nan')
            out.append((nmse, rho))
        return out

    rows = [('AGGREGATE', mask_te)]
    for p in sorted(np.unique(part)):
        pm = mask_te & (part == p)
        if pm.sum() >= 50:
            rows.append((PART_NAMES[p], pm))
    for name, m in rows:
        e_p = (feats[m] - mu) / sigma
        pred_p = ridge_pos.predict(e_p); pred_v = ridge_vel.predict(e_p)
        # ensure 2D shape even for single-output Ridge
        if pred_p.ndim == 1: pred_p = pred_p[:, None]
        if pred_v.ndim == 1: pred_v = pred_v[:, None]
        pm_pos = per_dim_metrics(prop[m], pred_p, pos_dims, pos_train_std)
        pm_vel = per_dim_metrics(action[m], pred_v, vel_dims, vel_train_std)
        row = f"  {name:12s} {m.sum():>5d}"
        for nmse, rho in pm_pos: row += f"  {nmse:>10.4f} {rho:>+9.4f}"
        for nmse, rho in pm_vel: row += f"  {nmse:>10.4f} {rho:>+9.4f}"
        print(row)


def load_uniform_with_partitions(lewm_path, source_path):
    with h5py.File(lewm_path, "r") as f:
        pix = f["pixels"][:]
        prop = f["proprio"][:]
        action = f["action"][:]
        ep_idx = f["episode_idx"][:]
        step_idx = f["step_idx"][:]
        ep_len = f["ep_len"][:]
    pix_u8 = torch.from_numpy(pix).permute(0, 3, 1, 2).contiguous()

    with h5py.File(source_path, "r") as f:
        init = np.concatenate(
            [f["init_streams"][k][...] for k in sorted(f["init_streams"])], 0)
    n_traj = len(init)
    assert n_traj == len(ep_len), f"traj mismatch: lewm {len(ep_len)} vs source {n_traj}"

    parts_per_traj = np.array(
        [partition_label(float(init[i, 0]), float(init[i, 1])) for i in range(n_traj)],
        dtype=np.uint8)
    parts_per_frame = parts_per_traj[ep_idx]
    print(f"  partitions (trajs): " +
          str({PART_NAMES[k]: int((parts_per_traj == k).sum()) for k in range(4)}))
    return pix_u8, prop, action, ep_idx, step_idx, parts_per_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", choices=["lewm_pusht", "dit_xl"], required=True)
    ap.add_argument("--dataset-tag", default="uniform_motion",
        help="dataset name used in emb cache file name (e.g. 'uniform_motion', 'parabola')")
    ap.add_argument("--lewm-data",
        default=os.path.expanduser("~/.stable_worldmodel/phyworld_uniform_motion.h5"))
    ap.add_argument("--source-hdf5",
        default="/home/qlib/agent_memory/wm/phyworld/data/uniform_motion_eval.hdf5")
    ap.add_argument("--lewm-weights",
        default="/home/qlib/.stable_worldmodel/lewm_paper_pusht/weights.pt")
    ap.add_argument("--dit-dir",
        default="/home/qlib/.stable_worldmodel/dit_xl_2_256")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="0 = encoder-specific default (lewm=128, dit=8)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[load] uniform_motion h5 + partitions ...", flush=True)
    pix_chw, prop, action, ep_idx, step_idx, parts = load_uniform_with_partitions(
        args.lewm_data, args.source_hdf5)
    print(f"  {pix_chw.shape[0]} frames", flush=True)

    # always use full 2D (x, y); fit_and_eval auto-skips dims with zero variance
    prop_xy = prop[:, :2].astype(np.float32)
    vel_xy = action[:, :2].astype(np.float32)

    cache_dir = os.path.expanduser("~/agent_memory/wm/artifacts/embeddings")
    os.makedirs(cache_dir, exist_ok=True)

    if args.encoder == "lewm_pusht":
        emb_cache = os.path.join(cache_dir,
            f"lewm_pusht_only_{args.dataset_tag}_emb_noproj.npy")
        label = "LeWM pusht-only (frozen, 5.5M)"
        if os.path.exists(emb_cache):
            print(f"[cache] loading {emb_cache}", flush=True)
            emb = np.load(emb_cache)
        else:
            print(f"\n[encode] LeWM pusht-only (frozen) ...", flush=True)
            bs = args.batch_size or 128
            emb = extract_lewm_embeddings(pix_chw, args.lewm_weights,
                                          batch_size=bs, device=args.device)
            np.save(emb_cache, emb)
            print(f"  cached to {emb_cache}", flush=True)
    else:
        emb_cache = os.path.join(cache_dir,
            f"dit_xl_zeroshot_{args.dataset_tag}_emb.npy")
        label = "DiT-XL zero-shot (frozen, 749.8M)"
        if os.path.exists(emb_cache):
            print(f"[cache] loading {emb_cache}", flush=True)
            emb = np.load(emb_cache)
        else:
            print(f"\n[encode] DiT-XL zero-shot ...", flush=True)
            bs = args.batch_size or 8
            emb = extract_dit_embeddings(pix_chw, args.dit_dir,
                                          batch_size=bs, device=args.device)
            np.save(emb_cache, emb)
            print(f"  cached to {emb_cache}", flush=True)

    print(f"  emb shape: {emb.shape}")

    for K in (1, 4):
        print(f"\n{'=' * 60}\n  K={K} probe (mixed-fit Ridge on 80% ID+OOD)\n{'=' * 60}")
        fit_and_eval(emb, label, prop_xy, vel_xy, ep_idx, parts,
                     K=K, step_idx=step_idx, seed=args.seed)

    print(f"\nTotal {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
