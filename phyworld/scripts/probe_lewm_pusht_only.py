"""Probe lewm-pusht weights *without* phyworld fine-tuning.

This is the "frozen pretrained" baseline complementing:
  · LeWM from-scratch + collision-trained  (COLLISION_REPORT §2.2-2.4)
  · LeWM paper-init   + collision-trained  (COLLISION_REPORT §2.5)
  · DiT-XL-2 zero-shot                     (DIT_REPORT.md)

Here we take the *unmodified* PushT-pretrained checkpoint from
`quentinll/lewm-pusht` and probe directly on phyworld_collision. This isolates
"PushT visual pretraining alone" from "PushT init + collision fine-tune".

Same protocol as probe_dit_zeroshot.py: 32k frames over 1000 trajs, 80/20
episode split, K=1 single-frame + K=4 multi-frame Ridge / LogReg probes.
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
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score

LEWM_ROOT = Path(__file__).resolve().parent.parent.parent / "le-wm"
sys.path.insert(0, str(LEWM_ROOT))


def build_lewm_pusht_model():
    """Rebuild the JEPA architecture matching lewm_paper_pusht/config.json."""
    import stable_pretraining as spt
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False,
    )
    hidden = encoder.config.hidden_size  # 192
    # Predictor params from lewm_paper_pusht/config.json
    predictor = ARPredictor(
        num_frames=3, input_dim=hidden, hidden_dim=hidden, output_dim=hidden,
        depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0,
    )
    # action_encoder: PushT had 10-D action. Not used for probing, just shape-matching.
    action_encoder = Embedder(input_dim=10, emb_dim=hidden)

    projector = MLP(input_dim=hidden, output_dim=hidden, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden, output_dim=hidden, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)

    model = JEPA(
        encoder=encoder, predictor=predictor, action_encoder=action_encoder,
        projector=projector, pred_proj=pred_proj,
    )
    return model


@torch.no_grad()
def extract_embeddings(model, pix_uint8_chw, *, use_projector, batch_size=128, device="cuda"):
    """Run encoder over all frames, return (N, D) cls-token embedding.

    `pix_uint8_chw`: (N, 3, H, W) uint8 from h5 'pixels' transposed.
    """
    encoder = model.encoder
    projector = model.projector if use_projector else None

    encoder.eval().to(device)
    if projector is not None:
        projector.eval().to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    N = pix_uint8_chw.shape[0]
    out = []
    for i in range(0, N, batch_size):
        x = pix_uint8_chw[i:i + batch_size].to(device, non_blocking=True).float() / 255.0
        x = (x - mean) / std
        h = encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]  # cls
        if projector is not None:
            h = projector(h)
        out.append(h.float().cpu().numpy())
        if (i // batch_size) % 50 == 0:
            print(f"  encoded {i + x.shape[0]}/{N}", flush=True)
    return np.concatenate(out, axis=0)


def split_episodes(n_episodes, train_frac=0.8, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    n_train = int(round(n_episodes * train_frac))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


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


def fit_eval_collision(emb, prop, state, coll, ep_idx, n_ep, *, train_frac=0.8, seed=0, K=1, valid_mask=None):
    train_eps, test_eps = split_episodes(n_ep, train_frac, seed)
    base_tr = np.isin(ep_idx, train_eps)
    base_te = np.isin(ep_idx, test_eps)
    if valid_mask is None:
        valid_mask = np.ones(len(ep_idx), dtype=bool)
    mask_tr = base_tr & valid_mask
    mask_te = base_te & valid_mask
    mu, sigma = emb[mask_tr].mean(0), emb[mask_tr].std(0) + 1e-6
    e_tr = (emb[mask_tr] - mu) / sigma
    e_te = (emb[mask_te] - mu) / sigma

    pos_x = prop[:, [0, 2]]
    vel_x = state[:, [0, 2]]
    print(f"  [K={K}] train={mask_tr.sum()}, test={mask_te.sum()}, feat_dim={emb.shape[1]}")
    print(f"  {'target':18s}  R² / AUC")
    m = Ridge(alpha=1.0); m.fit(e_tr, pos_x[mask_tr])
    print(f"  {'pos_x R²':18s}  {r2_score(pos_x[mask_te], m.predict(e_te)):+.4f}")
    m = Ridge(alpha=1.0); m.fit(e_tr, vel_x[mask_tr])
    print(f"  {'vel_x R²':18s}  {r2_score(vel_x[mask_te], m.predict(e_te)):+.4f}")
    if coll[mask_tr].sum() > 1 and coll[mask_tr].sum() < mask_tr.sum():
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
        clf.fit(e_tr, coll[mask_tr])
        p = clf.predict_proba(e_te)[:, 1]
        print(f"  {'collision AUC':18s}  {roc_auc_score(coll[mask_te], p):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default="/home/qlib/.stable_worldmodel/lewm_paper_pusht/weights.pt")
    ap.add_argument("--data", default=os.path.expanduser("~/.stable_worldmodel/phyworld_collision.h5"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-frames", type=int, default=32000,
                    help="Match DiT zero-shot subset (1000 trajs × 32 frames).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-projector", action="store_true",
                    help="Probe raw ViT cls token; matches the 'no-projector' setting in COLLISION_REPORT §2.5")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[build] lewm-pusht JEPA architecture", flush=True)
    model = build_lewm_pusht_model()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  JEPA total: {n_params:.2f} M params")

    print(f"[load] weights from {args.weights}", flush=True)
    sd = torch.load(args.weights, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded keys: {len(sd) - len(unexpected)} / {len(sd)} (unexpected={len(unexpected)}, missing={len(missing)})")
    if missing[:5]:
        print(f"  first missing: {missing[:5]}")
    if unexpected[:5]:
        print(f"  first unexpected: {unexpected[:5]}")

    print(f"[load] data {args.data}", flush=True)
    with h5py.File(args.data, "r") as f:
        N_total = f["pixels"].shape[0]
        N = min(args.max_frames, N_total)
        pix = f["pixels"][:N]
        prop = f["proprio"][:N]
        state = f["state"][:N]
        coll = f["collision_event"][:N]
        ep_idx = f["episode_idx"][:N]
        step_idx = f["step_idx"][:N]
    pix_chw = torch.from_numpy(pix).permute(0, 3, 1, 2).contiguous()
    n_ep = len(np.unique(ep_idx))
    print(f"  using {N} frames over {n_ep} trajs", flush=True)

    use_proj = not args.no_projector
    proj_tag = "no_projector" if args.no_projector else "with_projector"
    cache_dir = os.path.expanduser("~/agent_memory/wm/artifacts/embeddings")
    os.makedirs(cache_dir, exist_ok=True)
    emb_cache = os.path.join(cache_dir, f"lewm_pusht_only_collision_emb_32k_{proj_tag}.npy")
    if os.path.exists(emb_cache):
        print(f"[cache] loading {emb_cache}", flush=True)
        emb = np.load(emb_cache)
    else:
        print(f"\n[encode] lewm-pusht (frozen, no phyworld FT) ...", flush=True)
        emb = extract_embeddings(model, pix_chw, use_projector=use_proj,
                                  batch_size=args.batch_size, device=args.device)
        np.save(emb_cache, emb)
        print(f"  cached to {emb_cache}", flush=True)
    print(f"  emb shape: {emb.shape}, projector={'on' if use_proj else 'off'}")

    print(f"\n=== lewm-pusht ZERO-SHOT (no phyworld FT, projector={'on' if use_proj else 'off'}) ===")
    print(f"\n[K=1 single-frame]")
    fit_eval_collision(emb, prop, state, coll, ep_idx, n_ep, seed=args.seed, K=1)

    print(f"\n[K=4 multi-frame]")
    feats_k4, valid_k4 = build_multiframe_feats(emb, ep_idx, step_idx, K=4)
    fit_eval_collision(feats_k4, prop, state, coll, ep_idx, n_ep, seed=args.seed, K=4, valid_mask=valid_k4)

    print(f"\nTotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
