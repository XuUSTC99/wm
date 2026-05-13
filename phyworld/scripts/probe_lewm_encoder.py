"""Probe whether le-wm's encoder captures physical quantities on phyworld.

Methodology
-----------
1. Load a trained le-wm checkpoint (the ModelObjectCallBack-saved object .ckpt).
2. Forward every frame of the phyworld dataset through the encoder + projector
   to get one embedding per frame (D = projector output dim, default 192).
3. Split trajectories (NOT frames) into train/test 80/20.
4. Fit a linear regression  emb -> target  on the train trajectories' frames,
   evaluate R^2 on the test trajectories' frames.

Targets
-------
- position : the (x, y) ball position from `proprio` -- direct test of whether
  the encoder localises the object.
- velocity : `pos[t+1] - pos[t]` -- requires aggregating motion (single-frame
  emb cannot infer velocity, so this should fail; included as a sanity ceiling
  showing what the cls-token CAN'T do without temporal context).
- speed    : ||velocity||_2 -- a scalar physical quantity.

Baselines
---------
- Trained encoder      : the loaded checkpoint.
- Random encoder       : a freshly initialised ViT with the same config (control
  for "how much R^2 do you get just from random features?").
- Raw pixel mean       : 3-D mean of pixel values per frame (sanity floor).

Run
---
    python scripts/probe_lewm_encoder.py \
        --ckpt ~/.stable_worldmodel/phyworld_probe/lewm_phyworld_epoch_20_object.ckpt \
        --data ~/.stable_worldmodel/phyworld_uniform_motion.h5
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


# le-wm modules live in the project root, not on sys.path by default
LEWM_ROOT = Path(__file__).resolve().parent.parent.parent / "le-wm"
sys.path.insert(0, str(LEWM_ROOT))


def build_random_encoder(img_size: int = 224, patch_size: int = 14, scale: str = "tiny"):
    """Mirror what train.py builds, with random init and no projector."""
    import stable_pretraining as spt

    encoder = spt.backbone.utils.vit_hf(
        scale, patch_size=patch_size, image_size=img_size,
        pretrained=False, use_mask_token=False,
    )
    return encoder


@torch.no_grad()
def extract_embeddings(model, pixels_norm: torch.Tensor, *, batch_size: int = 128,
                       device: str = "cuda", use_projector: bool = True) -> np.ndarray:
    """Run encoder over all frames, return (N, D) cls-token embeddings.

    `model` may be a JEPA (with .encoder, .projector) or a bare encoder.
    `pixels_norm` is already ImageNet-normalised, shape (N, 3, H, W) float.
    """
    # JEPA wraps a ViTModel as `.encoder`; a raw ViTModel has its own `.encoder`
    # (the inner ViTEncoder stack), so we must dispatch on whether `model` is
    # itself the ViT (has `.embeddings`) vs a JEPA wrapper (has `.encoder`).
    if hasattr(model, "embeddings"):  # ViTModel
        encoder = model
        projector = None
    elif hasattr(model, "encoder"):    # JEPA-like wrapper
        encoder = model.encoder
        projector = getattr(model, "projector", None) if use_projector else None
    else:
        encoder = model
        projector = None

    encoder.eval().to(device)
    if projector is not None:
        projector.eval().to(device)

    N = pixels_norm.shape[0]
    out = []
    for i in range(0, N, batch_size):
        x = pixels_norm[i : i + batch_size].to(device, non_blocking=True)
        h = encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]  # cls token
        if projector is not None:
            h = projector(h)
        out.append(h.float().cpu().numpy())
        if (i // batch_size) % 50 == 0:
            print(f"  encoded {i+x.shape[0]}/{N}", flush=True)
    return np.concatenate(out, axis=0)


def load_pixels_normalised(h5_path: Path, img_size: int = 224) -> torch.Tensor:
    """Read all pixels, ImageNet-normalise them, return (N, 3, img_size, img_size).

    Mimics utils.get_img_preprocessor (ToImage + Resize). Pixels are already
    img_size in our converted phyworld file, so resize is a no-op.
    """
    with h5py.File(h5_path, "r") as f:
        pix = f["pixels"][:]  # (N, H, W, 3) uint8
        positions = f["proprio"][:]  # (N, 2) float32
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        episode_idx = f["episode_idx"][:]

    # uint8 (N, H, W, 3) -> float (N, 3, H, W) ImageNet-normalised
    x = torch.from_numpy(pix).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    if x.shape[-1] != img_size:
        x = torch.nn.functional.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
    return x, positions, ep_len, ep_offset, episode_idx


def split_episodes(n_episodes: int, train_frac: float = 0.8, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    n_train = int(round(n_episodes * train_frac))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def make_velocity(positions: np.ndarray, ep_offset: np.ndarray, ep_len: np.ndarray) -> np.ndarray:
    """Per-episode forward-difference velocity. Last frame uses previous step."""
    vel = np.empty_like(positions)
    for off, L in zip(ep_offset, ep_len):
        seg = positions[off : off + L]
        v = np.empty_like(seg)
        v[:-1] = seg[1:] - seg[:-1]
        v[-1] = v[-2]
        vel[off : off + L] = v
    return vel


def fit_probe(emb_train: np.ndarray, y_train: np.ndarray,
              emb_test: np.ndarray, y_test: np.ndarray,
              alpha: float = 1.0) -> dict:
    """Fit Ridge regression and report per-target & overall R^2."""
    model = Ridge(alpha=alpha)
    model.fit(emb_train, y_train)
    y_pred = model.predict(emb_test)
    r2_uniform = r2_score(y_test, y_pred, multioutput="uniform_average")
    r2_per_dim = r2_score(y_test, y_pred, multioutput="raw_values") if y_test.ndim > 1 else None
    rmse = float(np.sqrt(((y_pred - y_test) ** 2).mean()))
    return dict(r2=float(r2_uniform), r2_per_dim=r2_per_dim, rmse=rmse)


def select_frames(arr: np.ndarray, episode_idx: np.ndarray, eps: np.ndarray) -> np.ndarray:
    mask = np.isin(episode_idx, eps)
    return arr[mask], mask


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="path to *_object.ckpt (full le-wm model)")
    ap.add_argument("--data", default=os.path.expanduser("~/.stable_worldmodel/phyworld_uniform_motion.h5"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge regularisation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--no-projector", action="store_true",
                    help="probe the raw ViT cls token instead of post-projector emb")
    ap.add_argument("--skip-random", action="store_true", help="skip the random-encoder baseline")
    ap.add_argument("--skip-pixel", action="store_true", help="skip the pixel-mean baseline")
    args = ap.parse_args()

    print(f"loading data from {args.data} ...", flush=True)
    t0 = time.time()
    pixels_norm, positions, ep_len, ep_offset, episode_idx = load_pixels_normalised(Path(args.data))
    n_ep = len(ep_len)
    n_frames = pixels_norm.shape[0]
    print(f"  {n_ep} episodes, {n_frames} frames, {pixels_norm.shape[2]}x{pixels_norm.shape[3]} pixels  ({time.time()-t0:.1f}s)")

    velocity = make_velocity(positions, ep_offset, ep_len)
    speed = np.linalg.norm(velocity, axis=1, keepdims=True).astype(np.float32)

    train_eps, test_eps = split_episodes(n_ep, args.train_frac, args.seed)
    print(f"  split: {len(train_eps)} train eps / {len(test_eps)} test eps")

    pos_tr, mask_tr = select_frames(positions, episode_idx, train_eps)
    pos_te, mask_te = select_frames(positions, episode_idx, test_eps)
    vel_tr = velocity[mask_tr]
    vel_te = velocity[mask_te]
    sp_tr = speed[mask_tr]
    sp_te = speed[mask_te]
    print(f"  train frames: {mask_tr.sum()}  test frames: {mask_te.sum()}")

    targets = {"position": (pos_tr, pos_te), "velocity": (vel_tr, vel_te), "speed": (sp_tr, sp_te)}

    def evaluate(model, label):
        print(f"\n[{label}] extracting embeddings ...", flush=True)
        t0 = time.time()
        emb = extract_embeddings(model, pixels_norm, batch_size=args.batch_size,
                                 device=args.device, use_projector=not args.no_projector)
        print(f"  emb shape: {emb.shape}  ({time.time()-t0:.1f}s)")
        emb_tr = emb[mask_tr]
        emb_te = emb[mask_te]
        # standardise for stable Ridge
        mu = emb_tr.mean(0); sigma = emb_tr.std(0) + 1e-6
        emb_tr = (emb_tr - mu) / sigma
        emb_te = (emb_te - mu) / sigma
        for name, (y_tr, y_te) in targets.items():
            res = fit_probe(emb_tr, y_tr, emb_te, y_te, alpha=args.alpha)
            print(f"  probe {name:9s}: R^2 = {res['r2']:+.4f}  RMSE = {res['rmse']:.4f}"
                  + (f"  per-dim R^2 = {np.round(res['r2_per_dim'], 3).tolist()}"
                     if res["r2_per_dim"] is not None else ""))

    print(f"\nloading trained le-wm from {args.ckpt} ...", flush=True)
    trained = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"  loaded {type(trained).__name__}")
    evaluate(trained, "trained")

    if not args.skip_random:
        print("\nbuilding random-init encoder ...", flush=True)
        rand_enc = build_random_encoder()
        evaluate(rand_enc, "random encoder")

    if not args.skip_pixel:
        # Pixel mean over channels = (N, 3) — too low-D for meaningful Ridge.
        # Use channel means + spatial mean per channel + spatial std per channel = (N, 9)
        print("\n[pixel-stats baseline]", flush=True)
        x = pixels_norm  # already normalised
        flat = x.flatten(2)
        feat = torch.cat([flat.mean(-1), flat.std(-1), flat.mean(-1) ** 2], dim=1).numpy()
        feat_tr = feat[mask_tr]; feat_te = feat[mask_te]
        mu = feat_tr.mean(0); sigma = feat_tr.std(0) + 1e-6
        feat_tr = (feat_tr - mu) / sigma; feat_te = (feat_te - mu) / sigma
        for name, (y_tr, y_te) in targets.items():
            res = fit_probe(feat_tr, y_tr, feat_te, y_te, alpha=args.alpha)
            print(f"  probe {name:9s}: R^2 = {res['r2']:+.4f}  RMSE = {res['rmse']:.4f}")


if __name__ == "__main__":
    main()
