"""Zero-shot cross-domain probe on phyworld combinatorial OOT eval.

Tests whether our **collision-trained encoder** (which only ever saw 2-ball
elastic collisions) produces meaningful representations on **PHYRE**
multi-object scenes (5 objects of different types per scene).

This is NOT combinatorial generalization in the phyworld paper's sense
(which would require training our model on subset of PHYRE templates and
testing on held-out templates). It IS a cross-domain transfer test:
does encoder learn features that survive a huge domain shift?

Probe targets (frame-level):
  - scene_centroid_x / y : mean (x, y) over the 5 objects -- regression
  - n_dynamic_objects    : count of "dynamic" objects in frame  -- regression
  - template_id          : 10-way (one-hot 0-9 for templates 10060-10069) -- classification

Split protocol (combinatorial-style):
  - 10 OOT templates -> 8 train / 2 test by template (each template has 100
    samples × 50 frames = 5000 frames; total ~ 50k frames in eval set).
  - Probe Ridge / LogReg fit on train-templates frames; eval on test-templates.

Usage:
    python scripts/probe_combinatorial_zeroshot.py \\
        --ckpt ~/.stable_worldmodel/collision_paperinit/lewm_collision_paperinit_epoch_8_object.ckpt \\
        --data ~/.stable_worldmodel/phyworld_combinatorial/combinatorial_data/combinatorial_out_of_template_eval_1K.hdf5
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

LEWM_ROOT = Path(__file__).resolve().parent.parent.parent / "le-wm"
sys.path.insert(0, str(LEWM_ROOT))

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_random_encoder(img_size=224, patch_size=14, scale="tiny"):
    import stable_pretraining as spt
    return spt.backbone.utils.vit_hf(scale, patch_size=patch_size, image_size=img_size,
                                      pretrained=False, use_mask_token=False)


def decode_mp4_bytes(blob: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tf.write(blob); path = tf.name
    try:
        return iio.imread(path)
    finally:
        os.unlink(path)


def resize_frame(f: np.ndarray, size: int) -> np.ndarray:
    if f.shape[0] == size and f.shape[1] == size:
        return f
    return np.asarray(Image.fromarray(f).resize((size, size), Image.BILINEAR))


def load_combinatorial(path, img_size=224, frame_stride=5, max_samples=None):
    """Decode all sample videos, return (pixels_uint8, targets, ep_idx)."""
    print(f"[load] {path}", flush=True)
    pixels, x_cent, y_cent, n_dyn, tpl, ep_idx, sample_keys = [], [], [], [], [], [], []
    with h5py.File(path, "r") as f:
        keys = sorted(f["video_streams"].keys())
        if max_samples is not None:
            keys = keys[:max_samples]
        for s_i, k in enumerate(keys):
            template = int(k.split(":")[0])
            tid = template - 10060  # 0..9
            vbytes = bytes(f[f"video_streams/{k}"][:][0])
            frames = decode_mp4_bytes(vbytes)  # (T, H, W, 3)
            obj = f[f"object_streams/{k}"][:][0]  # (T, 5, 14)
            T = frames.shape[0]
            # subsample frames
            for t in range(0, T, frame_stride):
                fr = resize_frame(frames[t], img_size)
                pixels.append(fr)
                # object positions (cols 0,1), filter to dynamic (col 12 = 1)
                # cols 4..11 are type one-hot; sum of (4..11) > 0 → real object;
                # col 12 (which we observed always =1 for present objects) we use
                # as "is present"
                ob_t = obj[t]  # (5, 14)
                present = ob_t[:, 12] > 0.5
                if present.sum() > 0:
                    xc = float(ob_t[present, 0].mean())
                    yc = float(ob_t[present, 1].mean())
                else:
                    xc = yc = 0.5
                x_cent.append(xc); y_cent.append(yc)
                n_dyn.append(int(present.sum()))
                tpl.append(tid)
                ep_idx.append(s_i)
                sample_keys.append(k)
            if (s_i + 1) % 100 == 0:
                print(f"  {s_i + 1}/{len(keys)}  frames so far: {len(pixels)}", flush=True)

    pix = torch.from_numpy(np.stack(pixels)).permute(0, 3, 1, 2).contiguous()
    return (pix,
            np.asarray(x_cent, dtype=np.float32),
            np.asarray(y_cent, dtype=np.float32),
            np.asarray(n_dyn, dtype=np.float32),
            np.asarray(tpl,   dtype=np.int64),
            np.asarray(ep_idx, dtype=np.int64))


@torch.no_grad()
def extract_embeddings(model, pix_uint8, *, batch_size=64, device="cuda",
                       use_projector=True, img_size=224):
    if hasattr(model, "embeddings"):
        encoder, projector = model, None
    elif hasattr(model, "encoder"):
        encoder = model.encoder
        projector = getattr(model, "projector", None) if use_projector else None
    else:
        encoder, projector = model, None
    encoder.eval().to(device)
    if projector is not None:
        projector.eval().to(device)
    mean = _MEAN.to(device); std = _STD.to(device)
    N = pix_uint8.shape[0]
    out = []
    for i in range(0, N, batch_size):
        x = pix_uint8[i:i + batch_size].to(device, non_blocking=True).float() / 255.0
        x = (x - mean) / std
        if x.shape[-1] != img_size:
            x = torch.nn.functional.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
        h = encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        if projector is not None:
            h = projector(h)
        out.append(h.float().cpu().numpy())
        if (i // batch_size) % 100 == 0:
            print(f"    encoded {i + x.shape[0]}/{N}", flush=True)
    return np.concatenate(out, axis=0)


def make_pixel_stats(pix_uint8):
    N = pix_uint8.shape[0]
    feats = np.empty((N, 9), dtype=np.float32)
    for i in range(0, N, 512):
        x = pix_uint8[i:i + 512].float() / 255.0
        flat = x.flatten(2)
        m = flat.mean(-1); s = flat.std(-1)
        feats[i:i + x.shape[0]] = torch.cat([m, s, m ** 2], dim=1).numpy()
    return feats


def evaluate(label, emb, mask_tr, mask_te, targets):
    print(f"\n=== {label} ===  emb dim={emb.shape[1]}, train frames={mask_tr.sum()}, test frames={mask_te.sum()}", flush=True)
    mu, sigma = emb[mask_tr].mean(0), emb[mask_tr].std(0) + 1e-6
    e_tr = (emb[mask_tr] - mu) / sigma
    e_te = (emb[mask_te] - mu) / sigma
    for name, (kind, y) in targets.items():
        y_tr, y_te = y[mask_tr], y[mask_te]
        if kind == "regression":
            m = Ridge(alpha=1.0); m.fit(e_tr, y_tr)
            r2 = r2_score(y_te, m.predict(e_te))
            print(f"  {name:24s}  R²={r2:+.4f}")
        else:  # classification
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(e_tr, y_tr)
            acc = accuracy_score(y_te, clf.predict(e_te))
            n_cls = len(np.unique(y_tr))
            print(f"  {name:24s}  acc={acc:.4f}  (chance ≈ {1/n_cls:.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.expanduser(
        "~/.stable_worldmodel/phyworld_combinatorial/combinatorial_data/combinatorial_out_of_template_eval_1K.hdf5"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--frame-stride", type=int, default=5, help="sample every Nth frame")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--no-projector", action="store_true")
    ap.add_argument("--skip-random", action="store_true")
    ap.add_argument("--skip-pixel", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    pix, xc, yc, n_dyn, tpl, ep_idx = load_combinatorial(
        args.data, img_size=224, frame_stride=args.frame_stride, max_samples=args.max_samples)
    print(f"  pix shape: {pix.shape}, templates: {dict(zip(*np.unique(tpl, return_counts=True)))}")

    # split by template (8 train, 2 test) -- combinatorial-style
    rng = np.random.default_rng(args.seed)
    all_tpl = np.unique(tpl)
    perm = rng.permutation(all_tpl)
    test_tpl = perm[:2]; train_tpl = perm[2:]
    print(f"  train templates: {train_tpl.tolist()}, test templates: {test_tpl.tolist()}")
    mask_tr = np.isin(tpl, train_tpl)
    mask_te = np.isin(tpl, test_tpl)

    targets = {
        "scene_centroid_x":    ("regression",    xc),
        "scene_centroid_y":    ("regression",    yc),
        "n_present_objects":   ("regression",    n_dyn),
        "template_id (10way)": ("classification", tpl),
    }

    # 1) trained encoder
    print(f"\n[encoder] {args.ckpt}", flush=True)
    trained = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"  loaded {type(trained).__name__}")
    emb = extract_embeddings(trained, pix, batch_size=args.batch_size,
                             device=args.device, use_projector=not args.no_projector)
    evaluate("trained encoder (collision_paperinit)", emb, mask_tr, mask_te, targets)
    del trained, emb; torch.cuda.empty_cache()

    # 2) random
    if not args.skip_random:
        print(f"\n[encoder] random ViT-tiny", flush=True)
        rand = build_random_encoder()
        emb = extract_embeddings(rand, pix, batch_size=args.batch_size,
                                 device=args.device, use_projector=False)
        evaluate("random encoder", emb, mask_tr, mask_te, targets)
        del rand, emb; torch.cuda.empty_cache()

    # 3) pixel-stats
    if not args.skip_pixel:
        print(f"\n[encoder] pixel-stats (9-D)", flush=True)
        feat = make_pixel_stats(pix)
        evaluate("pixel-stats", feat, mask_tr, mask_te, targets)

    print(f"\nTotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
