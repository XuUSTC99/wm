"""Universal decoder — train on the FULL distribution (ID + all OOD partitions),
eval per-partition on a held-out split. This isolates the *encoder's* OOD
information-preservation ability, removing the confound where a decoder trained
only on ID latents fails on OOD simply because it never saw OOD appearance.

Pipeline:
  1. load eval h5 (has all 4 partitions) + per-trajectory partition labels
  2. split TRAJECTORIES 80/20 (stratified per partition) — no frame leakage
  3. precompute frozen-encoder embeddings for all frames
  4. train decoder on the train split (ID + OOD mixed)
  5. eval per-partition reconstruction PSNR on the held-out test split

If OOD recon is still bad here (decoder DID see OOD), it's purely the encoder
dropping OOD info — not decoder overfitting.

Usage:
  python train_universal_decoder.py --domain uniform_motion \
    --ckpt <encoder.ckpt> --emb-source cls --epochs 40 \
    --tag pusht --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from decoder import LatentDecoder, build_frozen_encoder  # noqa: E402
from eval_decoder_ood import EVAL, NAMES, frame_partitions, save_grid  # noqa: E402

DATA_DIR = "/data1/likun-share/junjxu/.stable_worldmodel/datasets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="uniform_motion", choices=list(EVAL))
    ap.add_argument("--ckpt", required=True, help="encoder ckpt (weights.pt or JEPA object)")
    ap.add_argument("--emb-source", default="cls", choices=["cls", "proj"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--test-frac", type=float, default=0.2, help="held-out trajectory fraction per partition")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = EVAL[args.domain]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] frozen encoder from {args.ckpt} (emb_source={args.emb_source})", flush=True)
    encode_fn, emb_dim = build_frozen_encoder(args.ckpt, device=device, emb_source=args.emb_source)
    print(f"      emb_dim = {emb_dim}")

    print(f"[2/5] load eval set {cfg['eval_h5']} (all partitions)", flush=True)
    ep_parts = frame_partitions(cfg)  # (num_traj,) partition label per trajectory
    with h5py.File(cfg["eval_h5"], "r") as f:
        pixels = f["pixels"][:]            # (N,224,224,3) uint8
        ep_idx = f["episode_idx"][:]       # (N,) trajectory id per frame
    N = len(pixels)
    parts_frame = ep_parts[ep_idx]         # partition per frame
    print(f"      {N} frames; per-partition frame counts: "
          + ", ".join(f"{NAMES[p]}={int((parts_frame==p).sum())}" for p in sorted(NAMES)))

    # --- 3) stratified trajectory split (no frame leakage) ---
    rng = np.random.default_rng(args.seed)
    uniq_eps = np.unique(ep_idx)
    ep_part_of = {int(e): int(ep_parts[e]) for e in uniq_eps}
    test_eps = set()
    for p in sorted(NAMES):
        eps_p = [e for e in uniq_eps if ep_part_of[int(e)] == p]
        rng.shuffle(eps_p)
        n_test = int(round(len(eps_p) * args.test_frac))
        test_eps.update(int(e) for e in eps_p[:n_test])
    is_test_frame = np.array([int(e) in test_eps for e in ep_idx])
    train_fidx = np.nonzero(~is_test_frame)[0]
    test_fidx = np.nonzero(is_test_frame)[0]
    print(f"      train frames {len(train_fidx)} / test frames {len(test_fidx)} "
          f"(test trajs={len(test_eps)})")

    # --- 4) precompute embeddings for ALL frames (frozen encoder) ---
    print(f"[3/5] precompute embeddings", flush=True)
    pix_t = torch.from_numpy(pixels).permute(0, 3, 1, 2).contiguous()  # (N,3,H,W) uint8 CPU
    emb = torch.empty((N, emb_dim), dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, N, 256):
            chunk = pix_t[i:i + 256].to(device)
            emb[i:i + 256] = encode_fn(chunk).float().cpu()
            if i % 5120 == 0:
                print(f"  embed {i}/{N}", flush=True)
    emb = emb.to(device)

    # --- 5) train decoder on train split (mixed ID+OOD) ---
    print(f"[4/5] train universal decoder, {args.epochs} epochs", flush=True)
    dec = LatentDecoder(emb_dim=emb_dim).to(device)
    train_idx_t = torch.from_numpy(train_fidx)
    dec.set_norm(emb[train_idx_t].mean(0), emb[train_idx_t].std(0))
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    g = torch.Generator().manual_seed(args.seed)

    def per_part_psnr():
        dec.eval()
        res = {}
        with torch.no_grad():
            for p in sorted(NAMES):
                mask = (parts_frame == p) & is_test_frame
                idx = np.nonzero(mask)[0]
                if len(idx) == 0:
                    continue
                tot, nb = 0.0, 0
                for i in range(0, len(idx), args.batch_size):
                    bidx = torch.from_numpy(idx[i:i + args.batch_size])
                    z = emb[bidx]
                    tgt = pix_t[bidx].to(device).float() / 255.0
                    pred = dec(z)
                    tot += F.mse_loss(pred, tgt).item()
                    nb += 1
                mse = tot / max(1, nb)
                res[NAMES[p]] = {"n": int(len(idx)), "mse": mse,
                                 "psnr": float(-10.0 * np.log10(max(mse, 1e-10)))}
        return res

    best_overall = float("inf")
    for ep in range(1, args.epochs + 1):
        dec.train()
        ep_perm = train_idx_t[torch.randperm(len(train_idx_t), generator=g)]
        run_loss, nb = 0.0, 0
        for i in range(0, len(ep_perm), args.batch_size):
            bidx = ep_perm[i:i + args.batch_size]
            z = emb[bidx]
            tgt = pix_t[bidx].to(device).float() / 255.0
            pred = dec(z)
            loss = F.l1_loss(pred, tgt) + F.mse_loss(pred, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item()
            nb += 1
        sched.step()
        if ep % 5 == 0 or ep == args.epochs:
            res = per_part_psnr()
            line = " | ".join(f"{k} {v['psnr']:.2f}dB" for k, v in res.items())
            print(f"  ep {ep:3d} | train {run_loss/nb:.4f} | {line}", flush=True)
            # track best by mean test MSE across partitions
            mean_mse = np.mean([v["mse"] for v in res.values()])
            if mean_mse < best_overall:
                best_overall = mean_mse
                torch.save({"decoder": dec.state_dict(), "emb_dim": emb_dim,
                            "tag": args.tag, "ckpt": args.ckpt}, out / f"udecoder_{args.tag}.pt")

    # --- final per-partition eval + save grids ---
    print(f"[5/5] final per-partition PSNR (held-out)", flush=True)
    res = per_part_psnr()
    print(f"\n=== UNIVERSAL decoder | {args.domain} | tag={args.tag} | enc={Path(args.ckpt).name} ===")
    print(f"{'partition':<10} {'n':>6} {'PSNR(dB)':>9} {'MSE':>10}")
    for p in sorted(NAMES):
        if NAMES[p] in res:
            v = res[NAMES[p]]
            print(f"{NAMES[p]:<10} {v['n']:>6} {v['psnr']:>9.2f} {v['mse']:>10.5f}")

    # write json FIRST (so results persist even if grid saving hiccups)
    with open(out / f"upsnr_{args.tag}.json", "w") as f:
        json.dump({"domain": args.domain, "tag": args.tag, "ckpt": args.ckpt,
                   "emb_source": args.emb_source, "test_frac": args.test_frac,
                   "results": res}, f, indent=2)
    print(f"[saved] upsnr_{args.tag}.json in {out}")

    # save a per-partition reconstruction grid on test frames
    # save_grid expects torch tensors (calls .clamp); keep them as tensors here.
    dec.eval()
    with torch.no_grad():
        for p in sorted(NAMES):
            mask = (parts_frame == p) & is_test_frame
            idx = np.nonzero(mask)[0]
            if len(idx) < 8:
                continue
            sel = idx[np.linspace(0, len(idx) - 1, 8).astype(int)]
            z = emb[torch.from_numpy(sel)]
            pred = dec(z).clamp(0, 1).cpu()
            gt = (pix_t[torch.from_numpy(sel)].float() / 255.0).cpu()
            save_grid(gt, pred, str(out / f"udec_{args.tag}_{NAMES[p].replace('/', '')}.png"))
    print(f"[saved] grids in {out}")


if __name__ == "__main__":
    main()
