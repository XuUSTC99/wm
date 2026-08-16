#!/usr/bin/env python3
"""生成保持统计结构但破坏物理语义的 GIPP 解码器对照。"""

import argparse
from pathlib import Path

import numpy as np


def crossed_permutation(state_dim, seed):
    if state_dim % 2:
        raise ValueError("state dimension must have equal position/velocity blocks")
    position_dim = state_dim // 2
    random = np.random.default_rng(seed)
    for _ in range(1000):
        permutation = random.permutation(state_dim)
        position_sources = permutation[:position_dim]
        velocity_sources = permutation[position_dim:]
        crosses_blocks = (
            np.any(position_sources >= position_dim)
            and np.any(velocity_sources < position_dim)
        )
        if crosses_blocks:
            return permutation
    raise RuntimeError("failed to sample a cross-block permutation")


def make_control(input_path, output_path, seed):
    with np.load(input_path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    # Gravity may be present as decoder metadata; constant-velocity
    # controls do not consume it, so it is preserved without interpretation.
    state_dim = payload["weight"].shape[0]
    permutation = crossed_permutation(state_dim, seed)
    for key in ("weight", "bias", "state_scale", "history_weight"):
        if key in payload:
            payload[key] = payload[key][permutation]
    payload["control_type"] = np.asarray("cross_block_state_permutation")
    payload["control_seed"] = np.int64(seed)
    payload["control_permutation"] = permutation.astype(np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return permutation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    permutation = make_control(args.input, args.output, args.seed)
    print(
        f"saved {args.output} | seed={args.seed} "
        f"permutation={permutation.tolist()}"
    )


if __name__ == "__main__":
    main()
