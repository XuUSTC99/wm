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


def crossed_orthogonal(state_dim, seed, minimum_cross_energy=0.25):
    if state_dim % 2:
        raise ValueError("state dimension must have equal position/velocity blocks")
    position_dim = state_dim // 2
    random = np.random.default_rng(seed)
    for _ in range(1000):
        orthogonal, triangular = np.linalg.qr(
            random.standard_normal((state_dim, state_dim)))
        signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
        orthogonal = orthogonal * signs[None, :]
        cross_energy = (
            np.square(orthogonal[:position_dim, position_dim:]).sum()
            + np.square(orthogonal[position_dim:, :position_dim]).sum()
        )
        if cross_energy >= minimum_cross_energy:
            return orthogonal
    raise RuntimeError("failed to sample a cross-block orthogonal transform")


def nearby_orthogonal(parent, seed, step):
    if step <= 0:
        raise ValueError("offspring step must be positive")
    parent = np.asarray(parent, dtype=np.float64)
    if parent.ndim != 2 or parent.shape[0] != parent.shape[1]:
        raise ValueError("parent transform must be square")
    random = np.random.default_rng(seed)
    noise = random.standard_normal(parent.shape)
    skew = noise - noise.T
    skew /= np.linalg.norm(skew, ord="fro")
    rotation, triangular = np.linalg.qr(np.eye(parent.shape[0]) + step * skew)
    signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    rotation = rotation * signs[None, :]
    return rotation @ parent


def standardized_transform(state_scale, orthogonal):
    scale = np.asarray(state_scale, dtype=np.float64)
    if np.any(scale <= 0):
        raise ValueError("state_scale must be positive")
    return scale[:, None] * orthogonal / scale[None, :]


def make_control(input_path, output_path, seed, mode="permutation",
                 parent_path=None, step=0.2):
    with np.load(input_path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    # Gravity may be present as decoder metadata; constant-velocity
    # controls do not consume it, so it is preserved without interpretation.
    state_dim = payload["weight"].shape[0]
    if mode == "permutation":
        descriptor = crossed_permutation(state_dim, seed)
        for key in ("weight", "bias", "state_scale", "history_weight"):
            if key in payload:
                payload[key] = payload[key][descriptor]
        payload["control_type"] = np.asarray("cross_block_state_permutation")
        payload["control_permutation"] = descriptor.astype(np.int64)
    elif mode == "orthogonal":
        orthogonal = crossed_orthogonal(state_dim, seed)
        scale = payload.get("state_scale", np.ones(state_dim))
        transform = standardized_transform(scale, orthogonal)
        for key in ("weight", "bias", "history_weight"):
            if key in payload:
                payload[key] = transform @ payload[key]
        descriptor = orthogonal
        payload["control_type"] = np.asarray(
            "standardized_cross_block_orthogonal")
        payload["control_standardized_transform"] = orthogonal
    elif mode == "orthogonal_offspring":
        if parent_path is None:
            raise ValueError("orthogonal_offspring requires parent_path")
        with np.load(parent_path, allow_pickle=False) as parent:
            if "control_standardized_transform" not in parent:
                raise ValueError("parent has no standardized transform")
            parent_transform = parent["control_standardized_transform"]
        orthogonal = nearby_orthogonal(parent_transform, seed, step)
        scale = payload.get("state_scale", np.ones(state_dim))
        transform = standardized_transform(scale, orthogonal)
        for key in ("weight", "bias", "history_weight"):
            if key in payload:
                payload[key] = transform @ payload[key]
        descriptor = orthogonal
        payload["control_type"] = np.asarray(
            "historical_selected_orthogonal_offspring")
        payload["control_standardized_transform"] = orthogonal
        payload["control_parent"] = np.asarray(str(parent_path))
        payload["control_step"] = np.float64(step)
    else:
        raise ValueError(f"unknown control mode: {mode}")
    payload["control_seed"] = np.int64(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return descriptor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("permutation", "orthogonal", "orthogonal_offspring"),
        default="permutation")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--step", type=float, default=0.2)
    args = parser.parse_args()
    descriptor = make_control(
        args.input, args.output, args.seed, args.mode, args.parent, args.step)
    print(
        f"saved {args.output} | seed={args.seed} mode={args.mode} "
        f"descriptor={descriptor.tolist()}"
    )


if __name__ == "__main__":
    main()
