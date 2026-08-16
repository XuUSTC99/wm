#!/usr/bin/env python3
"""Audit per-trajectory EvoShadow selection without updating the world model."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

TAGS = ("baseline", "a050", "a065", "a075", "a100")
PARTS = ("ID", "r/m-OOD", "v-OOD", "both-OOD")


def load_seed(root, seed):
    bundles = []
    for tag in TAGS:
        path = root / f"s{seed}_{tag}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        bundles.append(np.load(path, allow_pickle=False))
    ref = bundles[0]
    required = {"frame_sse", "frame_cos", "meta", "episode_ids",
                "episode_parts", "episode_context", "episode_in_train", "checkpoint"}
    if set(ref.files) != required:
        raise ValueError(f"seed {seed}: unexpected fields {set(ref.files)}")
    aligned = ("meta", "episode_ids", "episode_parts",
               "episode_context", "episode_in_train")
    for tag, bundle in zip(TAGS[1:], bundles[1:]):
        for field in aligned:
            if not np.array_equal(bundle[field], ref[field]):
                raise ValueError(f"seed {seed}: {tag} differs on {field}")
    ids = ref["episode_ids"]
    horizons = np.unique(ref["meta"][:, 1])
    n_ep, n_h = len(ids), len(horizons)
    if not np.array_equal(horizons, np.arange(1, n_h + 1)):
        raise ValueError(f"seed {seed}: non-contiguous horizons")
    if not np.array_equal(ref["meta"][:, 0], np.repeat(ids, n_h)):
        raise ValueError(f"seed {seed}: episodes not grouped")
    if not np.array_equal(ref["meta"][:, 1], np.tile(horizons, n_ep)):
        raise ValueError(f"seed {seed}: horizons not ordered")
    losses = np.stack(
        [z["frame_sse"].reshape(n_ep, n_h) for z in bundles], axis=1)
    return dict(
        losses=losses,
        context=ref["episode_context"],
        parts=ref["episode_parts"],
        train=ref["episode_in_train"].astype(bool),
        horizons=horizons,
    )


def context_features(context, train, pca_dim):
    scaler = StandardScaler().fit(context[train])
    scaled = scaler.transform(context)
    dim = min(pca_dim, int(train.sum()) - 1, scaled.shape[1])
    return PCA(n_components=dim, random_state=0).fit(
        scaled[train]).transform(scaled)


def choose(data, pca_dim, warmup, safety_margin):
    losses, train = data["losses"], data["train"]
    x = context_features(data["context"], train, pca_dim)
    long = data["horizons"] >= 16
    target = np.log1p(losses[:, :, long].mean(2))

    history_model = ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=4, max_features="sqrt",
        n_jobs=-1, random_state=0,
    ).fit(x[train], target[train])
    historical_pred = np.expm1(history_model.predict(x))
    historical = historical_pred.argmin(1)
    fixed = TAGS.index("a075")
    historical_gated = historical.copy()
    predicted_best = historical_pred[np.arange(len(train)), historical]
    reject = predicted_best >= historical_pred[:, fixed] * (1 - safety_margin)
    historical_gated[reject] = fixed

    test_ids = np.flatnonzero(~train)
    prequential = np.full(len(train), TAGS.index("a075"), dtype=np.int64)
    for pos, ep in enumerate(test_ids):
        if pos < warmup:
            continue
        previous = test_ids[:pos]
        model = Ridge(alpha=10.0).fit(x[previous], target[previous])
        prequential[ep] = model.predict(x[ep:ep + 1])[0].argmin()

    return {
        "baseline": np.zeros(len(train), dtype=np.int64),
        "fixed_a075": np.full(len(train), TAGS.index("a075"), dtype=np.int64),
        "prequential": prequential,
        "historical": historical,
        "historical_gated_posthoc": historical_gated,
        "oracle": target.argmin(1),
    }


def summarize(data, choices):
    losses, train = data["losses"], data["train"]
    test, horizons, parts = ~train, data["horizons"], data["parts"]
    long = horizons >= 16
    result = {}
    for name, selection in choices.items():
        frames = losses[np.arange(len(losses)), selection]
        row = {
            "long": float(frames[test][:, long].mean()),
            "h16": float(frames[test][:, horizons == 16].mean()),
            "h28": float(frames[test][:, horizons == 28].mean()),
            "all": float(frames[test].mean()),
            "choice_counts": dict(zip(
                TAGS,
                map(int, np.bincount(selection[test], minlength=len(TAGS))))),
            "by_partition": {},
        }
        for part_id, part_name in enumerate(PARTS):
            mask = test & (parts == part_id)
            row["by_partition"][part_name] = {
                "n": int(mask.sum()),
                "all": float(frames[mask].mean()),
                "long": float(frames[mask][:, long].mean()),
            }
        result[name] = row
    fixed, baseline = result["fixed_a075"]["long"], result["baseline"]["long"]
    for row in result.values():
        row["long_gain_vs_fixed_pct"] = 100 * (fixed - row["long"]) / fixed
        row["long_gain_vs_baseline_pct"] = 100 * (baseline - row["long"]) / baseline
    return result


def aggregate(seed_results):
    output = {}
    methods = next(iter(seed_results.values()))["metrics"]
    keys = ("long", "h16", "h28", "all",
            "long_gain_vs_fixed_pct", "long_gain_vs_baseline_pct")
    for method in methods:
        output[method] = {}
        for key in keys:
            values = np.array([
                result["metrics"][method][key] for result in seed_results.values()])
            output[method][key + "_mean"] = float(values.mean())
            output[method][key + "_std"] = float(values.std(ddof=1))
        output[method]["by_partition"] = {}
        for part in PARTS:
            part_rows = [
                result["metrics"][method]["by_partition"][part]
                for result in seed_results.values()
            ]
            method_long = np.array([row["long"] for row in part_rows])
            fixed_long = np.array([
                result["metrics"]["fixed_a075"]["by_partition"][part]["long"]
                for result in seed_results.values()
            ])
            output[method]["by_partition"][part] = {
                "long_mean": float(method_long.mean()),
                "long_std": float(method_long.std(ddof=1)),
                "gain_vs_fixed_pct_mean": float(
                    (100 * (fixed_long - method_long) / fixed_long).mean()
                ),
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 3072, 42])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--safety-margin", type=float, default=0.10)
    args = parser.parse_args()

    seeds = {}
    for seed in args.seeds:
        data = load_seed(args.input_dir, seed)
        seeds[str(seed)] = {
            "audit": {
                "aligned": True,
                "episodes": int(len(data["train"])),
                "historical_episodes": int(data["train"].sum()),
                "test_episodes": int((~data["train"]).sum()),
                "horizons": data["horizons"].tolist(),
            },
            "metrics": summarize(
                data, choose(data, args.pca_dim, args.warmup, args.safety_margin)
            ),
        }
    payload = {
        "protocol": {
            "expert_tags": list(TAGS), "long_horizon": "h>=16",
            "pca_dim": args.pca_dim, "prequential_warmup": args.warmup,
            "historical_selector": "ExtraTreesRegressor",
            "prequential_selector": "Ridge", "base_model_updated": False,
            "posthoc_safety_margin": args.safety_margin,
            "posthoc_safety_margin_used_test_results": True,
        },
        "seeds": seeds,
        "aggregate": aggregate(seeds),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
