#!/usr/bin/env python3
"""审计 parabola 的 K=3 时序重力 Shadow 专家。"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PARTS = ("ID", "r/m-OOD", "v-OOD", "both-OOD")
TAGS = (
    "baseline",
    "k3_gravity_a0005",
    "k3_gravity_a001",
    "k3_gravity_a0025",
    "k3_gravity_a005",
    "k3_gravity_a010",
    "k3_gravity_a025",
)


def load_seed(root, seed):
    bundles = []
    for tag in TAGS:
        path = root / f"parabola_s{seed}_{tag}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        bundles.append(np.load(path, allow_pickle=False))
    ref = bundles[0]
    aligned = (
        "meta", "episode_ids", "episode_parts",
        "episode_context", "episode_in_train",
    )
    for tag, bundle in zip(TAGS[1:], bundles[1:]):
        for field in aligned:
            if not np.array_equal(bundle[field], ref[field]):
                raise ValueError(
                    f"seed {seed}: {tag} differs on {field}")
    ids = ref["episode_ids"]
    horizons = np.unique(ref["meta"][:, 1])
    n_episode, n_horizon = len(ids), len(horizons)
    losses = np.stack([
        bundle["frame_sse"].reshape(n_episode, n_horizon)
        for bundle in bundles
    ], axis=1)
    return {
        "losses": losses,
        "context": ref["episode_context"],
        "parts": ref["episode_parts"],
        "train": ref["episode_in_train"].astype(bool),
        "horizons": horizons,
    }


def context_features(context, train, pca_dim):
    scaler = StandardScaler().fit(context[train])
    scaled = scaler.transform(context)
    dim = min(pca_dim, int(train.sum()) - 1, scaled.shape[1])
    return PCA(n_components=dim, random_state=0).fit(
        scaled[train]).transform(scaled)

def routed_split_conformal(
        features, target, train, parts, fallback, delta=0.1):
    historical = np.flatnonzero(train)
    fit, calibration = train_test_split(
        historical,
        test_size=0.25,
        random_state=0,
        stratify=parts[historical],
    )
    model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=4,
        max_features="sqrt",
        n_jobs=-1,
        random_state=0,
    ).fit(features[fit], target[fit])
    prediction = model.predict(features)
    candidate = prediction.argmin(1)
    predicted_gain = (
        prediction[:, fallback]
        - prediction[np.arange(len(train)), candidate]
    )
    actual_gain = (
        target[:, fallback]
        - target[np.arange(len(train)), candidate]
    )
    level = min(
        1.0,
        np.ceil((len(calibration) + 1) * (1 - delta))
        / len(calibration),
    )
    overestimation = (
        predicted_gain[calibration] - actual_gain[calibration])
    try:
        quantile = np.quantile(overestimation, level, method="higher")
    except TypeError:
        quantile = np.quantile(
            overestimation, level, interpolation="higher")
    accept = (
        (candidate != fallback)
        & (predicted_gain - quantile > 0)
    )
    return np.where(accept, candidate, fallback)


def selections(data, pca_dim):
    losses, train = data["losses"], data["train"]
    long = data["horizons"] >= 16
    target = np.log1p(losses[:, :, long].mean(2))
    features = context_features(data["context"], train, pca_dim)
    model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=4,
        max_features="sqrt",
        n_jobs=-1,
        random_state=0,
    ).fit(features[train], target[train])
    output = {
        "baseline": np.zeros(len(train), dtype=np.int64),
        "historical": model.predict(features).argmin(1),
        "routed_split_conformal": routed_split_conformal(
            features, target, train, data["parts"], fallback=0),
        "oracle": target.argmin(1),
    }
    for index, tag in enumerate(TAGS[1:], start=1):
        output["fixed_" + tag.removeprefix("k3_gravity_")] = np.full(
            len(train), index, dtype=np.int64)
    return output


def summarize(data, choices):
    losses = data["losses"]
    train, parts = data["train"], data["parts"]
    test, horizons = ~train, data["horizons"]
    long = horizons >= 16
    result = {}
    for name, choice in choices.items():
        frames = losses[np.arange(len(losses)), choice]
        row = {
            "long": float(frames[test][:, long].mean()),
            "h16": float(frames[test][:, horizons == 16].mean()),
            "h28": float(frames[test][:, horizons == 28].mean()),
            "all": float(frames[test].mean()),
            "choice_counts": {
                tag: int((choice[test] == index).sum())
                for index, tag in enumerate(TAGS)
            },
            "by_partition": {},
        }
        for part_id, part_name in enumerate(PARTS):
            mask = test & (parts == part_id)
            row["by_partition"][part_name] = {
                "n": int(mask.sum()),
                "long": float(frames[mask][:, long].mean()),
                "all": float(frames[mask].mean()),
            }
        result[name] = row
    baseline = result["baseline"]["long"]
    for row in result.values():
        row["long_gain_vs_baseline_pct"] = float(
            100 * (baseline - row["long"]) / baseline)
    return result


def aggregate(seed_results):
    methods = next(iter(seed_results.values()))["metrics"]
    output = {}
    for method in methods:
        rows = [seed["metrics"][method] for seed in seed_results.values()]
        output[method] = {}
        for key in ("long", "h16", "h28", "all",
                    "long_gain_vs_baseline_pct"):
            values = np.array([row[key] for row in rows])
            output[method][key + "_mean"] = float(values.mean())
            output[method][key + "_std"] = float(values.std(ddof=1))
        output[method]["by_partition"] = {}
        for part in PARTS:
            values = np.array([
                row["by_partition"][part]["long"] for row in rows])
            baselines = np.array([
                seed["metrics"]["baseline"]["by_partition"][part]["long"]
                for seed in seed_results.values()])
            output[method]["by_partition"][part] = {
                "long_mean": float(values.mean()),
                "gain_vs_baseline_pct_mean": float(
                    (100 * (baselines - values) / baselines).mean()),
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1234, 3072, 42])
    parser.add_argument("--pca-dim", type=int, default=64)
    args = parser.parse_args()

    seed_results = {}
    for seed in args.seeds:
        data = load_seed(args.input_dir, seed)
        choice = selections(data, args.pca_dim)
        seed_results[str(seed)] = {
            "metrics": summarize(data, choice),
        }
    result = {
        "protocol": {
            "expert_tags": TAGS,
            "temporal_history_size": 3,
            "long_horizon": "h>=16",
            "historical_feedback_episodes": 400,
            "routed_split_fit_episodes": 300,
            "routed_split_calibration_episodes": 100,
            "routed_split_conformal_delta": 0.1,
            "routed_split_guarantee": "marginal, not partition-conditional",
            "held_out_test_episodes": 100,
            "base_model_updated": False,
            "selector": "ExtraTreesRegressor",
            "fixed_a010_selected_after_alpha_sweep": True,
        },
        "seeds": seed_results,
        "aggregate": aggregate(seed_results),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {args.output_json}")


if __name__ == "__main__":
    main()
