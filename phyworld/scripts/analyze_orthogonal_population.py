#!/usr/bin/env python3
"""审计基于历史反馈的正交运输种群精英选择。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_evoshadow_oracle import (  # noqa: E402
    TAGS,
    aggregate,
    context_features,
    higher_quantile,
    load_seed,
    summarize,
)


INITIAL_ORTHOGONAL = (11, 23, 47)
ADDITIONAL_ORTHOGONAL = (59, 71, 89, 101, 113, 127, 139, 151, 163)
ALIGNED_FIELDS = (
    "meta", "episode_ids", "episode_parts",
    "episode_context", "episode_in_train",
)


def append_population(data, input_dir, root, seed):
    reference_path = root / f"uniform_s{seed}_ortho59_a075.npz"
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    base_reference = np.load(
        input_dir / f"s{seed}_baseline.npz",
        allow_pickle=False,
    )
    episode_count = len(data["train"])
    horizon_count = len(data["horizons"])
    extra_losses = []
    extra_tags = []
    for control in ADDITIONAL_ORTHOGONAL:
        tag = f"ortho{control}"
        path = root / f"uniform_s{seed}_{tag}_a075.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        bundle = np.load(path, allow_pickle=False)
        for field in ALIGNED_FIELDS:
            if not np.array_equal(bundle[field], base_reference[field]):
                raise ValueError(f"seed {seed}: {tag} differs on {field}")
        extra_losses.append(
            bundle["frame_sse"].reshape(episode_count, horizon_count))
        extra_tags.append(tag)
    data["losses"] = np.concatenate(
        [data["losses"], np.stack(extra_losses, axis=1)], axis=1)
    data["tags"] = data["tags"] + tuple(extra_tags)
    return data


def load_population(input_dir, initial_dir, population_dir, seed):
    data = load_seed(
        input_dir,
        seed,
        orthogonal_control_dir=initial_dir,
    )
    return append_population(data, input_dir, population_dir, seed)


def split_history(data):
    historical = np.flatnonzero(data["train"])
    fit, calibration = train_test_split(
        historical,
        test_size=0.25,
        random_state=0,
        stratify=data["parts"][historical],
    )
    return fit, calibration


def select_elites(data, fit, elite_count):
    long = data["horizons"] >= 16
    target = np.log1p(data["losses"][:, :, long].mean(2))
    orthogonal_indices = np.array([
        index for index, tag in enumerate(data["tags"])
        if tag.startswith("ortho")
    ])
    score = target[fit][:, orthogonal_indices].mean(0)
    elite_local = np.argsort(score)[:elite_count]
    elites = orthogonal_indices[elite_local]
    return elites, target, score


def route_with_split_conformal(
        features, target, fit, calibration, candidates, fallback, delta):
    model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=4,
        max_features="sqrt",
        n_jobs=-1,
        random_state=0,
    ).fit(features[fit], target[fit][:, candidates])
    prediction = model.predict(features)
    local_candidate = prediction.argmin(1)
    candidate = candidates[local_candidate]
    fallback_local = int(np.flatnonzero(candidates == fallback)[0])
    predicted_gain = (
        prediction[:, fallback_local]
        - prediction[np.arange(len(target)), local_candidate]
    )
    actual_gain = (
        target[:, fallback]
        - target[np.arange(len(target)), candidate]
    )
    level = min(
        1.0,
        np.ceil((len(calibration) + 1) * (1 - delta))
        / len(calibration),
    )
    quantile = higher_quantile(
        predicted_gain[calibration] - actual_gain[calibration], level)
    accept = (
        (candidate != fallback)
        & (predicted_gain - quantile > 0)
    )
    return np.where(accept, candidate, fallback), float(quantile)


def choices(data, pca_dim, elite_count, delta):
    fit, calibration = split_history(data)
    elites, target, population_score = select_elites(
        data, fit, elite_count)
    fit_mask = np.zeros(len(data["train"]), dtype=bool)
    fit_mask[fit] = True
    features = context_features(data["context"], fit_mask, pca_dim)
    base_indices = np.arange(len(TAGS))
    candidates = np.concatenate([base_indices, elites])
    fallback = data["tags"].index("a075")

    historical_model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=4,
        max_features="sqrt",
        n_jobs=-1,
        random_state=0,
    ).fit(
        features[data["train"]],
        target[data["train"]][:, candidates],
    )
    historical = candidates[historical_model.predict(features).argmin(1)]
    routed, quantile = route_with_split_conformal(
        features, target, fit, calibration, candidates, fallback, delta)

    best_fixed = int(elites[target[fit][:, elites].mean(0).argmin()])
    reduced_oracle = candidates[target[:, candidates].argmin(1)]
    all_indices = np.arange(len(data["tags"]))
    return {
        "baseline": np.full(len(target), data["tags"].index("baseline")),
        "fixed_a075": np.full(len(target), fallback),
        "elite_best_fixed": np.full(len(target), best_fixed),
        "elite_historical": historical,
        "elite_routed_split_conformal": routed,
        "elite_oracle": reduced_oracle,
        "population_oracle_all12": target.argmin(1),
    }, {
        "fit_indices": fit.tolist(),
        "calibration_indices": calibration.tolist(),
        "elite_tags": [data["tags"][index] for index in elites],
        "elite_fit_log_losses": [float(
            target[fit][:, index].mean()) for index in elites],
        "population_tags": [data["tags"][index] for index in all_indices
                            if data["tags"][index].startswith("ortho")],
        "population_fit_log_losses": population_score.tolist(),
        "routed_quantile": quantile,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--initial-dir", type=Path, required=True)
    parser.add_argument("--population-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1234, 3072, 42])
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--elite-count", type=int, default=3)
    parser.add_argument("--conformal-delta", type=float, default=0.1)
    args = parser.parse_args()

    seed_results = {}
    for seed in args.seeds:
        data = load_population(
            args.input_dir, args.initial_dir, args.population_dir, seed)
        selection, audit = choices(
            data, args.pca_dim, args.elite_count, args.conformal_delta)
        seed_results[str(seed)] = {
            "audit": audit,
            "metrics": summarize(data, selection),
        }
    result = {
        "protocol": {
            "population_size": 12,
            "elite_count": args.elite_count,
            "elite_selection_episodes": 300,
            "calibration_episodes": 100,
            "test_episodes": 100,
            "elite_selection_uses_calibration": False,
            "elite_selection_uses_test": False,
            "routed_split_conformal_delta": args.conformal_delta,
            "base_model_updated": False,
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
