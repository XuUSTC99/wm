#!/usr/bin/env python3
"""审计 collision 有类型 Shadow 专家，不更新世界模型参数。"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PARTS = ("ID", "r/m-OOD", "v-OOD", "both-OOD")
TAGS = (
    "baseline",
    "cv_a075",
    "impulse_d200a010",
    "impulse_d200a050",
    "impulse_d250a005",
    "impulse_d250a010",
    "impulse_d250a015",
    "impulse_d250a020",
    "impulse_d250a025",
    "impulse_d250a050",
    "impulse_d250a075",
    "impulse_d300a010",
    "impulse_d300a050",
)
SMALL_ALPHA_TAGS = (
    "baseline",
    "impulse_d200a010",
    "impulse_d250a005",
    "impulse_d250a010",
    "impulse_d250a015",
    "impulse_d250a020",
    "impulse_d250a025",
    "impulse_d300a010",
)
ORTHOGONAL_TAGS = (
    "ortho11", "ortho23", "ortho47",
    "ortho59", "ortho71", "ortho89",
)


def load_seed(root, seed, orthogonal_dir=None):
    if orthogonal_dir is None:
        tags = TAGS
        paths = [root / f"collision_s{seed}_{tag}.npz" for tag in tags]
    else:
        tags = ("baseline",) + ORTHOGONAL_TAGS
        paths = [root / f"collision_s{seed}_baseline.npz"] + [
            orthogonal_dir / f"collision_s{seed}_{tag}_a075.npz"
            for tag in ORTHOGONAL_TAGS
        ]
    bundles = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        bundles.append(np.load(path, allow_pickle=False))
    ref = bundles[0]
    aligned = (
        "meta", "episode_ids", "episode_parts",
        "episode_context", "episode_in_train",
    )
    for tag, bundle in zip(tags[1:], bundles[1:]):
        for field in aligned:
            if not np.array_equal(bundle[field], ref[field]):
                raise ValueError(
                    f"seed {seed}: {tag} differs on {field}")
    ids = ref["episode_ids"]
    horizons = np.unique(ref["meta"][:, 1])
    n_episode, n_horizon = len(ids), len(horizons)
    if not np.array_equal(
            ref["meta"][:, 0], np.repeat(ids, n_horizon)):
        raise ValueError(f"seed {seed}: episodes not grouped")
    losses = np.stack([
        bundle["frame_sse"].reshape(n_episode, n_horizon)
        for bundle in bundles
    ], axis=1)
    return {
        "losses": losses,
        "context": ref["episode_context"],
        "parts": ref["episode_parts"],
        "train": ref["episode_in_train"].astype(bool),
        "ids": ids,
        "horizons": horizons,
        "tags": tags,
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
    x = context_features(data["context"], train, pca_dim)

    def historical(candidate_indices):
        model = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=4,
            max_features="sqrt",
            n_jobs=-1,
            random_state=0,
        ).fit(x[train], target[train][:, candidate_indices])
        local = model.predict(x).argmin(1)
        return np.asarray(candidate_indices)[local]

    tags = data["tags"]
    all_indices = np.arange(len(tags))
    best_fixed = int(target[train].mean(0).argmin())
    output = {
        "baseline": np.full(len(train), tags.index("baseline")),
        "history_best_fixed": np.full(len(train), best_fixed),
        "historical_all": historical(all_indices),
        "routed_split_conformal": routed_split_conformal(
            x, target, train, data["parts"],
            fallback=tags.index("baseline")),
    }
    if all(tag in tags for tag in SMALL_ALPHA_TAGS):
        small_indices = np.array([tags.index(tag) for tag in SMALL_ALPHA_TAGS])
        output["historical_small_alpha"] = historical(small_indices)
    output["oracle_all"] = target.argmin(1)
    if all(tag in tags for tag in SMALL_ALPHA_TAGS):
        output["oracle_small_alpha"] = small_indices[
            target[:, small_indices].argmin(1)]
    return output


def collision_phases(eval_h5, episode_ids, horizons):
    with h5py.File(eval_h5, "r") as handle:
        raw_episode = handle["episode_idx"][:]
        raw_step = handle["step_idx"][:]
        raw_event = handle["collision_event"][:]
    first_event = {}
    for episode in episode_ids:
        mask = (raw_episode == episode) & (raw_event > 0)
        steps = raw_step[mask]
        first_event[int(episode)] = (
            int(steps.min()) if len(steps) else None)
    phases = np.empty((len(episode_ids), len(horizons)), dtype=np.int8)
    for row, episode in enumerate(episode_ids):
        event_step = first_event[int(episode)]
        for col, horizon in enumerate(horizons):
            target_step = int(horizon) + 2
            if event_step is None:
                phases[row, col] = 3
            elif target_step < event_step:
                phases[row, col] = 0
            elif target_step == event_step:
                phases[row, col] = 1
            else:
                phases[row, col] = 2
    return phases


def summarize(data, choices, phases):
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
                for index, tag in enumerate(data["tags"])
            },
            "by_partition": {},
            "by_phase": {},
        }
        for part_id, part_name in enumerate(PARTS):
            mask = test & (parts == part_id)
            row["by_partition"][part_name] = {
                "n": int(mask.sum()),
                "long": float(frames[mask][:, long].mean()),
                "all": float(frames[mask].mean()),
            }
        for phase_id, phase_name in enumerate(
                ("pre", "impact", "post", "no_event")):
            mask = test[:, None] & (phases == phase_id)
            row["by_phase"][phase_name] = {
                "n_frames": int(mask.sum()),
                "sse": float(frames[mask].mean()) if mask.any() else None,
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
            output[method][key + "_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0)
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
    parser.add_argument("--orthogonal-dir", type=Path)
    parser.add_argument("--eval-h5", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 3072])
    parser.add_argument("--pca-dim", type=int, default=64)
    args = parser.parse_args()

    seed_results = {}
    loaded_tags = None
    for seed in args.seeds:
        data = load_seed(args.input_dir, seed, args.orthogonal_dir)
        loaded_tags = data["tags"]
        phases = collision_phases(
            args.eval_h5, data["ids"], data["horizons"])
        choice = selections(data, args.pca_dim)
        seed_results[str(seed)] = {
            "metrics": summarize(data, choice, phases),
        }
    result = {
        "protocol": {
            "expert_tags": loaded_tags,
            "small_alpha_tags": (
                SMALL_ALPHA_TAGS if args.orthogonal_dir is None else []),
            "long_horizon": "h>=16",
            "historical_feedback_episodes": 400,
            "routed_split_fit_episodes": 300,
            "routed_split_calibration_episodes": 100,
            "routed_split_conformal_delta": 0.1,
            "routed_split_guarantee": "marginal, not partition-conditional",
            "held_out_test_episodes": 100,
            "base_model_updated": False,
            "selector": "ExtraTreesRegressor",
            "collision_phase_target_step": "horizon + history_size - 1",
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
