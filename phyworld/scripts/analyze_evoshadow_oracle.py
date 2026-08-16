#!/usr/bin/env python3
"""Audit per-trajectory EvoShadow selection without updating the world model."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

TAGS = ("baseline", "a050", "a065", "a075", "a100")
TYPED_SPECS = (
    ("const_acc", "uniform_s{seed}_const_acc_a075.npz"),
    ("damp095", "uniform_s{seed}_damp095_a075.npz"),
    ("damp099", "uniform_s{seed}_damp099_a075.npz"),
)
CONTROL_SPECS = (
    ("shuf11", "uniform_s{seed}_shuf11_a075.npz"),
    ("shuf23", "uniform_s{seed}_shuf23_a075.npz"),
    ("shuf47", "uniform_s{seed}_shuf47_a075.npz"),
)
ORTHOGONAL_CONTROL_SPECS = (
    ("ortho11", "uniform_s{seed}_ortho11_a075.npz"),
    ("ortho23", "uniform_s{seed}_ortho23_a075.npz"),
    ("ortho47", "uniform_s{seed}_ortho47_a075.npz"),
)
EVOLVED_CONTROL_SPECS = (
    ("evo08", "uniform_s{seed}_evo08_a075.npz"),
    ("evo16", "uniform_s{seed}_evo16_a075.npz"),
    ("evo32", "uniform_s{seed}_evo32_a075.npz"),
)
PARTS = ("ID", "r/m-OOD", "v-OOD", "both-OOD")


def load_seed(root, seed, typed_dir=None, control_dir=None,
              orthogonal_control_dir=None, evolved_control_dir=None):
    tags = list(TAGS)
    paths = [root / f"s{seed}_{tag}.npz" for tag in tags]
    if typed_dir is not None:
        for tag, pattern in TYPED_SPECS:
            tags.append(tag)
            paths.append(typed_dir / pattern.format(seed=seed))
    if control_dir is not None:
        for tag, pattern in CONTROL_SPECS:
            tags.append(tag)
            paths.append(control_dir / pattern.format(seed=seed))
    if orthogonal_control_dir is not None:
        for tag, pattern in ORTHOGONAL_CONTROL_SPECS:
            tags.append(tag)
            paths.append(orthogonal_control_dir / pattern.format(seed=seed))
    if evolved_control_dir is not None:
        for tag, pattern in EVOLVED_CONTROL_SPECS:
            tags.append(tag)
            paths.append(evolved_control_dir / pattern.format(seed=seed))
    bundles = []
    for tag, path in zip(tags, paths):
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
    for tag, bundle in zip(tags[1:], bundles[1:]):
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
        tags=tuple(tags),
    )


def context_features(context, train, pca_dim):
    scaler = StandardScaler().fit(context[train])
    scaled = scaler.transform(context)
    dim = min(pca_dim, int(train.sum()) - 1, scaled.shape[1])
    return PCA(n_components=dim, random_state=0).fit(
        scaled[train]).transform(scaled)


def dual_context_features(context, train, pca_dim):
    latent = context.reshape(len(context), 3, -1)
    gauge = latent.mean(axis=1)
    dynamics = np.concatenate(
        [latent[:, 1] - latent[:, 0], latent[:, 2] - latent[:, 1]], axis=1)
    return (
        context_features(dynamics, train, pca_dim),
        context_features(gauge, train, pca_dim),
    )


def higher_quantile(values, level):
    try:
        return np.quantile(values, level, method="higher")
    except TypeError:
        return np.quantile(values, level, interpolation="higher")


def choose(data, pca_dim, warmup, safety_margin, id_threshold,
           conformal_delta):
    losses, train = data["losses"], data["train"]
    tags = data["tags"]
    x = context_features(data["context"], train, pca_dim)
    dynamics_x, gauge_x = dual_context_features(
        data["context"], train, pca_dim)
    long = data["horizons"] >= 16
    target = np.log1p(losses[:, :, long].mean(2))

    history_model = ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=4, max_features="sqrt",
        n_jobs=-1, random_state=0,
    ).fit(x[train], target[train])
    historical_pred = np.expm1(history_model.predict(x))
    historical = historical_pred.argmin(1)
    fixed = tags.index("a075")
    historical_gated = historical.copy()
    predicted_best = historical_pred[np.arange(len(train)), historical]
    reject = predicted_best >= historical_pred[:, fixed] * (1 - safety_margin)
    historical_gated[reject] = fixed

    # Supervised dual memory: dynamics deltas route experts, while a separate
    # gauge/appearance memory detects the historical ID support and falls back
    # to the unmodified baseline there. Partition labels are used only from
    # completed historical episodes, never from held-out test episodes.
    dynamics_model = ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=4, max_features="sqrt",
        n_jobs=-1, random_state=0,
    ).fit(dynamics_x[train], target[train])
    dynamics_log_pred = dynamics_model.predict(dynamics_x)
    dual_memory = dynamics_log_pred.argmin(1)
    id_model = ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=4, max_features="sqrt",
        class_weight="balanced", n_jobs=-1, random_state=0,
    ).fit(gauge_x[train], data["parts"][train] == 0)
    id_class_index = list(id_model.classes_).index(True)
    id_probability = id_model.predict_proba(gauge_x)[:, id_class_index]
    dual_memory[id_probability >= id_threshold] = tags.index("baseline")

    # Candidate-wise cross-conformal lower confidence bounds. This is
    # intentionally conservative: the candidate is accepted only when its
    # predicted log-loss gain over fixed a075 exceeds a Bonferroni-corrected
    # upper quantile of historical gain overestimation.
    historical_indices = np.flatnonzero(train)
    oof_prediction = np.zeros((len(historical_indices), len(tags)))
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for fold, (fit, calibration) in enumerate(splitter.split(
            dynamics_x[historical_indices],
            data["parts"][historical_indices])):
        fold_model = ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=4, max_features="sqrt",
            n_jobs=-1, random_state=fold,
        ).fit(
            dynamics_x[historical_indices[fit]],
            target[historical_indices[fit]],
        )
        oof_prediction[calibration] = fold_model.predict(
            dynamics_x[historical_indices[calibration]])
    candidate_delta = conformal_delta / (len(tags) - 1)
    level = min(
        1.0,
        np.ceil((len(historical_indices) + 1) * (1 - candidate_delta))
        / len(historical_indices),
    )
    fixed_prediction = oof_prediction[:, fixed]
    fixed_target = target[historical_indices, fixed]
    overestimation_quantile = np.zeros(len(tags))
    for candidate in range(len(tags)):
        predicted_gain = fixed_prediction - oof_prediction[:, candidate]
        actual_gain = fixed_target - target[historical_indices, candidate]
        overestimation_quantile[candidate] = higher_quantile(
            predicted_gain - actual_gain, level)
    overestimation_quantile[fixed] = 0.0
    lower_gain = (
        dynamics_log_pred[:, fixed, None]
        - dynamics_log_pred
        - overestimation_quantile
    )
    conformal_safe = lower_gain.argmax(1)
    safe_gain = lower_gain[np.arange(len(train)), conformal_safe]
    conformal_safe[safe_gain <= 0] = fixed

    # Split-conformal calibration after routing a single candidate. The first
    # 300 historical episodes fit the router; the remaining stratified 100
    # calibrate the marginal gain overestimation of that routed action. Unlike
    # candidate-wise Bonferroni calibration, this pays for one action only.
    route_fit, route_calibration = train_test_split(
        historical_indices,
        test_size=0.25,
        random_state=0,
        stratify=data["parts"][historical_indices],
    )
    routed_model = ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=4, max_features="sqrt",
        n_jobs=-1, random_state=0,
    ).fit(dynamics_x[route_fit], target[route_fit])
    routed_prediction = routed_model.predict(dynamics_x)
    routed_candidate = routed_prediction.argmin(1)
    routed_predicted_gain = (
        routed_prediction[:, fixed]
        - routed_prediction[np.arange(len(train)), routed_candidate]
    )
    routed_actual_gain = (
        target[:, fixed]
        - target[np.arange(len(train)), routed_candidate]
    )
    routed_level = min(
        1.0,
        np.ceil((len(route_calibration) + 1) * (1 - conformal_delta))
        / len(route_calibration),
    )
    routed_quantile = higher_quantile(
        routed_predicted_gain[route_calibration]
        - routed_actual_gain[route_calibration],
        routed_level,
    )
    routed_split_safe = routed_candidate.copy()
    routed_accept = (
        (routed_candidate != fixed)
        & (routed_predicted_gain - routed_quantile > 0)
    )
    routed_split_safe[~routed_accept] = fixed

    test_ids = np.flatnonzero(~train)
    prequential = np.full(len(train), fixed, dtype=np.int64)
    for pos, ep in enumerate(test_ids):
        if pos < warmup:
            continue
        previous = test_ids[:pos]
        model = Ridge(alpha=10.0).fit(x[previous], target[previous])
        prequential[ep] = model.predict(x[ep:ep + 1])[0].argmin()

    return {
        "baseline": np.full(len(train), tags.index("baseline"), dtype=np.int64),
        "fixed_a075": np.full(len(train), fixed, dtype=np.int64),
        "prequential": prequential,
        "historical": historical,
        "historical_gated_posthoc": historical_gated,
        "dual_memory_supervised": dual_memory,
        "candidate_conformal_safe": conformal_safe,
        "routed_split_conformal": routed_split_safe,
        "oracle": target.argmin(1),
    }


def summarize(data, choices):
    losses, train = data["losses"], data["train"]
    tags = data["tags"]
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
                tags,
                map(int, np.bincount(selection[test], minlength=len(tags))))),
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
    parser.add_argument("--typed-dir", type=Path)
    parser.add_argument("--control-dir", type=Path)
    parser.add_argument("--orthogonal-control-dir", type=Path)
    parser.add_argument("--evolved-control-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 3072, 42])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--safety-margin", type=float, default=0.10)
    parser.add_argument("--id-threshold", type=float, default=0.50)
    parser.add_argument("--conformal-delta", type=float, default=0.10)
    args = parser.parse_args()

    seeds = {}
    loaded_tags = None
    for seed in args.seeds:
        data = load_seed(
            args.input_dir, seed, args.typed_dir, args.control_dir,
            args.orthogonal_control_dir, args.evolved_control_dir)
        loaded_tags = data["tags"]
        seeds[str(seed)] = {
            "audit": {
                "aligned": True,
                "episodes": int(len(data["train"])),
                "historical_episodes": int(data["train"].sum()),
                "test_episodes": int((~data["train"]).sum()),
                "horizons": data["horizons"].tolist(),
            },
            "metrics": summarize(
                data, choose(
                    data, args.pca_dim, args.warmup, args.safety_margin,
                    args.id_threshold, args.conformal_delta,
                )
            ),
        }
    payload = {
        "protocol": {
            "expert_tags": list(loaded_tags), "long_horizon": "h>=16",
            "pca_dim": args.pca_dim, "prequential_warmup": args.warmup,
            "historical_selector": "ExtraTreesRegressor",
            "prequential_selector": "Ridge", "base_model_updated": False,
            "posthoc_safety_margin": args.safety_margin,
            "posthoc_safety_margin_used_test_results": True,
            "dual_memory_dynamics": "two latent first differences",
            "dual_memory_gauge": "three-latent mean",
            "dual_memory_uses_historical_partition_labels": True,
            "id_probability_threshold": args.id_threshold,
            "candidate_conformal_delta": args.conformal_delta,
            "routed_split_conformal_delta": args.conformal_delta,
            "routed_split_fit_episodes": 300,
            "routed_split_calibration_episodes": 100,
            "routed_split_guarantee": "marginal, not partition-conditional",
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
