import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("make_shuffled_gipp_state.py")
SPEC = importlib.util.spec_from_file_location("state_controls", SCRIPT)
CONTROLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROLS)


def make_source(path):
    random = np.random.default_rng(7)
    weight = random.standard_normal((4, 6)).astype(np.float32)
    covariance_root = random.standard_normal((6, 6)).astype(np.float32)
    covariance = covariance_root @ covariance_root.T + np.eye(6, dtype=np.float32)
    np.savez_compressed(
        path,
        weight=weight,
        bias=random.standard_normal(4).astype(np.float32),
        covariance=covariance,
        latent_mean=np.zeros(6, dtype=np.float32),
        state_scale=np.asarray([2.0, 3.0, 0.2, 0.4], dtype=np.float32),
        n_samples=np.int64(100),
        ridge_alpha=np.float64(1.0),
        gravity=np.zeros(2, dtype=np.float32),
    )


def standardized_singular_values(bundle):
    return np.linalg.svd(
        bundle["weight"] / bundle["state_scale"][:, None],
        compute_uv=False,
    )


def test_permutation_control_preserves_matched_statistics(tmp_path):
    source_path = tmp_path / "source.npz"
    output_path = tmp_path / "permutation.npz"
    make_source(source_path)
    permutation = CONTROLS.make_control(
        source_path, output_path, seed=11, mode="permutation")
    source = np.load(source_path)
    output = np.load(output_path)

    assert np.array_equal(output["weight"], source["weight"][permutation])
    assert np.array_equal(output["bias"], source["bias"][permutation])
    assert np.array_equal(output["covariance"], source["covariance"])
    assert np.any(permutation[:2] >= 2) and np.any(permutation[2:] < 2)


def test_orthogonal_control_preserves_standardized_spectrum(tmp_path):
    source_path = tmp_path / "source.npz"
    output_path = tmp_path / "orthogonal.npz"
    make_source(source_path)
    transform = CONTROLS.make_control(
        source_path, output_path, seed=23, mode="orthogonal")
    source = np.load(source_path)
    output = np.load(output_path)

    assert np.allclose(transform @ transform.T, np.eye(4), atol=1e-12)
    assert np.allclose(
        standardized_singular_values(output),
        standardized_singular_values(source),
    )
    assert np.array_equal(output["covariance"], source["covariance"])
    assert np.linalg.matrix_rank(output["weight"]) == 4


def test_offspring_is_nearby_orthogonal_and_reproducible(tmp_path):
    source_path = tmp_path / "source.npz"
    parent_path = tmp_path / "parent.npz"
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    make_source(source_path)
    CONTROLS.make_control(source_path, parent_path, seed=47, mode="orthogonal")
    for path in (first_path, second_path):
        CONTROLS.make_control(
            source_path, path, seed=101, mode="orthogonal_offspring",
            parent_path=parent_path, step=0.16)

    parent = np.load(parent_path)["control_standardized_transform"]
    first = np.load(first_path)["control_standardized_transform"]
    second = np.load(second_path)["control_standardized_transform"]
    assert np.array_equal(first, second)
    assert np.allclose(first @ first.T, np.eye(4), atol=1e-12)
    assert 0.1 < np.linalg.norm(first - parent, ord="fro") < 0.25
