import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("analyze_orthogonal_population.py")
SPEC = importlib.util.spec_from_file_location("population", SCRIPT)
POPULATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POPULATION)


def synthetic_population():
    episode_count = 40
    tags = (
        "baseline", "a050", "a065", "a075", "a100",
    ) + tuple(f"ortho{index}" for index in range(12))
    losses = np.full((episode_count, len(tags), 2), 100.0)
    for index in range(12):
        losses[:32, 5 + index] = float(index + 1)
    # A tempting candidate that is excellent only on held-out test episodes
    # must not influence historical elite selection.
    losses[32:, 16] = 0.0
    train = np.zeros(episode_count, dtype=bool)
    train[:32] = True
    return {
        "losses": losses,
        "train": train,
        "parts": np.tile(np.arange(4), 10),
        "horizons": np.asarray([16, 28]),
        "tags": tags,
    }


def test_elite_selection_uses_fit_only_and_keeps_calibration_disjoint():
    data = synthetic_population()
    fit, calibration = POPULATION.split_history(data)
    elites, _, _ = POPULATION.select_elites(data, fit, elite_count=3)

    assert len(fit) == 24
    assert len(calibration) == 8
    assert not np.intersect1d(fit, calibration).size
    assert set(fit) | set(calibration) == set(range(32))
    assert elites.tolist() == [5, 6, 7]


def test_test_only_winner_is_not_selected():
    data = synthetic_population()
    fit, _ = POPULATION.split_history(data)
    elites, _, _ = POPULATION.select_elites(data, fit, elite_count=3)

    assert 16 not in elites
