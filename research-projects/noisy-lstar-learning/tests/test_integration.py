"""End-to-end integration test: run a small version of the real experiment
grid and check that the qualitative research finding holds — i.e. that this
is a real effect reproducible from a fresh run, not an artifact of one
hand-picked seed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import run_grid
from src.analysis import success_rate_table, union_bound_success_prediction


def test_noiseless_learning_always_succeeds():
    results = run_grid(noise_rates=[0.0], sizes=(5, 8), num_seeds=4, seed_offset=1000)
    table = success_rate_table(results)
    for strategy in ("none", "fixed5", "adaptive"):
        assert table[(0.0, strategy)] == 1.0, f"noiseless learning should always succeed ({strategy})"


def test_high_noise_without_redundancy_degrades_learning():
    results = run_grid(noise_rates=[0.35], strategies=("none",), sizes=(5, 8, 10),
                        num_seeds=8, seed_offset=2000)
    table = success_rate_table(results)
    # Without any error-correction, a 35%-flip oracle should make exact learning
    # unreliable -- well below the near-certain success adaptive redundancy gets.
    assert table[(0.35, "none")] < 0.8


def test_adaptive_redundancy_recovers_success_rate_under_moderate_to_high_noise():
    results = run_grid(noise_rates=[0.1, 0.2, 0.3], strategies=("adaptive",),
                        sizes=(5, 8, 10), num_seeds=6, seed_offset=3000)
    table = success_rate_table(results)
    for noise_rate in (0.1, 0.2, 0.3):
        rate = table[(noise_rate, "adaptive")]
        assert rate >= 0.9, f"adaptive redundancy should recover success at p={noise_rate}, got {rate}"


def test_adaptive_success_rate_respects_union_bound_prediction():
    results = run_grid(noise_rates=[0.2, 0.3], strategies=("adaptive",),
                        sizes=(5, 8, 10, 13), num_seeds=6, seed_offset=4000)
    success = success_rate_table(results)
    predicted_lower_bound = union_bound_success_prediction(results, "adaptive", delta_q=1e-4)
    for noise_rate in (0.2, 0.3):
        observed = success[(noise_rate, "adaptive")]
        bound = predicted_lower_bound[(noise_rate,)]
        # The union bound is conservative: observed success should not fall
        # meaningfully below it (allow float slack since both are finite-sample estimates).
        assert observed >= bound - 0.05, (
            f"observed success {observed} fell below the union-bound prediction {bound} "
            f"at noise_rate={noise_rate}"
        )


def test_adaptive_uses_more_raw_queries_than_no_redundancy():
    results = run_grid(noise_rates=[0.25], strategies=("none", "adaptive"),
                        sizes=(8,), num_seeds=4, seed_offset=5000)
    from src.analysis import mean_field_table
    raw = mean_field_table(results, "raw_queries")
    assert raw[(0.25, "adaptive")] > raw[(0.25, "none")]
