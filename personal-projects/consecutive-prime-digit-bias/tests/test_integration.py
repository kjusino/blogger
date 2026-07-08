import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_experiments import analyze_window, make_window_edges, run  # noqa: E402


def test_make_window_edges_spans_full_range():
    edges = make_window_edges(10**8, 8)
    assert edges[0] >= 10_000
    assert edges[-1] == 10**8
    assert np.all(np.diff(edges) > 0)


def test_analyze_window_structure():
    p_arr = np.array([11, 13, 17, 101, 103])
    q_arr = np.array([13, 17, 19, 103, 107])
    result = analyze_window(p_arr, q_arr, lo=10, hi=200)
    assert result["total_pairs"] == 5
    assert 0.0 <= result["same_digit_fraction"] <= 1.0
    assert sum(sum(row) for row in result["matrix"]) == 5


def test_full_pipeline_small_scale_is_deterministic_and_well_formed():
    # n_max = 100_000 is fixed and small enough to run in a unit test; the
    # sieve is deterministic so every number below is an exact regression
    # check, not a statistical estimate.
    results = run(n_max=100_000, num_windows=4)

    assert results["n_max"] == 100_000
    assert len(results["windows"]) == 4

    total_pairs_across_windows = sum(w["total_pairs"] for w in results["windows"])
    assert total_pairs_across_windows == results["overall"]["total_pairs"]

    for w in results["windows"]:
        if w["total_pairs"] > 0:
            assert 0.0 <= w["same_digit_fraction"] <= 1.0
            assert -0.25 <= w["bias"] <= 0.75

    # The known headline effect: even at this modest scale, consecutive
    # primes repeat their last digit noticeably less than the naive 1/4.
    assert results["overall"]["same_digit_fraction"] < 0.25
    # Windows start at 10_000, so pairs with p_n <= 10_000 are excluded from
    # the windowed analysis; this is an exact count for n_max=100_000, num_windows=4.
    assert results["overall"]["total_pairs"] == 8362

    fit = results["decay_fit"]
    assert "slope" in fit and "r_squared" in fit
