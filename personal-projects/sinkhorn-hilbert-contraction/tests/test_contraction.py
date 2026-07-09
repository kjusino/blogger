import numpy as np
import pytest

from src.cost_matrices import build_cost
from src.contraction import (
    cost_diameter_brute_force,
    cost_diameter_fast,
    hilbert_diameter,
    theoretical_contraction_rate,
)


@pytest.mark.parametrize("family", ["random_points", "clustered_points", "grid_1d", "iid_random"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_diameter_matches_brute_force(family, seed):
    rng = np.random.default_rng(seed)
    n, m = 6, 5  # small enough for O(n^2 m^2) brute force to be instant
    C = build_cost(family, n, m, rng)
    fast = cost_diameter_fast(C)
    brute = cost_diameter_brute_force(C)
    assert fast == pytest.approx(brute, rel=1e-9, abs=1e-9)


def test_diameter_is_zero_for_constant_cost_matrix():
    C = np.full((5, 4), 3.7)
    assert cost_diameter_fast(C) == pytest.approx(0.0, abs=1e-12)
    assert cost_diameter_brute_force(C) == pytest.approx(0.0, abs=1e-12)


def test_diameter_is_nonnegative():
    rng = np.random.default_rng(7)
    for family in ["random_points", "clustered_points", "grid_1d", "iid_random"]:
        C = build_cost(family, 10, 8, rng)
        assert cost_diameter_fast(C) >= -1e-12


def test_hilbert_diameter_of_kernel_matches_cost_diameter_over_eps():
    rng = np.random.default_rng(9)
    C = build_cost("random_points", 6, 6, rng)
    for eps in (2.0, 0.5, 0.1):
        K = np.exp(-C / eps)
        expected = cost_diameter_fast(C) / eps
        assert hilbert_diameter(K) == pytest.approx(expected, rel=1e-6)


def test_theoretical_contraction_rate_is_in_unit_interval():
    # tanh(x) saturates to exactly 1.0 in float64 once x gtrsim 19, so
    # kappa == 1.0 exactly is a legitimate (if uninformative) floating-point
    # outcome at extreme Delta(K)/eps -- the interval is closed at 1.0, not
    # open. The experiment sweep keeps eps large enough to avoid this
    # regime for the configurations it actually measures rates on.
    rng = np.random.default_rng(11)
    C = build_cost("clustered_points", 12, 10, rng)
    for eps in (5.0, 1.0, 0.2, 0.02):
        kappa = theoretical_contraction_rate(C, eps)
        assert 0.0 <= kappa <= 1.0

    # In a non-saturating regime the rate must be strictly below 1.
    for eps in (5.0, 1.0, 0.2):
        assert theoretical_contraction_rate(C, eps) < 1.0


def test_theoretical_contraction_rate_increases_toward_one_as_eps_shrinks():
    """Smaller eps => larger condition number => rate closer to 1 (slower
    convergence). Delta(K) = D(C)/eps is monotone decreasing in eps whenever
    D(C) > 0, so kappa = tanh(Delta/4)^2 is monotone increasing as eps -> 0.
    """
    rng = np.random.default_rng(13)
    C = build_cost("random_points", 10, 10, rng)
    epsilons = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05]
    rates = [theoretical_contraction_rate(C, eps) for eps in epsilons]
    assert all(r2 >= r1 - 1e-12 for r1, r2 in zip(rates, rates[1:]))
    assert rates[-1] > rates[0]


def test_theoretical_contraction_rate_zero_for_degenerate_constant_cost():
    C = np.full((4, 4), 1.0)
    assert theoretical_contraction_rate(C, eps=0.5) == pytest.approx(0.0, abs=1e-12)
