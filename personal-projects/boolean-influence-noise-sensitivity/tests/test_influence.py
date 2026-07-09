import numpy as np
import pytest

from src.functions import MajorityFunction, ParityFunction, TribesFunction, majority_influence_exact, tribes_influence_exact
from src.influence import exact_truth_table, monte_carlo_influence


def test_parity_influence_is_exactly_one_everywhere():
    f = ParityFunction(n=8)
    result = exact_truth_table(f)
    assert np.allclose(result.per_coordinate, 1.0)
    assert result.total_influence == pytest.approx(8.0)
    assert result.variance == pytest.approx(1.0)


def test_parity_variance_is_one_for_odd_and_even_n():
    for n in (3, 4, 6, 7):
        f = ParityFunction(n=n)
        result = exact_truth_table(f)
        assert result.variance == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("n", [3, 5, 7, 9])
def test_majority_influence_exact_matches_brute_force(n):
    f = MajorityFunction(n=n)
    brute = exact_truth_table(f)
    closed_form = majority_influence_exact(n)
    assert closed_form == pytest.approx(brute.per_coordinate[0], rel=1e-9)
    # Symmetric function: every coordinate has identical influence.
    assert np.allclose(brute.per_coordinate, brute.per_coordinate[0])


@pytest.mark.parametrize("w,s", [(2, 3), (3, 2), (2, 4)])
def test_tribes_influence_exact_matches_brute_force(w, s):
    f = TribesFunction(w=w, s=s)
    brute = exact_truth_table(f)
    closed_form = tribes_influence_exact(w, s)
    assert closed_form == pytest.approx(brute.per_coordinate[0], rel=1e-9)
    assert np.allclose(brute.per_coordinate, brute.per_coordinate[0], atol=1e-9)


def test_monte_carlo_influence_agrees_with_exact_for_majority():
    f = MajorityFunction(n=9)
    exact = exact_truth_table(f)
    rng = np.random.default_rng(0)
    mc = monte_carlo_influence(f, n_samples=20000, rng=rng)
    assert mc.total_influence == pytest.approx(exact.total_influence, abs=0.15)


def test_monte_carlo_influence_agrees_with_exact_for_tribes():
    f = TribesFunction(w=2, s=3)
    exact = exact_truth_table(f)
    rng = np.random.default_rng(1)
    mc = monte_carlo_influence(f, n_samples=20000, rng=rng)
    assert mc.total_influence == pytest.approx(exact.total_influence, abs=0.15)


def test_exact_truth_table_rejects_large_n():
    f = ParityFunction(n=25)
    with pytest.raises(ValueError):
        exact_truth_table(f)


def test_monte_carlo_influence_partial_coordinates_extrapolates_total():
    f = MajorityFunction(n=101)
    rng = np.random.default_rng(2)
    subset = np.arange(10)
    mc = monte_carlo_influence(f, n_samples=5000, rng=rng, coordinates=subset)
    exact_total = 101 * majority_influence_exact(101)
    assert mc.total_influence == pytest.approx(exact_total, rel=0.25)
