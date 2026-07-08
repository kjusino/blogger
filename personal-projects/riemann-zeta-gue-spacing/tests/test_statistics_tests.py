import numpy as np
import pytest

from src.gue_theory import gue_surmise_cdf, poisson_cdf, montgomery_pair_correlation
from src.statistics_tests import (
    ks_against_cdf,
    pair_correlation_histogram,
    pair_correlation_l2_error,
    repulsion_fraction,
)


def _poisson_process(rng, n, rate=1.0):
    spacings = rng.exponential(1.0 / rate, size=n)
    return spacings, np.cumsum(spacings)


def test_ks_against_cdf_accepts_matching_null():
    rng = np.random.default_rng(0)
    spacings, _ = _poisson_process(rng, 4000)
    stat, p_value = ks_against_cdf(spacings, poisson_cdf)
    assert stat < 0.03
    assert p_value > 0.05


def test_ks_against_cdf_rejects_mismatched_null():
    rng = np.random.default_rng(0)
    spacings, _ = _poisson_process(rng, 4000)
    stat, p_value = ks_against_cdf(spacings, gue_surmise_cdf)
    assert stat > 0.2
    assert p_value < 1e-6


def test_pair_correlation_of_poisson_process_is_flat_near_one():
    rng = np.random.default_rng(1)
    _, positions = _poisson_process(rng, 6000)
    bin_centers, r2_hat = pair_correlation_histogram(positions, u_max=3.0, n_bins=25)
    assert np.mean(r2_hat) == pytest.approx(1.0, abs=0.08)
    # No systematic trend with u for an uncorrelated process.
    assert np.std(r2_hat) < 0.15


def test_pair_correlation_matches_flat_reference_better_than_montgomery_for_poisson():
    rng = np.random.default_rng(2)
    _, positions = _poisson_process(rng, 6000)
    bin_centers, r2_hat = pair_correlation_histogram(positions, u_max=3.0, n_bins=25)
    err_flat = pair_correlation_l2_error(bin_centers, r2_hat, lambda u: np.ones_like(u))
    err_montgomery = pair_correlation_l2_error(bin_centers, r2_hat, montgomery_pair_correlation)
    assert err_flat < err_montgomery


def test_pair_correlation_raises_on_too_small_window():
    with pytest.raises(ValueError):
        pair_correlation_histogram(np.array([0.0, 0.5, 1.0]), u_max=3.0, n_bins=10)


def test_repulsion_fraction_matches_poisson_theory():
    rng = np.random.default_rng(3)
    spacings, _ = _poisson_process(rng, 20_000)
    frac = repulsion_fraction(spacings, threshold=0.2)
    assert frac == pytest.approx(1 - np.exp(-0.2), abs=0.01)


def test_repulsion_fraction_is_zero_for_shifted_lattice():
    # A perfectly rigid point process (all spacings == 1) has zero mass
    # below any threshold < 1 -- the extreme case of level repulsion.
    spacings = np.ones(100)
    assert repulsion_fraction(spacings, threshold=0.5) == 0.0
