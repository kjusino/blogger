import numpy as np
import pytest

from src.stats_utils import (
    block_bootstrap_ci,
    exact_spearman,
    parabolic_peak,
    relative_error,
)


def test_parabolic_peak_recovers_exact_quadratic_maximum():
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ys = -(xs - 3.2) ** 2 + 10.0
    x_peak, y_peak = parabolic_peak(xs, ys)
    assert x_peak == pytest.approx(3.2, abs=1e-9)
    assert y_peak == pytest.approx(10.0, abs=1e-9)


def test_parabolic_peak_falls_back_at_grid_endpoint():
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    ys = np.array([1.0, 2.0, 3.0, 4.0])  # strictly increasing -> argmax at endpoint
    x_peak, y_peak = parabolic_peak(xs, ys)
    assert x_peak == 3.0
    assert y_peak == 4.0


def test_block_bootstrap_ci_brackets_true_mean_for_iid_data():
    rng = np.random.default_rng(0)
    series = rng.normal(loc=5.0, scale=1.0, size=2000)
    lo, hi = block_bootstrap_ci(series, block_size=20, n_resamples=500, seed=1)
    assert lo < series.mean() < hi
    assert hi - lo < 1.0  # reasonably tight for n=2000


def test_relative_error_basic_cases():
    assert relative_error(1.1, 1.0) == pytest.approx(0.1)
    assert relative_error(0.9, 1.0) == pytest.approx(0.1)
    assert np.isnan(relative_error(1.0, 0.0))


def test_exact_spearman_perfect_monotonic_relationship():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([10, 20, 30, 40, 50])
    rho, p = exact_spearman(x, y)
    assert rho == pytest.approx(1.0)
    assert p == pytest.approx(2 / 120, abs=1e-9)  # 2 extreme orderings out of 5!


def test_exact_spearman_perfect_inverse_relationship():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([50, 40, 30, 20, 10])
    rho, p = exact_spearman(x, y)
    assert rho == pytest.approx(-1.0)
    assert p == pytest.approx(2 / 120, abs=1e-9)


def test_exact_spearman_no_relationship_gives_large_p():
    rng = np.random.default_rng(3)
    x = np.arange(9)
    y = rng.permutation(9)
    rho, p = exact_spearman(x, y)
    assert -1.0 <= rho <= 1.0
    assert 0.0 <= p <= 1.0
