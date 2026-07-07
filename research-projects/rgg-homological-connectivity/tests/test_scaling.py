import numpy as np
import pytest

from src.scaling import bootstrap_ci, transition_width


def test_bootstrap_ci_contains_true_mean_for_normal_samples():
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=5.0, scale=1.0, size=500)
    point, lo, hi = bootstrap_ci(samples, rng, statistic=np.mean, n_boot=2000)
    assert lo < point < hi
    assert lo < 5.0 < hi  # true mean should fall inside a 95% CI virtually always


def test_bootstrap_ci_shrinks_with_more_data():
    rng = np.random.default_rng(1)
    small = rng.normal(size=20)
    large = rng.normal(size=2000)
    _, lo_s, hi_s = bootstrap_ci(small, rng, n_boot=2000)
    _, lo_l, hi_l = bootstrap_ci(large, rng, n_boot=2000)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_bootstrap_ci_rejects_empty_samples():
    import pytest

    with pytest.raises(ValueError):
        bootstrap_ci([], np.random.default_rng(0))


def test_transition_width_on_step_function():
    grid_spacing = 0.01
    r_grid = np.linspace(0, 1, 101)
    # Perfect step at r=0.5: np.interp linearly bridges the single grid cell
    # where the jump happens, so the width is bounded by grid spacing, not
    # exactly zero.
    frac = (r_grid >= 0.5).astype(float)
    width, r_low, r_high = transition_width(r_grid, frac, low=0.1, high=0.9)
    assert 0 <= width < grid_spacing
    assert r_low == pytest.approx(0.5, abs=grid_spacing)


def test_transition_width_on_linear_ramp():
    r_grid = np.linspace(0, 1, 1001)
    frac = np.clip(r_grid, 0, 1)  # frac(r) = r, a linear ramp from 0 to 1
    width, r_low, r_high = transition_width(r_grid, frac, low=0.1, high=0.9)
    assert width == pytest.approx(0.8, abs=1e-2)
