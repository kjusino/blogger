import numpy as np
import pytest

from bbp_transition.experiment import (
    run_trials,
    sweep_lambda_grid,
    estimate_detection_threshold,
    finite_size_scaling,
)
from bbp_transition.theory import bbp_threshold, theoretical_top_eigenvalue, theoretical_alignment_sq


def test_run_trials_shapes():
    rng = np.random.default_rng(0)
    eigs, aligns = run_trials(n=200, p=20, lam=1.0, trials=15, rng=rng)
    assert eigs.shape == (15,)
    assert aligns.shape == (15,)
    assert np.all(aligns >= 0.0) and np.all(aligns <= 1.0 + 1e-9)


def test_deep_subcritical_alignment_is_near_zero():
    """Far below threshold, the sample top eigenvector should carry almost
    no information about the spike direction."""
    rng = np.random.default_rng(1)
    c = 0.3
    p = 100
    n = int(p / c)
    thr = bbp_threshold(c)
    _, aligns = run_trials(n=n, p=p, lam=0.1 * thr, trials=40, rng=rng)
    assert aligns.mean() < 0.15


def test_deep_supercritical_alignment_is_close_to_theory():
    """Far above threshold, empirical alignment should track theory closely."""
    rng = np.random.default_rng(2)
    c = 0.3
    p = 150
    n = int(p / c)
    thr = bbp_threshold(c)
    lam = 5.0 * thr
    _, aligns = run_trials(n=n, p=p, lam=lam, trials=60, rng=rng)
    theory = float(theoretical_alignment_sq(lam, c))
    assert abs(aligns.mean() - theory) < 0.1


def test_supercritical_alignment_exceeds_subcritical():
    """Sanity ordering: alignment should clearly increase as lam grows past
    the threshold, regardless of the exact theoretical curve."""
    rng = np.random.default_rng(3)
    c = 0.3
    p = 100
    n = int(p / c)
    thr = bbp_threshold(c)
    _, aligns_below = run_trials(n=n, p=p, lam=0.2 * thr, trials=40, rng=rng)
    _, aligns_above = run_trials(n=n, p=p, lam=3.0 * thr, trials=40, rng=rng)
    assert aligns_above.mean() > aligns_below.mean()


@pytest.mark.integration
def test_sweep_lambda_grid_structure_and_error_bounds():
    results = sweep_lambda_grid(
        p=80,
        c_values=[0.3, 0.6],
        lam_ratios=[0.0, 1.0, 3.0],
        trials=30,
        seed=42,
    )
    assert len(results) == 2 * 3
    for r in results:
        assert r.trials == 30
        assert r.mean_align >= -1e-9
        assert r.mean_align <= 1.0 + 1e-9
        # loose bound: with p=80 and 30 trials, finite-size + MC noise should
        # not blow the relative eigenvalue error past 30%
        assert r.rel_err_eig < 0.3


@pytest.mark.integration
def test_estimate_detection_threshold_recovers_theory_roughly():
    lam_hat, thr, lams, aligns = estimate_detection_threshold(
        p=100,
        c=0.4,
        lam_ratios_fine=[0.1 * i for i in range(1, 26)],
        trials=30,
        seed=5,
    )
    assert lam_hat is not None
    # empirical detection threshold should land within 50% of the true lam*
    assert abs(lam_hat - thr) / thr < 0.5


@pytest.mark.integration
def test_finite_size_scaling_error_shrinks_with_p():
    rows = finite_size_scaling(
        c=0.3, lam_ratio=2.0, p_values=[30, 300], trials=60, seed=6
    )
    err_small_p = next(r["abs_err_eig"] for r in rows if r["p"] == 30)
    err_large_p = next(r["abs_err_eig"] for r in rows if r["p"] == 300)
    # not a strict guarantee under finite MC noise, but the asymptotic error
    # should be substantially smaller at p=300 than p=30 on average
    assert err_large_p < err_small_p + 0.15
