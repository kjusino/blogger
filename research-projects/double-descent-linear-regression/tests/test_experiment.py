import numpy as np

from src.experiment import run_config, run_grid, run_ridge_sweep
from src.theory import exact_risk


def test_run_config_matches_theory_underparameterized():
    n, p, r2, sigma2 = 80, 30, 1.0, 1.0
    risks, emp_bias2, emp_variance = run_config(
        n, p, r2, sigma2, n_trials=300, seed=0
    )
    theory = exact_risk(n, p, r2, sigma2)
    assert abs(risks.mean() - theory) / theory < 0.15
    assert emp_bias2 < 0.05  # should be ~0 in the underparameterized regime
    assert abs(emp_variance - theory) / theory < 0.2


def test_run_config_matches_theory_overparameterized():
    n, p, r2, sigma2 = 60, 150, 1.0, 1.0
    risks, emp_bias2, emp_variance = run_config(
        n, p, r2, sigma2, n_trials=300, seed=1
    )
    theory = exact_risk(n, p, r2, sigma2)
    assert abs(risks.mean() - theory) / theory < 0.2


def test_run_grid_skips_configs_near_threshold():
    results = run_grid(
        n=50, gammas=[0.5, 0.98, 1.0, 1.02, 2.0], r2=1.0, sigma2=1.0,
        n_trials=20, seed=0, min_gap=2,
    )
    gammas_kept = sorted(r["gamma"] for r in results)
    assert 1.0 not in gammas_kept
    assert 0.98 not in gammas_kept  # p=49, |49-50|=1 < min_gap
    assert 1.02 not in gammas_kept  # p=51, |51-50|=1 < min_gap
    assert any(abs(g - 0.5) < 1e-9 for g in gammas_kept)
    assert any(abs(g - 2.0) < 1e-9 for g in gammas_kept)


def test_run_grid_overall_agreement_with_theory():
    results = run_grid(
        n=60, gammas=[0.3, 0.6, 1.5, 3.0], r2=1.0, sigma2=1.0,
        n_trials=250, seed=7,
    )
    assert len(results) == 4
    mean_rel_err = np.mean([r["relative_error"] for r in results])
    assert mean_rel_err < 0.15


def test_ridge_sweep_suppresses_peak_at_gamma_one():
    results = run_ridge_sweep(
        n=40, gammas=[1.0], lambdas=[0.0, 1.0, 20.0], r2=1.0, sigma2=1.0,
        n_trials=100, seed=3,
    )
    by_lambda = {r["lambda"]: r["empirical_risk"] for r in results}
    assert by_lambda[20.0] < by_lambda[1.0] < by_lambda[0.0]
