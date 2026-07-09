import math

import numpy as np

from src import theory
from src.experiment import (
    L_of_n,
    find_empirical_threshold_angle,
    fit_power_law_exponent,
    k_of_n,
    run_and_or_threshold_experiment,
    run_scaling_experiment,
    run_single_hash_experiment,
)


def test_run_single_hash_experiment_small_scale_shape_and_accuracy():
    df = run_single_hash_experiment(
        thetas=[0.3, 1.2, 2.5], dims=[16, 64], num_trials=8000, seed=123
    )
    assert set(df.columns) >= {"dim", "theta", "empirical_prob", "theory_prob", "stderr", "abs_error", "z_score"}
    assert len(df) == 6
    # At 8000 trials the theory should be well within a handful of stderrs.
    assert df["z_score"].max() < 6.0


def test_run_and_or_threshold_experiment_shape_and_monotonic_theory():
    df = run_and_or_threshold_experiment(
        k=4, L=8, thetas=[0.3, 0.8, 1.3, 1.8, 2.3], num_trials=500, dim=20, seed=321
    )
    assert len(df) == 5
    # Theory recall must be monotonically decreasing in theta.
    theory_vals = df.sort_values("theta")["theory_recall"].to_numpy()
    assert all(theory_vals[i] >= theory_vals[i + 1] for i in range(len(theory_vals) - 1))


def test_find_empirical_threshold_angle_on_synthetic_perfect_curve():
    """Feed a DataFrame whose 'empirical_recall' exactly matches theory, and
    verify the interpolated threshold matches theory.threshold_angle."""
    import pandas as pd

    k, L = 6, 10
    theta_star = theory.threshold_angle(k, L)
    thetas = np.linspace(0.05, math.pi - 0.05, 40)
    df = pd.DataFrame(
        {
            "theta": thetas,
            "empirical_recall": [theory.or_of_bands_prob(t, k, L) for t in thetas],
        }
    )
    found = find_empirical_threshold_angle(df)
    assert abs(found - theta_star) < 0.05


def test_k_of_n_and_L_of_n_grow_with_n():
    p2 = 0.5
    rho = 0.3
    k_small = k_of_n(100, p2)
    k_large = k_of_n(100000, p2)
    assert k_large > k_small

    L_small = L_of_n(100, rho)
    L_large = L_of_n(100000, rho)
    assert L_large >= L_small


def test_run_scaling_experiment_small_scale_produces_sane_candidate_counts():
    df = run_scaling_experiment(
        near_theta=math.pi / 6,
        far_theta=math.pi / 2,
        dim=16,
        n_list=[50, 200, 800],
        trials_per_n=5,
        seed=99,
    )
    assert list(df["n"]) == [50, 200, 800]
    assert (df["mean_candidates"] >= 0).all()
    assert df.attrs["rho_theory"] > 0


def test_fit_power_law_exponent_recovers_known_exponent():
    rng = np.random.default_rng(0)
    n_values = np.array([100, 300, 1000, 3000, 10000, 30000], dtype=float)
    true_rho = 0.4
    y_values = 2.0 * n_values**true_rho
    rho_hat, r_squared = fit_power_law_exponent(n_values, y_values)
    assert abs(rho_hat - true_rho) < 1e-6
    assert r_squared > 0.999


def test_fit_power_law_exponent_handles_zero_candidate_counts():
    n_values = np.array([10, 100, 1000, 10000], dtype=float)
    y_values = np.array([0.0, 0.0, 2.0, 8.0])
    rho_hat, r_squared = fit_power_law_exponent(n_values, y_values)
    assert not math.isnan(rho_hat)
