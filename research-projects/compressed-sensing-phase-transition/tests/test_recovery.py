import numpy as np
import pytest

from src.recovery import check_recovery, solve_basis_pursuit
from src.sensing import gaussian_sensing_matrix, sparse_signal
from src.theory import phase_transition_delta


def test_recovers_well_above_threshold():
    rng = np.random.default_rng(10)
    n, rho = 80, 0.1
    k = round(rho * n)
    delta = phase_transition_delta(rho) * 1.6
    m = round(delta * n)

    successes = 0
    trials = 12
    for _ in range(trials):
        A = gaussian_sensing_matrix(m, n, rng)
        x0 = sparse_signal(n, k, rng)
        y = A @ x0
        x_hat = solve_basis_pursuit(A, y)
        successes += check_recovery(x_hat, x0)
    assert successes / trials >= 0.9


def test_fails_well_below_threshold():
    rng = np.random.default_rng(11)
    n, rho = 80, 0.1
    k = round(rho * n)
    delta = phase_transition_delta(rho) * 0.5
    m = round(delta * n)

    successes = 0
    trials = 12
    for _ in range(trials):
        A = gaussian_sensing_matrix(m, n, rng)
        x0 = sparse_signal(n, k, rng)
        y = A @ x0
        x_hat = solve_basis_pursuit(A, y)
        successes += check_recovery(x_hat, x0)
    assert successes / trials <= 0.2


def test_exact_recovery_on_noiseless_full_rank_square_system():
    # m == n: A is square and (almost surely) invertible, so Ax=y has a
    # unique solution regardless of sparsity, and basis pursuit must find it.
    rng = np.random.default_rng(12)
    n = 15
    A = gaussian_sensing_matrix(n, n, rng)
    x0 = sparse_signal(n, 4, rng)
    y = A @ x0
    x_hat = solve_basis_pursuit(A, y)
    assert check_recovery(x_hat, x0, tol=1e-4)


def test_check_recovery_detects_mismatch():
    x0 = np.array([1.0, 0.0, -2.0])
    x_hat = np.array([1.0, 0.0, -1.0])
    assert not check_recovery(x_hat, x0)


def test_check_recovery_within_tolerance():
    x0 = np.array([1.0, 0.0, -2.0])
    x_hat = x0 + 1e-8
    assert check_recovery(x_hat, x0, tol=1e-5)


def test_check_recovery_none_input():
    x0 = np.array([1.0, 2.0])
    assert not check_recovery(None, x0)


def test_solve_basis_pursuit_rejects_shape_mismatch():
    rng = np.random.default_rng(13)
    A = gaussian_sensing_matrix(5, 10, rng)
    y = np.ones(4)
    with pytest.raises(ValueError):
        solve_basis_pursuit(A, y)
