import numpy as np

from src.dynamics import lorenz_flow_map, logistic_map
from src.lyapunov import lyapunov_spectrum, finite_difference_jacobian


def test_logistic_map_lyapunov_matches_analytic_ln2():
    # The logistic map x -> 4x(1-x) has an exactly-known Lyapunov exponent
    # of ln(2) ~= 0.6931 (a classical result: it is topologically conjugate
    # to the tent map / doubling map via x = sin^2(pi*theta/2)). This is our
    # ground-truth sanity check for the QR/Benettin estimator itself, fully
    # independent of Lorenz-63 and of the neural-network surrogate code.
    def step(x, r=4.0):
        return np.array([logistic_map(x[0], r)])

    exponents = lyapunov_spectrum(
        step, np.array([0.234]), n_iters=200_000, dt=1.0, warmup=1_000,
        renorm_interval=1,
    )
    assert exponents.shape == (1,)
    # Tight tolerance since this is a 1-D map with a huge number of
    # iterations available cheaply; measured error is ~5e-6 in practice.
    assert abs(exponents[0] - np.log(2)) < 0.01


def test_logistic_map_lyapunov_robust_to_initial_condition():
    # Sanity check: the estimate shouldn't depend meaningfully on which
    # (generic, irrational-like) initial condition we start from, since the
    # invariant measure is ergodic.
    def step(x, r=4.0):
        return np.array([logistic_map(x[0], r)])

    exponents = lyapunov_spectrum(
        step, np.array([0.71]), n_iters=200_000, dt=1.0, warmup=1_000,
    )
    assert abs(exponents[0] - np.log(2)) < 0.01


def test_lorenz_lyapunov_spectrum_matches_literature():
    # Well-established literature values for the classical Lorenz-63
    # parameters (sigma=10, rho=28, beta=8/3), e.g. Wolf et al. 1985 /
    # Sprott's "Chaos and Time-Series Analysis": lambda1 ~ 0.905 (chaotic),
    # lambda2 ~ 0.0 (associated with the flow direction, always ~0 for a
    # continuous-time flow), lambda3 ~ -14.57 (strongly contracting,
    # consistent with the well-known Lyapunov-dimension / volume-contraction
    # identity sum(lambda_i) ~= -(sigma + 1 + beta) = -13.67).
    #
    # Tolerances below were chosen based on what this finite-difference QR
    # estimator actually achieves in ~20k iterations at dt=0.01 (measured
    # errors in development were ~0.005, ~0.003, ~0.006 respectively) with
    # generous headroom for machine-to-machine floating point variation.
    dt = 0.01
    step = lambda s: lorenz_flow_map(s, dt)
    x0 = np.array([1.0, 1.0, 1.0])

    exponents = lyapunov_spectrum(
        step, x0, n_iters=20_000, dt=dt, warmup=2_000, renorm_interval=1,
    )

    assert exponents.shape == (3,)
    lambda1, lambda2, lambda3 = exponents
    assert abs(lambda1 - 0.905) < 0.08
    assert abs(lambda2 - 0.0) < 0.1
    assert abs(lambda3 - (-14.57)) < 1.0

    # Exponents should be sorted descending.
    assert lambda1 >= lambda2 >= lambda3


def test_finite_difference_jacobian_matches_analytic_lorenz_jacobian():
    from src.dynamics import lorenz_jacobian, lorenz_rhs

    state = np.array([3.1, -2.2, 18.0])
    J_fd = finite_difference_jacobian(lorenz_rhs, state, eps=1e-6)
    J_analytic = lorenz_jacobian(state)
    np.testing.assert_allclose(J_fd, J_analytic, atol=1e-6)
