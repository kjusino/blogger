import numpy as np
import pytest

from src.dynamics import (
    lorenz_rhs, lorenz_jacobian, rk4_step, integrate, lorenz_flow_map,
    logistic_map, logistic_iterate, SIGMA, RHO, BETA,
)


def test_lorenz_rhs_known_point():
    # By hand at state = [1, 1, 1], sigma=10, rho=28, beta=8/3:
    #   dx = sigma*(y-x) = 10*(1-1) = 0
    #   dy = x*(rho-z) - y = 1*(28-1) - 1 = 26
    #   dz = x*y - beta*z = 1*1 - (8/3)*1 = 1 - 8/3 = -5/3
    state = np.array([1.0, 1.0, 1.0])
    d = lorenz_rhs(state)
    expected = np.array([0.0, 26.0, 1.0 - BETA])
    np.testing.assert_allclose(d, expected, atol=1e-12)


def test_lorenz_rhs_origin_is_fixed_point():
    # The origin is always an equilibrium of the Lorenz system.
    d = lorenz_rhs(np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(d, np.zeros(3), atol=1e-12)


def test_lorenz_jacobian_matches_finite_difference():
    state = np.array([2.3, -1.7, 15.2])
    J_analytic = lorenz_jacobian(state)
    eps = 1e-6
    J_fd = np.zeros((3, 3))
    for j in range(3):
        step = np.zeros(3)
        step[j] = eps
        f_plus = lorenz_rhs(state + step)
        f_minus = lorenz_rhs(state - step)
        J_fd[:, j] = (f_plus - f_minus) / (2 * eps)
    np.testing.assert_allclose(J_analytic, J_fd, atol=1e-6)


def test_rk4_fourth_order_convergence():
    # Simple linear ODE with a known analytic solution: dy/dt = -y,
    # y(0) = 1  =>  y(t) = exp(-t). RK4's local/global error should scale
    # as O(dt^4), so halving dt should shrink the global error by ~2^4=16x.
    rhs = lambda y: -y
    t_final = 1.0
    y0 = np.array([1.0])
    true_val = np.array([np.exp(-t_final)])

    def global_error(dt):
        n_steps = int(round(t_final / dt))
        traj = integrate(rhs, y0, dt, n_steps)
        return np.abs(traj[-1] - true_val)[0]

    err_coarse = global_error(0.02)
    err_fine = global_error(0.01)
    ratio = err_coarse / err_fine
    # Expect close to 16x; allow generous slack since this is a discrete
    # asymptotic property, not exact at finite dt.
    assert 12.0 < ratio < 20.0, f"RK4 convergence ratio {ratio} not close to 16"


def test_rk4_step_matches_hand_solvable_case():
    # dy/dt = y (exponential growth), y(0)=1: a single RK4 step should match
    # exp(dt) to O(dt^5) accuracy.
    rhs = lambda y: y
    dt = 0.1
    y1 = rk4_step(rhs, np.array([1.0]), dt)
    np.testing.assert_allclose(y1, np.array([np.exp(dt)]), atol=1e-6)


def test_lorenz_flow_map_shape_and_finite():
    state = np.array([1.0, 1.0, 1.0])
    nxt = lorenz_flow_map(state, dt=0.01)
    assert nxt.shape == (3,)
    assert np.all(np.isfinite(nxt))


def test_logistic_map_known_values():
    # r=4, x0=0.2: x1 = 4*0.2*0.8 = 0.64; x2 = 4*0.64*0.36 = 0.9216
    assert logistic_map(0.2, r=4.0) == pytest.approx(0.64)
    assert logistic_map(0.64, r=4.0) == pytest.approx(0.9216)


def test_logistic_iterate_matches_manual_stepping():
    traj = logistic_iterate(0.2, r=4.0, n_steps=5)
    assert traj[0] == pytest.approx(0.2)
    x = 0.2
    for i in range(5):
        x = logistic_map(x, r=4.0)
        assert traj[i + 1] == pytest.approx(x)
