"""Lorenz-63 dynamics, its analytic Jacobian, a fixed-step RK4 integrator,
and the logistic map (used only to validate the Lyapunov estimator against
an exactly-known analytic exponent, ln(2) at r=4).

All functions operate on plain numpy arrays and have no external
dependencies beyond numpy, keeping this subproject self-contained.
"""
from __future__ import annotations

import numpy as np

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0


def lorenz_rhs(state: np.ndarray, sigma: float = SIGMA, rho: float = RHO,
               beta: float = BETA) -> np.ndarray:
    """Right-hand side of the Lorenz-63 ODE system: dstate/dt = f(state).

    state = [x, y, z], autonomous (no explicit time dependence).
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


def lorenz_jacobian(state: np.ndarray, sigma: float = SIGMA, rho: float = RHO,
                     beta: float = BETA) -> np.ndarray:
    """Analytic Jacobian of the Lorenz-63 vector field, d f_i / d state_j."""
    x, y, z = state
    return np.array([
        [-sigma, sigma, 0.0],
        [rho - z, -1.0, -x],
        [y, x, -beta],
    ])


def rk4_step(rhs, state: np.ndarray, dt: float) -> np.ndarray:
    """A single fixed-step classical RK4 update for an autonomous ODE
    dstate/dt = rhs(state).
    """
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(rhs, state0: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    """Integrate an autonomous ODE with fixed-step RK4 for n_steps steps.

    Returns an array of shape (n_steps + 1, dim) including the initial state.
    """
    state0 = np.asarray(state0, dtype=float)
    dim = state0.shape[0]
    traj = np.empty((n_steps + 1, dim))
    traj[0] = state0
    state = state0.copy()
    for i in range(n_steps):
        state = rk4_step(rhs, state, dt)
        traj[i + 1] = state
    return traj


def lorenz_flow_map(state: np.ndarray, dt: float, sigma: float = SIGMA,
                     rho: float = RHO, beta: float = BETA) -> np.ndarray:
    """The Lorenz-63 flow map advanced by a fixed dt via a single RK4 step
    with that dt as the step size (used to build the discrete-time map whose
    Lyapunov spectrum we estimate, and to build surrogate training pairs).

    Note: this uses one RK4 step of size dt as the "true" one-step map.
    For very chaotic systems and larger dt, sub-stepping internally would be
    more accurate; dt=0.01 (the default used throughout this project) is
    small enough that a single RK4(dt) step matches a finely sub-stepped
    integration to high precision.
    """
    rhs = lambda s: lorenz_rhs(s, sigma=sigma, rho=rho, beta=beta)
    return rk4_step(rhs, state, dt)


def logistic_map(x: float, r: float = 4.0) -> float:
    """The logistic map x -> r * x * (1 - x)."""
    return r * x * (1.0 - x)


def logistic_iterate(x0: float, r: float, n_steps: int) -> np.ndarray:
    """Iterate the logistic map n_steps times, returning the full orbit
    (length n_steps + 1, including x0).
    """
    traj = np.empty(n_steps + 1)
    traj[0] = x0
    x = x0
    for i in range(n_steps):
        x = logistic_map(x, r)
        traj[i + 1] = x
    return traj
