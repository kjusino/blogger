"""Generates Lorenz-63 trajectory data, builds (x_t -> x_{t+dt}) training
pairs with configurable additive observation noise, trains an nn.MLP
surrogate to approximate the one-step flow map, and exposes the trained
surrogate as an iterable discrete dynamical system (for Lyapunov-spectrum
estimation and long-horizon attractor comparison).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .dynamics import lorenz_flow_map
from .nn import MLP

DEFAULT_DT = 0.01


def generate_trajectory(n_points: int, dt: float = DEFAULT_DT,
                         warmup_steps: int = 1000,
                         state0: np.ndarray = None,
                         seed: int = 0) -> np.ndarray:
    """Generate a Lorenz-63 trajectory of length n_points (+1 for the
    initial state) sampled at fixed dt, after discarding warmup_steps
    transient steps so the trajectory starts on the attractor.
    """
    if state0 is None:
        rng = np.random.default_rng(seed)
        state0 = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.1, size=3)
    state = np.asarray(state0, dtype=float)
    for _ in range(warmup_steps):
        state = lorenz_flow_map(state, dt)
    traj = np.empty((n_points + 1, 3))
    traj[0] = state
    for i in range(n_points):
        state = lorenz_flow_map(state, dt)
        traj[i + 1] = state
    return traj


def make_training_pairs(n_samples: int, dt: float = DEFAULT_DT,
                         noise_level: float = 0.0, seed: int = 0,
                         warmup_steps: int = 1000
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, Y) one-step training pairs X_t -> X_{t+dt} from a Lorenz-63
    trajectory, with i.i.d. Gaussian observation noise of std `noise_level`
    (in the same units as the state, roughly O(1-40) scale for Lorenz-63)
    added independently to both X and Y to emulate noisy sensor
    observations of consecutive states.
    """
    traj = generate_trajectory(n_samples, dt=dt, warmup_steps=warmup_steps,
                                seed=seed)
    X = traj[:-1].copy()
    Y = traj[1:].copy()
    if noise_level > 0.0:
        rng = np.random.default_rng(seed + 10_000)
        X = X + rng.normal(0.0, noise_level, size=X.shape)
        Y = Y + rng.normal(0.0, noise_level, size=Y.shape)
    return X, Y


def normalize_data(X: np.ndarray, Y: np.ndarray):
    """Standardize inputs/outputs by the training data's mean/std (shared
    scaler for X and Y since both live in the same Lorenz state space).
    Returns (Xn, Yn, mean, std) so the surrogate can be un-normalized later.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xn = (X - mean) / std
    Yn = (Y - mean) / std
    return Xn, Yn, mean, std


def train_surrogate(X: np.ndarray, Y: np.ndarray, hidden_width: int = 32,
                     n_hidden_layers: int = 2, epochs: int = 200,
                     batch_size: int = 64, lr: float = 2e-3, seed: int = 0,
                     activation: str = "tanh") -> Tuple[MLP, np.ndarray, np.ndarray]:
    """Train an MLP surrogate on normalized data. Returns (net, mean, std)
    where mean/std are the normalization stats needed to use the network as
    a discrete map on the *raw* (physical) Lorenz state space via
    `make_surrogate_map`.
    """
    Xn, Yn, mean, std = normalize_data(X, Y)
    dim = X.shape[1]
    layer_sizes = [dim] + [hidden_width] * n_hidden_layers + [dim]
    net = MLP(layer_sizes, activation=activation, seed=seed)
    net.train(Xn, Yn, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    return net, mean, std


def make_surrogate_map(net: MLP, mean: np.ndarray, std: np.ndarray):
    """Wrap a trained (normalized-space) surrogate as a discrete map on the
    raw physical Lorenz state space: state -> next_state.
    """
    def step(state: np.ndarray) -> np.ndarray:
        sn = (state - mean) / std
        yn = net.forward(sn)
        return yn * std + mean
    return step


def make_surrogate_jacobian(net: MLP, mean: np.ndarray, std: np.ndarray):
    """Analytic Jacobian of the raw-space surrogate map (chain rule through
    the input/output normalization, which is just an affine rescaling).

    raw_step(x) = net((x - mean)/std) * std + mean
    d raw_step / d x = diag(std) @ J_net((x-mean)/std) @ diag(1/std)
    """
    def jac(state: np.ndarray) -> np.ndarray:
        sn = (state - mean) / std
        Jn = net.jacobian(sn)
        return (std[:, None] * Jn) / std[None, :]
    return jac


def surrogate_val_mse(net: MLP, mean: np.ndarray, std: np.ndarray,
                       X_val: np.ndarray, Y_val: np.ndarray) -> float:
    """One-step validation MSE of the surrogate map in raw physical units
    (the quantity ML papers usually report as "the loss")."""
    step = make_surrogate_map(net, mean, std)
    preds = np.array([step(x) for x in X_val])
    return float(np.mean((preds - Y_val) ** 2))


def iterate_surrogate(net: MLP, mean: np.ndarray, std: np.ndarray,
                       state0: np.ndarray, n_steps: int) -> np.ndarray:
    """Iterate the trained surrogate forward as a discrete dynamical system,
    generating its own long-horizon trajectory (the surrogate's implied
    "attractor" -- possibly wildly different from the true Lorenz attractor
    if the surrogate has not learned the true dynamics faithfully).
    """
    step = make_surrogate_map(net, mean, std)
    state = np.asarray(state0, dtype=float).copy()
    traj = np.empty((n_steps + 1, state.shape[0]))
    traj[0] = state
    for i in range(n_steps):
        state = step(state)
        # Guard against a surrogate that blows up numerically (a real
        # failure mode for high-noise/low-data configs); clip to keep the
        # trajectory finite so downstream metrics don't crash on NaN/inf.
        state = np.clip(state, -1e6, 1e6)
        if not np.all(np.isfinite(state)):
            state = np.zeros_like(state)
        traj[i + 1] = state
    return traj
