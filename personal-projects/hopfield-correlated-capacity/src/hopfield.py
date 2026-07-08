"""Classical Hopfield network: Hebbian storage + asynchronous update dynamics.

W_ij = (1/N) sum_mu xi^mu_i xi^mu_j,  W_ii = 0

Update rule (asynchronous / sequential): pick a neuron i, set
    s_i <- sign( sum_j W_ij s_j )
Run in sweeps of N random-order single-neuron updates; a sweep with zero
flips signals convergence to a fixed point.
"""

from __future__ import annotations

import numpy as np


def hebbian_weights(patterns: np.ndarray) -> np.ndarray:
    """Build the Hebbian weight matrix for a set of stored patterns.

    Parameters
    ----------
    patterns : ndarray of shape (P, N), entries in {-1, +1}.

    Returns
    -------
    W : ndarray of shape (N, N), symmetric, zero diagonal.
    """
    p, n = patterns.shape
    w = (patterns.T @ patterns) / n
    np.fill_diagonal(w, 0.0)
    return w


def _sign_no_zero(x: np.ndarray) -> np.ndarray:
    """sign() that maps exact zeros to +1 (measure-zero event for h_i, but
    guards against degenerate/symmetric initial states)."""
    return np.where(x >= 0.0, 1.0, -1.0)


def run_async_dynamics_batch(
    w: np.ndarray,
    init_states: np.ndarray,
    rng: np.random.Generator,
    max_sweeps: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Run asynchronous (random sequential) Hopfield dynamics on a batch of
    independent initial states sharing the same weight matrix, to
    convergence or a max-sweep cap.

    Vectorization note: all instances in the batch share the same per-sweep
    neuron update *order* (a fresh random permutation each sweep), but each
    instance's state evolves independently since every single-neuron update
    reads only that instance's current state. This is a standard
    vectorization of independent asynchronous-dynamics trials and does not
    alter the validity of any individual trajectory.

    Parameters
    ----------
    w : (N, N) Hebbian weight matrix.
    init_states : (B, N) batch of initial +-1 states.
    rng : numpy Generator for the random update order.
    max_sweeps : cap on the number of full sweeps.

    Returns
    -------
    (final_states, n_sweeps_used) where final_states has shape (B, N) and
    n_sweeps_used is a scalar int (sweeps actually executed before either
    convergence or hitting the cap).
    """
    n = w.shape[0]
    s = init_states.copy().astype(np.float64)
    sweeps_used = 0
    for sweep in range(max_sweeps):
        order = rng.permutation(n)
        changed = False
        for i in order:
            h_i = s @ w[:, i]
            new_si = _sign_no_zero(h_i)
            if np.any(new_si != s[:, i]):
                changed = True
            s[:, i] = new_si
        sweeps_used = sweep + 1
        if not changed:
            break
    return s, sweeps_used


def overlap(state: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Normalized overlap m = (1/N) sum_i s_i xi_i, batched over the leading
    dimension if state/pattern are 2D."""
    n = state.shape[-1]
    return (state * pattern).sum(axis=-1) / n


def corrupt(pattern: np.ndarray, flip_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of `pattern` with a random `flip_frac` fraction of bits
    sign-flipped (batched: pattern may be (B, N))."""
    out = pattern.copy()
    n = out.shape[-1]
    n_flip = int(round(flip_frac * n))
    if out.ndim == 1:
        idx = rng.choice(n, size=n_flip, replace=False)
        out[idx] *= -1
    else:
        for row in out:
            idx = rng.choice(n, size=n_flip, replace=False)
            row[idx] *= -1
    return out
