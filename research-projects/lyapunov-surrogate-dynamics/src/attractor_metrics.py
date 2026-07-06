"""A simple, deterministic attractor-shape comparison metric: per-coordinate
marginal histograms (fixed bins over a fixed range) of a long true
trajectory vs. a long surrogate-iterated trajectory, compared via the
Jensen-Shannon divergence, averaged across coordinates.

Kept deliberately simple: this is not meant to be a state-of-the-art
attractor-reconstruction metric, just a fast, well-defined, reproducible
summary of "do these two point clouds occupy the same region of state
space with similar density," which is enough to demonstrate qualitative
attractor collapse/blow-up in surrogates that no longer preserve the
system's ergodic statistics.
"""
from __future__ import annotations

import numpy as np

# Fixed bin ranges spanning the classical Lorenz-63 (sigma=10, rho=28,
# beta=8/3) attractor comfortably, with margin for the divergence metric to
# be able to actually detect surrogates that overshoot the true attractor.
DEFAULT_RANGES = np.array([
    [-25.0, 25.0],   # x
    [-30.0, 30.0],   # y
    [0.0, 55.0],     # z
])
DEFAULT_BINS = 60


def _histogram_prob(values: np.ndarray, value_range, bins: int) -> np.ndarray:
    """Fixed-range, fixed-bin-count histogram normalized to a probability
    distribution (adds a tiny epsilon floor so JS divergence stays finite
    even when a bin is empty in one distribution but not the other).
    """
    counts, _ = np.histogram(values, bins=bins, range=value_range)
    probs = counts.astype(float)
    probs += 1e-12
    probs /= probs.sum()
    return probs


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JS divergence in nats, base-e, symmetric, bounded in [0, ln(2)]."""
    m = 0.5 * (p + q)

    def kl(a, b):
        return np.sum(a * np.log(a / b))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def marginal_js_divergence(traj_true: np.ndarray, traj_surrogate: np.ndarray,
                            ranges=DEFAULT_RANGES, bins: int = DEFAULT_BINS
                            ) -> dict:
    """Compute per-coordinate JS divergence between the marginal
    distributions of two trajectories (each shape (N, dim)), plus their
    mean across coordinates as a single scalar summary.

    Returns a dict: {'per_coord': array of length dim, 'mean': float}.
    """
    dim = traj_true.shape[1]
    divergences = np.empty(dim)
    for d in range(dim):
        p = _histogram_prob(traj_true[:, d], ranges[d], bins)
        q = _histogram_prob(traj_surrogate[:, d], ranges[d], bins)
        divergences[d] = _jensen_shannon_divergence(p, q)
    return {"per_coord": divergences, "mean": float(divergences.mean())}
