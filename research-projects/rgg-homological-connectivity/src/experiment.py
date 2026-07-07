"""Experiment driver: connectivity-threshold estimation (exact, via the
longest edge of the periodic Euclidean minimum spanning tree) and an
exploratory Betti_1 sweep illustrating the "homological connectivity
window" above the graph-connectivity threshold.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from .betti import betti_numbers
from .point_cloud import sample_torus_points
from .simplicial_complex import build_complex
from .theory import penrose_threshold_radius, threshold_ratio


def periodic_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Dense n x n minimum-image periodic distance matrix on the unit torus."""
    diff = points[:, None, :] - points[None, :, :]
    diff -= np.round(diff)
    return np.sqrt((diff ** 2).sum(-1))


def exact_connectivity_threshold(points: np.ndarray) -> float:
    """The graph is connected at radius r iff r >= the longest MST edge.

    This is *exact* (no bisection tolerance): the connectivity threshold of
    a Euclidean/periodic point set is precisely the longest edge of its
    minimum spanning tree -- the classical fact Penrose's threshold theorem
    is built on top of.
    """
    n = points.shape[0]
    if n <= 1:
        return 0.0
    dist = periodic_distance_matrix(points)
    mst = minimum_spanning_tree(dist)
    return float(mst.data.max()) if mst.nnz else 0.0


def sample_thresholds(n: int, trials: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `trials` independent point sets of size n and return their exact
    connectivity thresholds (longest MST edge each)."""
    return np.array([
        exact_connectivity_threshold(sample_torus_points(n, rng))
        for _ in range(trials)
    ])


@dataclass
class ThresholdResult:
    n: int
    thresholds: np.ndarray
    theory_r_c: float


def run_threshold_experiment(
    n_values: List[int], trials_per_n: int, rng: np.random.Generator
) -> Dict[int, ThresholdResult]:
    results = {}
    for n in n_values:
        thresholds = sample_thresholds(n, trials_per_n, rng)
        results[n] = ThresholdResult(n=n, thresholds=thresholds, theory_r_c=penrose_threshold_radius(n))
    return results


def connected_fraction_curve(thresholds: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
    """Empirical P(connected at radius r) = fraction of trials whose exact
    threshold is <= r, for each r in r_grid. Exact (no re-simulation needed)."""
    return np.array([(thresholds <= r).mean() for r in r_grid])


@dataclass
class Betti1CurveResult:
    n: int
    r_grid: np.ndarray
    mean_betti0: np.ndarray
    mean_betti1: np.ndarray
    frac_connected: np.ndarray


def run_betti1_curve(
    n: int, r_grid: np.ndarray, trials: int, rng: np.random.Generator
) -> Betti1CurveResult:
    """Average Betti_0 and Betti_1 over `trials` point sets, at every radius
    in r_grid. This is the expensive path (needs the full 2-skeleton and a
    GF(2) rank per (trial, radius) pair) so callers should keep `trials` and
    `len(r_grid)` modest.
    """
    betti0_acc = np.zeros(len(r_grid))
    betti1_acc = np.zeros(len(r_grid))
    connected_acc = np.zeros(len(r_grid))
    for _ in range(trials):
        points = sample_torus_points(n, rng)
        for i, r in enumerate(r_grid):
            complex_ = build_complex(points, float(r))
            b0, b1 = betti_numbers(complex_)
            betti0_acc[i] += b0
            betti1_acc[i] += b1
            connected_acc[i] += 1.0 if b0 == 1 else 0.0
    return Betti1CurveResult(
        n=n,
        r_grid=r_grid,
        mean_betti0=betti0_acc / trials,
        mean_betti1=betti1_acc / trials,
        frac_connected=connected_acc / trials,
    )


def ratio_summary(result: ThresholdResult) -> Dict[str, float]:
    """ratio(n) = mean(r_empirical) / r_c(n); theory predicts this -> 1."""
    n = result.n
    mean_r = float(result.thresholds.mean())
    return {
        "n": n,
        "mean_threshold": mean_r,
        "theory_r_c": result.theory_r_c,
        "ratio": threshold_ratio(mean_r, n),
    }
