"""Known asymptotic / mean-field percolation thresholds for each model.

These serve as the ground truth against which the topology-derived
detectors (percolation.py, cycle_onset.py) are validated.
"""

from __future__ import annotations

import numpy as np

# Critical mean number of neighbours for 2D continuum (Poisson Boolean
# model) percolation, from Monte-Carlo estimates in the statistical
# physics literature (e.g. Quintanilla, Torquato & Ziff, J. Phys. A 33
# (2000)). Used as the reference constant for the RGG giant-component
# threshold below.
RGG_CRITICAL_MEAN_DEGREE = 4.512


def er_giant_component_threshold(n: int) -> float:
    """Erdos-Renyi giant-component threshold p_c = 1/n (mean degree 1)."""
    return 1.0 / n


def rgg_giant_component_threshold(n: int) -> float:
    """RGG giant-component radius: n * pi * r_c^2 = critical mean degree."""
    return float(np.sqrt(RGG_CRITICAL_MEAN_DEGREE / (np.pi * n)))


def chung_lu_giant_component_threshold(weights: np.ndarray) -> float:
    """Molloy-Reed / branching-process threshold for the Chung-Lu model.

    The giant component emerges when theta * E[w^2] / E[w] = 1, i.e.
    theta_c = E[w] / E[w^2].
    """
    mean_w = weights.mean()
    mean_w2 = (weights ** 2).mean()
    return float(mean_w / mean_w2)
