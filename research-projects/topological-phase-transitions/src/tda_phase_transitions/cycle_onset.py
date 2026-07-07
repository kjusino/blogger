"""Detect the onset of the first-homology (cycle) population.

Near the percolation critical point, a sparse random graph transitions
from being essentially tree-like (no cycles) to containing an extensive
number of independent cycles. We locate this "cycle onset" from the H1
persistence diagram: the birth times of 1-dimensional homology classes
(loops that are not yet filled in by a 2-simplex).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def cycle_onset_threshold(
    dgm1: np.ndarray,
    birth_quantile: float = 0.1,
    min_persistence: float = 0.0,
) -> Optional[float]:
    """Threshold below which only ``birth_quantile`` of significant cycles
    have been born.

    ``min_persistence`` filters out short-lived bars often attributed to
    sampling noise. Returns ``None`` if no cycles survive the filter
    (e.g. the filtration range never reached the cyclic regime).
    """
    if dgm1.size == 0:
        return None
    births = dgm1[:, 0]
    deaths = dgm1[:, 1]
    persistence = deaths - births
    significant = births[persistence >= min_persistence]
    if significant.size == 0:
        return None
    return float(np.quantile(significant, birth_quantile))


def cycle_birth_rate(dgm1: np.ndarray, thresholds) -> np.ndarray:
    """Cumulative count of H1 births at or before each threshold."""
    thresholds = np.asarray(thresholds, dtype=float)
    if dgm1.size == 0:
        return np.zeros(len(thresholds))
    births = np.sort(dgm1[:, 0])
    return np.searchsorted(births, thresholds, side="right").astype(float)
