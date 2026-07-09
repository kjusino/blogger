"""Closed-form approximation-ratio guarantees for greedy submodular
maximization under a cardinality constraint.

Worst-case bound: Nemhauser, Wolsey & Fisher (1978), Math. Programming.
Curvature-refined bound: Conforti & Cornuejols (1984), Discrete Applied
Math.; see also Vondrak (2010), "Submodularity and curvature: the optimal
algorithm."
"""

import math

WORST_CASE_BOUND = 1.0 - 1.0 / math.e  # ~0.6321


def curvature_bound(c):
    """(1 - e^{-c}) / c, with the removable singularity at c=0 taken as 1."""
    if c <= 1e-9:
        return 1.0
    return (1.0 - math.exp(-c)) / c
