"""Unfolds zeta zero heights to a unit-mean-spacing sequence.

The true zero-counting function N(T) = #{zeros with 0 < Im(rho) <= T}
decomposes as N(T) = N-bar(T) + S(T), where N-bar(T) = theta(T)/pi + 1 is
the smooth Riemann-Siegel-theta part carrying the systematic growth of the
zero density, and S(T) = O(log T) is the fluctuating part that GUE
statistics are conjectured to describe. Mapping each zero height t_n to
x_n = N-bar(t_n) removes the systematic density trend so consecutive zeros
have asymptotic unit mean spacing, making different heights T directly
comparable -- this is the standard unfolding used in Odlyzko's numerical
studies of the Montgomery-Odlyzko law.
"""

import mpmath
import numpy as np


def smooth_counting_function(t):
    """N-bar(T) = theta(T)/pi + 1 for a single height T."""
    theta = mpmath.siegeltheta(mpmath.mpf(t))
    return float(theta / mpmath.pi + 1)


def unfold_heights(heights):
    """Unfolded positions x_n = N-bar(t_n) for a sorted sequence of zero
    heights. Returns a numpy array with (asymptotically) unit mean spacing.
    """
    return np.array([smooth_counting_function(t) for t in heights], dtype=float)


def spacings(unfolded_positions):
    """Consecutive differences of an already-sorted 1D array of unfolded
    positions."""
    x = np.asarray(unfolded_positions, dtype=float)
    if len(x) < 2:
        raise ValueError("need at least 2 positions to form a spacing")
    return np.diff(x)
