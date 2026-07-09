"""Independent bit-flip (X) noise model."""

from __future__ import annotations

import numpy as np


def sample_errors(L: int, p: float, rng: np.random.Generator):
    """Sample i.i.d. X errors with probability p on every edge qubit.

    Returns (h_err, v_err), each an L x L uint8 array.
    """
    h_err = (rng.random((L, L)) < p).astype(np.uint8)
    v_err = (rng.random((L, L)) < p).astype(np.uint8)
    return h_err, v_err
