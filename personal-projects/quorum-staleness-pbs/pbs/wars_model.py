"""The classical WARS Monte Carlo staleness model (Bailis et al., VLDB 2012).

The WARS model treats replicas as i.i.d. and exchangeable: every replica's
message latency is an independent draw from a single shared distribution,
and a read quorum is a uniformly random subset of the N replicas. Under
those assumptions there is (in general) no closed form for P(stale | delta),
so the original paper itself evaluates the model by Monte Carlo sampling of
the latency random variables directly (not via a full discrete-event system
simulation). We reproduce exactly that: this module is the "theory" side of
the comparison, and ``pbs.simulator`` is the "empirical system" side.
"""

from __future__ import annotations

import numpy as np

from pbs.latency import homogeneous_model
from pbs.quorum import ack_time, is_stale_random


def wars_staleness_curve(
    n_replicas: int,
    w: int,
    r: int,
    family: str,
    rate: float,
    deltas: np.ndarray,
    n_trials: int,
    rng: np.random.Generator,
    sigma: float = 1.0,
) -> np.ndarray:
    """Return P(stale | delta) for each delta in ``deltas``, under WARS assumptions.

    Latencies are drawn once (an (n_trials, n_replicas) matrix) and reused
    across every delta value, since the ack time A does not depend on delta;
    only the read threshold A + delta changes. The read quorum is re-drawn
    independently, per trial, for every delta value (matching the paper's
    treatment of reads as independent operations).
    """
    model = homogeneous_model(n_replicas, family=family, rate=rate, sigma=sigma)
    latencies = model.sample(n_trials, rng)
    ack = ack_time(latencies, w)

    curve = np.empty(len(deltas))
    for i, delta in enumerate(deltas):
        read_time = ack + delta
        stale = is_stale_random(latencies, read_time, r, rng)
        curve[i] = stale.mean()
    return curve
