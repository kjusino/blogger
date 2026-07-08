"""The "empirical system" side of the comparison.

Structurally this uses the same write/ack/read mechanics as
``pbs.wars_model``, but allows the replica population to be heterogeneous
(each replica has a fixed, persistent speed multiplier) and the read quorum
to be a *fixed* subset rather than resampled uniformly at random every time
-- e.g. a client that is always routed to the same nearby replicas. Both
of these are realistic properties of deployed quorum systems that the
classical WARS model abstracts away by assuming i.i.d., exchangeable
replicas.

Setting sigma_het=0 and selection="random" recovers exactly the WARS
model's assumptions, and is used as a validation baseline: the two modules
should then agree up to Monte Carlo noise (see tests and
experiments/run_experiments.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pbs.latency import heterogeneous_model
from pbs.quorum import ack_time, is_stale_fixed, is_stale_random, slowest_fixed_subset


@dataclass(frozen=True)
class SimulationResult:
    curve: np.ndarray
    multipliers: np.ndarray
    fixed_idx: np.ndarray | None


def simulate_staleness_curve(
    n_replicas: int,
    w: int,
    r: int,
    family: str,
    rate: float,
    deltas: np.ndarray,
    n_trials: int,
    rng: np.random.Generator,
    sigma_het: float = 0.0,
    selection: str = "random",
    sigma: float = 1.0,
) -> SimulationResult:
    """Return P(stale | delta) for each delta, from the simulated system.

    selection: "random" (fresh uniformly random read quorum per trial, per
        delta) or "fixed" (read quorum is the r slowest replicas, held fixed
        across every trial and every delta -- a sticky/regional client).
    """
    if selection not in ("random", "fixed"):
        raise ValueError(f"selection must be 'random' or 'fixed', got {selection!r}")

    model = heterogeneous_model(n_replicas, family=family, rate=rate, sigma_het=sigma_het, rng=rng, sigma=sigma)
    latencies = model.sample(n_trials, rng)
    ack = ack_time(latencies, w)

    fixed_idx = None
    if selection == "fixed":
        fixed_idx = slowest_fixed_subset(model.multipliers, r)

    curve = np.empty(len(deltas))
    for i, delta in enumerate(deltas):
        read_time = ack + delta
        if selection == "fixed":
            stale = is_stale_fixed(latencies, read_time, fixed_idx)
        else:
            stale = is_stale_random(latencies, read_time, r, rng)
        curve[i] = stale.mean()

    return SimulationResult(curve=curve, multipliers=model.multipliers, fixed_idx=fixed_idx)
