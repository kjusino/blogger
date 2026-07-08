"""Quorum mechanics: commit (ack) time and read-staleness indicators.

Protocol modeled, per trial:

1. A client sends a write to all N replicas simultaneously at time 0.
2. Replica i applies the write after latency L_i (drawn from a
   ``ReplicaLatencyModel``).
3. The write is considered committed/acked once W replicas have applied it,
   i.e. at time A = the W-th order statistic of (L_1, ..., L_N).
4. Some time later, at t = A + delta (delta >= 0, "visibility delay" since
   commit), the client issues a read to a quorum of R replicas.
5. The read is *stale* iff none of those R replicas had applied the write by
   time t, i.e. iff L_i > t for every replica i in the read set.

Two read-quorum selection policies are supported:

- "random": the R replicas are chosen uniformly at random (without
  replacement) from all N, independently for every trial. This is the
  implicit assumption of the classical WARS model (replicas are
  exchangeable).
- "fixed": the R replicas are a single, pre-determined subset used on every
  trial (e.g. a client "stuck" reading from the same nearby replicas). This
  models sticky/regional routing and breaks the exchangeability assumption
  when replica speeds are heterogeneous.
"""

from __future__ import annotations

import numpy as np


def ack_time(latencies: np.ndarray, w: int) -> np.ndarray:
    """W-th order statistic (ack/commit time) for each trial (row)."""
    n_replicas = latencies.shape[1]
    if not (1 <= w <= n_replicas):
        raise ValueError(f"w={w} must be in [1, {n_replicas}]")
    partitioned = np.partition(latencies, w - 1, axis=1)
    return partitioned[:, w - 1]


def slowest_fixed_subset(multipliers: np.ndarray, r: int) -> np.ndarray:
    """Indices of the r replicas with the largest (slowest) multiplier."""
    n_replicas = len(multipliers)
    if not (1 <= r <= n_replicas):
        raise ValueError(f"r={r} must be in [1, {n_replicas}]")
    order = np.argsort(-multipliers, kind="stable")
    return order[:r]


def is_stale_fixed(latencies: np.ndarray, read_time: np.ndarray, fixed_idx: np.ndarray) -> np.ndarray:
    """Boolean staleness indicator per trial, reading a fixed replica subset."""
    subset = latencies[:, fixed_idx]
    return np.all(subset > read_time[:, np.newaxis], axis=1)


def is_stale_random(
    latencies: np.ndarray,
    read_time: np.ndarray,
    r: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Boolean staleness indicator per trial, reading a fresh random subset each trial."""
    n_trials, n_replicas = latencies.shape
    if not (1 <= r <= n_replicas):
        raise ValueError(f"r={r} must be in [1, {n_replicas}]")
    # argsort of per-row random keys gives an independent random permutation per row;
    # the first r columns are then a uniformly random size-r subset without replacement.
    random_keys = rng.random((n_trials, n_replicas))
    chosen = np.argsort(random_keys, axis=1)[:, :r]
    subset = np.take_along_axis(latencies, chosen, axis=1)
    return np.all(subset > read_time[:, np.newaxis], axis=1)
