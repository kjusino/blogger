"""Replica message-latency models.

Two replica populations are supported:

- Homogeneous: every replica's latency is drawn i.i.d. from the same
  distribution. This is the assumption baked into the classical WARS
  (Write, Replica-ack, Read, Staleness) model of Bailis et al., "Probabilistically
  Bounded Staleness for Practical Partial Quorums" (VLDB 2012).
- Heterogeneous: each replica gets a fixed per-replica speed multiplier drawn
  once (representing e.g. a permanently-slower rack, AZ, or WAN link), so
  latencies are still independent across trials but no longer identically
  distributed across replicas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SUPPORTED_FAMILIES = ("exponential", "lognormal")


def _validate_family(family: str) -> None:
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unknown latency family {family!r}; expected one of {SUPPORTED_FAMILIES}")


@dataclass(frozen=True)
class ReplicaLatencyModel:
    """Samples an (n_trials, n_replicas) matrix of message latencies.

    Parameters
    ----------
    n_replicas: number of replicas N.
    family: "exponential" or "lognormal".
    rate: for exponential, the rate parameter (mean latency = 1/rate).
    sigma: for lognormal, the shape parameter of the underlying normal (mu is
        chosen so the marginal mean latency equals 1/rate, for comparability
        across families).
    multipliers: per-replica positive speed multipliers, length n_replicas.
        A multiplier > 1 means that replica is slower than baseline. All ones
        recovers the homogeneous, i.i.d. case assumed by the classical model.
    """

    n_replicas: int
    family: str
    rate: float
    sigma: float
    multipliers: np.ndarray

    def __post_init__(self) -> None:
        _validate_family(self.family)
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        if len(self.multipliers) != self.n_replicas:
            raise ValueError("multipliers must have length n_replicas")
        if np.any(self.multipliers <= 0):
            raise ValueError("multipliers must be strictly positive")

    def sample(self, n_trials: int, rng: np.random.Generator) -> np.ndarray:
        """Return an (n_trials, n_replicas) matrix of latencies."""
        if self.family == "exponential":
            base = rng.exponential(scale=1.0 / self.rate, size=(n_trials, self.n_replicas))
        else:  # lognormal, mean-matched to 1/rate
            target_mean = 1.0 / self.rate
            mu = np.log(target_mean) - 0.5 * self.sigma**2
            base = rng.lognormal(mean=mu, sigma=self.sigma, size=(n_trials, self.n_replicas))
        return base * self.multipliers[np.newaxis, :]


def homogeneous_model(n_replicas: int, family: str, rate: float, sigma: float = 1.0) -> ReplicaLatencyModel:
    """Every replica draws latency i.i.d. from the same distribution."""
    return ReplicaLatencyModel(
        n_replicas=n_replicas,
        family=family,
        rate=rate,
        sigma=sigma,
        multipliers=np.ones(n_replicas),
    )


def heterogeneous_model(
    n_replicas: int,
    family: str,
    rate: float,
    sigma_het: float,
    rng: np.random.Generator,
    sigma: float = 1.0,
) -> ReplicaLatencyModel:
    """Assign each replica a fixed lognormal speed multiplier, once.

    sigma_het=0 degenerates to the homogeneous case (all multipliers == 1).
    Multipliers are drawn from LogNormal(0, sigma_het) so their geometric
    mean is 1 (no shift in overall average speed, only added spread/skew).
    """
    if sigma_het < 0:
        raise ValueError("sigma_het must be non-negative")
    if sigma_het == 0:
        multipliers = np.ones(n_replicas)
    else:
        multipliers = rng.lognormal(mean=0.0, sigma=sigma_het, size=n_replicas)
    return ReplicaLatencyModel(
        n_replicas=n_replicas,
        family=family,
        rate=rate,
        sigma=sigma,
        multipliers=multipliers,
    )
