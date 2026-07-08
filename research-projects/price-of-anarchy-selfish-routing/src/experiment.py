"""The two experiments this project runs:

1. `run_topology_battery` -- generate many random series-parallel networks
   per polynomial degree p and check that the empirical price of anarchy
   never exceeds the theoretical bound beta(p), regardless of how large or
   convoluted the topology is (the *upper-bound* half of topology
   independence).

2. `run_worst_case_convergence` -- sweep the constant `b` on the simplest
   possible two-link network for each degree p, run it through the exact
   same generic solver (not the closed-form shortcut), and show the
   achieved PoA peaks essentially at beta(p) (the *tightness* half of
   topology independence: the simplest topology already realizes the
   worst case, so more complex topologies cannot do better/worse).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.network import random_series_parallel, two_link_pigou_style
from src.solvers import price_of_anarchy
from src.theory import poa_bound


@dataclass
class TopologyTrial:
    degree: int
    n_edges: int
    poa: float


@dataclass
class BatteryResult:
    degree: int
    bound: float
    trials: list = field(default_factory=list)

    @property
    def poas(self) -> np.ndarray:
        return np.array([t.poa for t in self.trials])

    @property
    def max_poa(self) -> float:
        return float(self.poas.max())

    @property
    def n_violations(self) -> int:
        return int(np.sum(self.poas > self.bound + 1e-3))


def run_topology_battery(degrees, n_trials: int, max_edges: int, seed: int) -> dict:
    """For each degree, generate `n_trials` random series-parallel networks
    and record the empirical PoA of each against the theoretical bound."""
    rng = np.random.default_rng(seed)
    results = {}
    for p in degrees:
        battery = BatteryResult(degree=p, bound=poa_bound(p))
        for _ in range(n_trials):
            n_edges = int(rng.integers(3, max_edges + 1))
            net = random_series_parallel(rng, degree=p, max_edges=n_edges)
            r = price_of_anarchy(net)
            battery.trials.append(TopologyTrial(degree=p, n_edges=net.n_edges, poa=r["poa"]))
        results[p] = battery
    return results


@dataclass
class ConvergencePoint:
    b: float
    poa: float


def run_worst_case_convergence(degrees, b_grid: np.ndarray) -> dict:
    """For each degree, sweep b on the two-link network through the *generic*
    solver and record PoA(b) -- independent of the closed-form theory.py
    shortcut, this exercises the same solver code path as the random
    topology battery above."""
    results = {}
    for p in degrees:
        points = []
        for b in b_grid:
            net = two_link_pigou_style(p, float(b))
            r = price_of_anarchy(net)
            points.append(ConvergencePoint(b=float(b), poa=r["poa"]))
        results[p] = points
    return results
