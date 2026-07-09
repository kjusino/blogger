"""Experiment orchestration: sweep cost-matrix families x epsilon x seeds,
run stabilized Sinkhorn, fit the empirical asymptotic contraction rate, and
compare it against the Birkhoff/Hilbert-metric theoretical bound.

Two sweeps:

- `run_main_sweep`: a "safe" epsilon range (chosen so Delta(K)/4 stays well
  below the ~18.4 threshold where tanh saturates to exactly 1.0 in float64)
  used for the primary bound-validity and tightness analysis.
- `run_extreme_sweep`: a smaller, harder epsilon range used only to
  characterize the growth of actual iteration counts as eps -> 0 -- the
  Birkhoff bound is expected to become numerically uninformative here
  (kappa_theory saturates to 1.0), which is itself part of the finding.

Plus `run_cost_convergence_check`: confirms entropic OT cost converges to
the exact (Hungarian) OT cost as eps -> 0, on one representative instance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from .contraction import theoretical_contraction_rate
from .cost_matrices import build_cost
from .exact_ot import exact_ot_uniform
from .rate_fitting import fit_asymptotic_rate
from .sinkhorn import entropic_cost, sinkhorn_log

N_SEEDS = {
    "random_points": 6,
    "clustered_points": 6,
    "grid_1d": 1,  # deterministic family -- one canonical instance
    "iid_random": 6,
}


@dataclass
class SweepRecord:
    family: str
    seed: int
    n: int
    m: int
    eps: float
    kappa_theory: float
    rate_empirical: float
    r_squared: float
    n_iter: int
    converged: bool
    tightness: Optional[float]  # log(kappa_theory) / log(rate_empirical), in (0, 1]
    bound_violated: bool        # True iff rate_empirical > kappa_theory + tol (should never happen)


_FAMILY_SEED_BASE = {
    "random_points": 11,
    "clustered_points": 23,
    "grid_1d": 37,
    "iid_random": 47,
}


def _instance_seed(family: str, rep: int) -> int:
    # grid_1d is deterministic and ignores the rng, but every family still
    # gets a distinct, reproducible seed per repetition index. Uses a fixed
    # table rather than Python's hash() of str, which is randomized per
    # process (PYTHONHASHSEED) and would break run-to-run reproducibility.
    return 1000 * _FAMILY_SEED_BASE[family] + rep


def _make_record(family: str, seed: int, n: int, m: int, eps: float, C: np.ndarray,
                  max_iter: int, tol: float, viol_tol: float) -> SweepRecord:
    r = np.full(n, 1.0 / n)
    c = np.full(m, 1.0 / m)
    result = sinkhorn_log(C, r, c, eps=eps, max_iter=max_iter, tol=tol)
    fit = fit_asymptotic_rate(result.residual_history)
    kappa = theoretical_contraction_rate(C, eps)

    tightness = None
    if np.isfinite(fit.rate) and fit.rate > 0 and 0 < kappa < 1:
        tightness = float(np.log(kappa) / np.log(fit.rate))

    violated = bool(np.isfinite(fit.rate) and fit.rate > kappa + viol_tol)

    return SweepRecord(
        family=family, seed=seed, n=n, m=m, eps=eps,
        kappa_theory=kappa, rate_empirical=fit.rate, r_squared=fit.r_squared,
        n_iter=result.n_iter, converged=result.converged,
        tightness=tightness, bound_violated=violated,
    )


def run_main_sweep(
    n: int = 30,
    m: int = 30,
    eps_values: Optional[np.ndarray] = None,
    max_iter: int = 8000,
    tol: float = 1e-10,
    viol_tol: float = 1e-6,
) -> list[SweepRecord]:
    if eps_values is None:
        eps_values = np.geomspace(2.5, 0.1, 12)

    records: list[SweepRecord] = []
    for family, n_seeds in N_SEEDS.items():
        for rep in range(n_seeds):
            seed = _instance_seed(family, rep)
            rng = np.random.default_rng(seed)
            C = build_cost(family, n, m, rng)
            for eps in eps_values:
                records.append(_make_record(family, seed, n, m, float(eps), C, max_iter, tol, viol_tol))
    return records


def run_extreme_sweep(
    n: int = 30,
    m: int = 30,
    eps_values: Optional[np.ndarray] = None,
    max_iter: int = 5000,
    tol: float = 1e-9,
    viol_tol: float = 1e-6,
    families: tuple[str, ...] = ("random_points", "iid_random"),
    n_seeds: int = 3,
) -> list[SweepRecord]:
    if eps_values is None:
        eps_values = np.geomspace(0.08, 0.02, 5)

    records: list[SweepRecord] = []
    for family in families:
        for rep in range(n_seeds):
            seed = _instance_seed(family, rep) + 500_000  # disjoint from main sweep seeds
            rng = np.random.default_rng(seed)
            C = build_cost(family, n, m, rng)
            for eps in eps_values:
                records.append(_make_record(family, seed, n, m, float(eps), C, max_iter, tol, viol_tol))
    return records


@dataclass
class CostConvergencePoint:
    eps: float
    entropic_cost: float
    exact_cost: float
    gap: float


def run_cost_convergence_check(
    n: int = 30,
    seed: int = 7,
    eps_values: Optional[np.ndarray] = None,
    max_iter: int = 20_000,
    tol: float = 1e-11,
) -> list[CostConvergencePoint]:
    if eps_values is None:
        eps_values = np.geomspace(1.0, 0.02, 10)

    rng = np.random.default_rng(seed)
    C = build_cost("random_points", n, n, rng)
    exact = exact_ot_uniform(C)
    r = c = np.full(n, 1.0 / n)

    points = []
    for eps in eps_values:
        result = sinkhorn_log(C, r, c, eps=float(eps), max_iter=max_iter, tol=tol)
        cost = entropic_cost(result.plan, C)
        points.append(CostConvergencePoint(eps=float(eps), entropic_cost=cost, exact_cost=exact, gap=cost - exact))
    return points


def records_to_dicts(records: list[SweepRecord]) -> list[dict]:
    return [asdict(r) for r in records]
