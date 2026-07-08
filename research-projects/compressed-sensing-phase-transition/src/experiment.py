"""Grid-sweep experiment: empirical basis-pursuit recovery probability over
a (delta, rho) grid, at several ambient dimensions n, compared against the
theoretical phase-transition curve."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .recovery import check_recovery, solve_basis_pursuit
from .sensing import gaussian_sensing_matrix, sparse_signal
from .theory import phase_transition_delta


@dataclass
class GridPoint:
    n: int
    delta: float
    rho: float
    m: int
    k: int
    success_rate: float
    trials: int


@dataclass
class ExperimentResult:
    n_values: list[int]
    delta_grid: np.ndarray
    rho_grid: np.ndarray
    points: list[GridPoint] = field(default_factory=list)

    def success_matrix(self, n: int) -> np.ndarray:
        """rho-by-delta matrix of success rates for a fixed n."""
        mat = np.full((len(self.rho_grid), len(self.delta_grid)), np.nan)
        for p in self.points:
            if p.n != n:
                continue
            i = int(np.argmin(np.abs(self.rho_grid - p.rho)))
            j = int(np.argmin(np.abs(self.delta_grid - p.delta)))
            mat[i, j] = p.success_rate
        return mat


def run_trial(n: int, m: int, k: int, rng: np.random.Generator) -> bool:
    """One recovery trial: random Gaussian A, random k-sparse x0, solve
    basis pursuit, check exact recovery."""
    A = gaussian_sensing_matrix(m, n, rng)
    x0 = sparse_signal(n, k, rng)
    y = A @ x0
    x_hat = solve_basis_pursuit(A, y)
    return check_recovery(x_hat, x0)


def success_probability(n: int, delta: float, rho: float, trials: int,
                         rng: np.random.Generator) -> tuple[float, int, int]:
    """Empirical recovery probability at a single (n, delta, rho) grid point.

    m and k are each rounded to at least 1 (delta=0 or rho=0 grid points are
    not swept in practice, but this keeps the function well-defined) and k
    is capped at m (l1-minimization needs m >= k to have any chance)."""
    m = max(1, int(round(delta * n)))
    k = max(1, min(m, int(round(rho * n))))
    successes = sum(run_trial(n, m, k, rng) for _ in range(trials))
    return successes / trials, m, k


def run_grid_sweep(n_values: list[int], delta_grid: np.ndarray, rho_grid: np.ndarray,
                    trials: int, seed: int = 0) -> ExperimentResult:
    """Full sweep: for every (n, delta, rho) combination, estimate the
    empirical recovery probability from `trials` independent instances."""
    rng = np.random.default_rng(seed)
    result = ExperimentResult(n_values=list(n_values), delta_grid=np.asarray(delta_grid),
                               rho_grid=np.asarray(rho_grid))
    for n in n_values:
        for rho in rho_grid:
            for delta in delta_grid:
                rate, m, k = success_probability(n, delta, rho, trials, rng)
                result.points.append(GridPoint(n=n, delta=delta, rho=rho, m=m, k=k,
                                                success_rate=rate, trials=trials))
    return result


def empirical_threshold(delta_grid: np.ndarray, success_rates: np.ndarray) -> float | None:
    """Estimate the delta at which success probability crosses 0.5 by linear
    interpolation on the empirical curve. Returns None if the curve never
    crosses 0.5 within the grid."""
    order = np.argsort(delta_grid)
    d = np.asarray(delta_grid)[order]
    s = np.asarray(success_rates)[order]
    if s[0] >= 0.5:
        return float(d[0])
    if s[-1] < 0.5:
        return None
    for i in range(len(d) - 1):
        if s[i] < 0.5 <= s[i + 1]:
            if s[i + 1] == s[i]:
                return float(d[i])
            frac = (0.5 - s[i]) / (s[i + 1] - s[i])
            return float(d[i] + frac * (d[i + 1] - d[i]))
    return None


def transition_width(delta_grid: np.ndarray, success_rates: np.ndarray,
                      lo: float = 0.1, hi: float = 0.9) -> float | None:
    """Width of the empirical transition band: the delta-distance between
    the `lo` and `hi` success-probability crossings. Smaller width means a
    sharper (more theory-consistent) transition. Returns None if either
    crossing is not bracketed by the grid."""
    order = np.argsort(delta_grid)
    d = np.asarray(delta_grid)[order]
    s = np.asarray(success_rates)[order]

    def crossing(level: float) -> float | None:
        if s[0] >= level:
            return float(d[0])
        if s[-1] < level:
            return None
        for i in range(len(d) - 1):
            if s[i] < level <= s[i + 1]:
                if s[i + 1] == s[i]:
                    return float(d[i])
                frac = (level - s[i]) / (s[i + 1] - s[i])
                return float(d[i] + frac * (d[i + 1] - d[i]))
        return None

    d_lo = crossing(lo)
    d_hi = crossing(hi)
    if d_lo is None or d_hi is None:
        return None
    return d_hi - d_lo


def theory_rmse(result: ExperimentResult, n: int) -> float:
    """RMSE between the empirical 50%-crossing threshold (one per rho row)
    and the ALMT-predicted phase_transition_delta(rho), for a fixed n."""
    mat = result.success_matrix(n)
    errors = []
    for i, rho in enumerate(result.rho_grid):
        emp = empirical_threshold(result.delta_grid, mat[i])
        if emp is None:
            continue
        errors.append((emp - phase_transition_delta(rho)) ** 2)
    if not errors:
        return float("nan")
    return float(np.sqrt(np.mean(errors)))
