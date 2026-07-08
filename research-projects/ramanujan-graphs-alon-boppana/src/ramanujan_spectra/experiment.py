"""The main sweep: sample many random d-regular graphs across a grid of
(d, n), compute lambda(G) for each, and check both theoretical predictions
from theory.py against the empirical distribution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .graphs import is_connected, networkx_regular_graph, pairing_model_regular_graph
from .spectrum import extremal_eigenvalues
from .theory import alon_boppana_bound, close_to_bound, within_epsilon_of_bound

DEGREES = (3, 4, 6, 10)
N_GRID = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
EPSILONS = (0.05, 0.1, 0.25, 0.5)
BASE_SEED = 20260707


def trials_for_n(n: int) -> int:
    """Fewer trials at large n to keep the sweep's wall-clock bounded --
    the dominant cost is the eigensolver, whose per-graph cost grows with n."""
    if n <= 512:
        return 40
    if n <= 2048:
        return 20
    return 10


def make_rng(d: int, n: int, trial: int, base_seed: int = BASE_SEED) -> np.random.Generator:
    seq = np.random.SeedSequence([base_seed, d, n, trial])
    return np.random.default_rng(seq)


@dataclass(frozen=True)
class TrialResult:
    d: int
    n: int
    trial: int
    lambda1: float
    lambda2: float
    lambda_min: float
    lambda2_abs: float
    bipartite_like: bool
    connected: bool
    bound: float
    gap: float  # lambda2_abs - bound; theory predicts this -> 0+ as n grows


def run_single_trial(d: int, n: int, trial: int, base_seed: int = BASE_SEED) -> TrialResult:
    """Sample one random d-regular graph and compute its spectral summary.

    Uses networkx's generator (fast for every d used here, including d=10,
    where the from-scratch full-restart pairing model in graphs.py becomes
    intractable -- see graphs.py's module docstring and
    ``compare_generators`` below, which cross-checks the two generators
    against each other at the degrees where both are tractable)."""
    rng = make_rng(d, n, trial, base_seed)
    seed = int(rng.integers(0, 2**31 - 1))
    adj = networkx_regular_graph(d, n, seed)
    spec = extremal_eigenvalues(adj, d)
    connected = is_connected(adj)
    bound = alon_boppana_bound(d)
    return TrialResult(
        d=d,
        n=n,
        trial=trial,
        lambda1=spec.lambda1,
        lambda2=spec.lambda2,
        lambda_min=spec.lambda_min,
        lambda2_abs=spec.lambda2_abs,
        bipartite_like=spec.bipartite_like,
        connected=connected,
        bound=bound,
        gap=spec.lambda2_abs - bound,
    )


def run_sweep(
    degrees=DEGREES, n_grid=N_GRID, base_seed: int = BASE_SEED, trials_fn=trials_for_n
) -> list[TrialResult]:
    results = []
    for d in degrees:
        for n in n_grid:
            for trial in range(trials_fn(n)):
                results.append(run_single_trial(d, n, trial, base_seed))
    return results


@dataclass(frozen=True)
class CellSummary:
    d: int
    n: int
    bound: float
    trials: int
    lambda2_abs_mean: float
    lambda2_abs_std: float
    lambda2_abs_min: float
    lambda2_abs_max: float
    gap_mean: float
    gap_min: float
    gap_max: float
    frac_connected: float
    frac_bipartite_like: float
    frac_exceeds_bound: float
    frac_within_eps: dict
    frac_close_to_bound: dict


def summarize_cell(rows: list[TrialResult]) -> CellSummary:
    d, n = rows[0].d, rows[0].n
    vals = np.array([r.lambda2_abs for r in rows])
    gaps = np.array([r.gap for r in rows])
    bound = rows[0].bound
    frac_within_eps = {
        f"eps_{eps}": float(np.mean([within_epsilon_of_bound(r.lambda2_abs, d, eps) for r in rows]))
        for eps in EPSILONS
    }
    frac_close_to_bound = {
        f"eps_{eps}": float(np.mean([close_to_bound(r.lambda2_abs, d, eps) for r in rows]))
        for eps in EPSILONS
    }
    return CellSummary(
        d=d,
        n=n,
        bound=bound,
        trials=len(rows),
        lambda2_abs_mean=float(vals.mean()),
        lambda2_abs_std=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        lambda2_abs_min=float(vals.min()),
        lambda2_abs_max=float(vals.max()),
        gap_mean=float(gaps.mean()),
        gap_min=float(gaps.min()),
        gap_max=float(gaps.max()),
        frac_connected=float(np.mean([r.connected for r in rows])),
        frac_bipartite_like=float(np.mean([r.bipartite_like for r in rows])),
        frac_exceeds_bound=float(np.mean(gaps > 1e-9)),
        frac_within_eps=frac_within_eps,
        frac_close_to_bound=frac_close_to_bound,
    )


def summarize_sweep(results: list[TrialResult]) -> list[CellSummary]:
    by_cell: dict[tuple[int, int], list[TrialResult]] = {}
    for r in results:
        by_cell.setdefault((r.d, r.n), []).append(r)
    cells = [summarize_cell(rows) for rows in by_cell.values()]
    cells.sort(key=lambda c: (c.d, c.n))
    return cells


def fit_gap_power_law(cells: list[CellSummary], d: int) -> dict:
    """Fit |gap_mean| ~ C * n^alpha via least squares on log|gap| vs log(n),
    for a fixed degree d. mean lambda(G) approaches the Alon-Boppana bound
    from *below* at every (d, n) tested here (see results/summary.csv --
    gap_mean is negative throughout), so we fit the magnitude of the gap,
    which is the quantity theory predicts shrinks to 0. Returns the fitted
    exponent alpha, its intercept, and the R^2 of the log-log fit. Points
    where the mean gap is (numerically) exactly zero are dropped since log
    is undefined there."""
    d_cells = [c for c in cells if c.d == d and abs(c.gap_mean) > 0]
    if len(d_cells) < 2:
        return {"alpha": None, "intercept": None, "r_squared": None, "n_points": len(d_cells)}

    log_n = np.log(np.array([c.n for c in d_cells], dtype=float))
    log_gap = np.log(np.abs(np.array([c.gap_mean for c in d_cells], dtype=float)))
    alpha, intercept = np.polyfit(log_n, log_gap, 1)

    pred = alpha * log_n + intercept
    ss_res = float(np.sum((log_gap - pred) ** 2))
    ss_tot = float(np.sum((log_gap - log_gap.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "alpha": float(alpha),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "n_points": len(d_cells),
    }


GENERATOR_VALIDATION_DEGREES = (3, 4)
GENERATOR_VALIDATION_N = (100, 500)
GENERATOR_VALIDATION_TRIALS = 150


def compare_generators(
    d: int,
    n: int,
    trials: int = GENERATOR_VALIDATION_TRIALS,
    base_seed: int = BASE_SEED,
) -> dict:
    """Cross-validate the from-scratch pairing-model generator against
    networkx's generator: sample `trials` independent d-regular graphs from
    each, and check that their lambda(G) distributions agree. Only run at
    degrees where the pairing model's full-restart rejection sampling is
    still fast (see graphs.py's module docstring)."""
    pairing_vals = []
    nx_vals = []
    for trial in range(trials):
        rng_a = make_rng(d, n, trial, base_seed=base_seed ^ 0xA5A5)
        adj_a = pairing_model_regular_graph(d, n, rng_a)
        pairing_vals.append(extremal_eigenvalues(adj_a, d).lambda2_abs)

        rng_b = make_rng(d, n, trial, base_seed=base_seed ^ 0x5A5A)
        seed_b = int(rng_b.integers(0, 2**31 - 1))
        adj_b = networkx_regular_graph(d, n, seed_b)
        nx_vals.append(extremal_eigenvalues(adj_b, d).lambda2_abs)

    pairing_vals = np.array(pairing_vals)
    nx_vals = np.array(nx_vals)
    mean_diff = float(pairing_vals.mean() - nx_vals.mean())
    pooled_se = float(
        np.sqrt(pairing_vals.var(ddof=1) / trials + nx_vals.var(ddof=1) / trials)
    )

    return {
        "d": d,
        "n": n,
        "trials": trials,
        "pairing_mean": float(pairing_vals.mean()),
        "pairing_std": float(pairing_vals.std(ddof=1)),
        "networkx_mean": float(nx_vals.mean()),
        "networkx_std": float(nx_vals.std(ddof=1)),
        "mean_diff": mean_diff,
        "pooled_standard_error": pooled_se,
        "diff_in_standard_errors": mean_diff / pooled_se if pooled_se > 0 else float("nan"),
    }


def trial_result_to_row(r: TrialResult) -> dict:
    row = asdict(r)
    return row


def cell_summary_to_row(c: CellSummary) -> dict:
    row = asdict(c)
    within_dict = row.pop("frac_within_eps")
    row.update({f"within_{k}": v for k, v in within_dict.items()})
    close_dict = row.pop("frac_close_to_bound")
    row.update({f"close_{k}": v for k, v in close_dict.items()})
    return row
