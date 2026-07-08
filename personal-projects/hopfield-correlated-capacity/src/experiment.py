"""Sweep runner: for each (N, rho, alpha) condition, draw correlated pattern
sets, store them in a Hopfield network, corrupt a sample of stored patterns,
run asynchronous retrieval dynamics, and record the retrieval-overlap
statistics (mean, SEM, bootstrap CI) needed to localize the storage-capacity
phase transition.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .hopfield import hebbian_weights, run_async_dynamics_batch, overlap, corrupt
from .patterns import generate_correlated_patterns, arcsin_law, empirical_pairwise_correlation
from .stats_utils import mean_and_sem, bootstrap_ci, find_critical_alpha, finite_size_extrapolate, fit_capacity_vs_rho


# ---------------------------------------------------------------------------
# Sweep grid definition
# ---------------------------------------------------------------------------

DEFAULT_NS = (100, 200, 400)
DEFAULT_RHOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
CLASSICAL_ALPHA_C = 0.138  # Amit-Gutfreund-Sompolinsky (1985) i.i.d. reference


def alpha_grid_for_rho(
    n: int,
    rho: float,
    n_dense: int = 10,
    n_sparse: int = 8,
    low_p_max: int = 25,
) -> np.ndarray:
    """Alpha grid concentrated near the H1-predicted transition location
    alpha_c(0) * (1 - rho), plus a coarser sweep across the full plausible
    range so the grid still brackets the true transition if H1 is wrong
    about *where* it sits, PLUS a fine grid of small absolute pattern
    counts P = 1..low_p_max converted to alpha = P/N.

    The low-P branch matters because correlated patterns (rho > 0) turn out
    empirically to collapse retrieval at a roughly N-independent *pattern
    count* P_c(rho) rather than an N-independent *load* alpha_c(rho) -- so
    for larger N the transition can sit at very small alpha, below what a
    fixed alpha-only grid would resolve. Only used to place sample points;
    the actual transition location is always measured from data."""
    center = CLASSICAL_ALPHA_C * (1 - rho)
    lo = max(0.01, center - 0.09)
    hi = center + 0.09
    dense = np.linspace(lo, hi, n_dense)
    sparse = np.linspace(0.02, 0.34, n_sparse)
    low_p = np.arange(1, low_p_max + 1) / n
    grid = np.unique(np.round(np.concatenate([dense, sparse, low_p]), 4))
    grid = grid[(grid > 0) & (grid <= 0.35)]
    return grid


@dataclass
class SweepPoint:
    n: int
    rho: float
    alpha: float
    p: int
    n_trials: int
    n_samples: int
    mean_overlap: float
    sem_overlap: float
    ci_lo: float
    ci_hi: float
    mean_sweeps_used: float


def run_sweep(
    ns=DEFAULT_NS,
    rhos=DEFAULT_RHOS,
    n_trials: int = 15,
    n_patterns_per_trial: int = 5,
    flip_frac: float = 0.05,
    max_sweeps: int = 20,
    seed: int = 12345,
    verbose: bool = True,
) -> list[SweepPoint]:
    """Run the full (N, rho, alpha) sweep and return per-point statistics."""
    rng = np.random.default_rng(seed)
    results: list[SweepPoint] = []

    total_points = sum(len(alpha_grid_for_rho(n, rho)) for n in ns for rho in rhos)
    done = 0
    t_start = time.time()

    for n in ns:
        for rho in rhos:
            alphas = alpha_grid_for_rho(n, rho)
            for alpha in alphas:
                p = max(2, int(round(alpha * n)))
                overlaps: list[float] = []
                sweeps_used_list: list[int] = []

                for _ in range(n_trials):
                    patterns = generate_correlated_patterns(n, p, rho, rng)
                    w = hebbian_weights(patterns)

                    n_test = min(n_patterns_per_trial, p)
                    test_idx = rng.choice(p, size=n_test, replace=False)
                    test_patterns = patterns[test_idx]

                    init_states = corrupt(test_patterns, flip_frac, rng)
                    final_states, sweeps_used = run_async_dynamics_batch(
                        w, init_states, rng, max_sweeps=max_sweeps
                    )
                    m = overlap(final_states, test_patterns)
                    overlaps.extend(np.abs(m).tolist())
                    sweeps_used_list.append(sweeps_used)

                mean_ov, sem_ov = mean_and_sem(np.array(overlaps))
                ci_lo, ci_hi = bootstrap_ci(np.array(overlaps), rng)

                results.append(
                    SweepPoint(
                        n=n,
                        rho=rho,
                        alpha=float(alpha),
                        p=p,
                        n_trials=n_trials,
                        n_samples=len(overlaps),
                        mean_overlap=mean_ov,
                        sem_overlap=sem_ov,
                        ci_lo=ci_lo,
                        ci_hi=ci_hi,
                        mean_sweeps_used=float(np.mean(sweeps_used_list)),
                    )
                )
                done += 1
                if verbose and done % 10 == 0:
                    elapsed = time.time() - t_start
                    print(f"  [{done}/{total_points}] N={n} rho={rho} alpha={alpha:.4f} "
                          f"mean_overlap={mean_ov:.3f} ({elapsed:.1f}s elapsed)")

    return results


def write_sweep_csv(results: list[SweepPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def load_sweep_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                "n": int(row["n"]),
                "rho": float(row["rho"]),
                "alpha": float(row["alpha"]),
                "p": int(row["p"]),
                "n_trials": int(row["n_trials"]),
                "n_samples": int(row["n_samples"]),
                "mean_overlap": float(row["mean_overlap"]),
                "sem_overlap": float(row["sem_overlap"]),
                "ci_lo": float(row["ci_lo"]),
                "ci_hi": float(row["ci_hi"]),
                "mean_sweeps_used": float(row["mean_sweeps_used"]),
            })
        return rows


# ---------------------------------------------------------------------------
# Post-processing: critical alpha, finite-size scaling, capacity-vs-rho fit
# ---------------------------------------------------------------------------

def compute_critical_alphas(rows: list[dict], threshold: float = 0.95) -> list[dict]:
    """For each (N, rho), interpolate alpha_c from the measured overlap curve."""
    ns = sorted({r["n"] for r in rows})
    rhos = sorted({r["rho"] for r in rows})
    out = []
    for n in ns:
        for rho in rhos:
            pts = [r for r in rows if r["n"] == n and r["rho"] == rho]
            pts.sort(key=lambda r: r["alpha"])
            alphas = np.array([r["alpha"] for r in pts])
            overlaps = np.array([r["mean_overlap"] for r in pts])
            alpha_c = find_critical_alpha(alphas, overlaps, threshold=threshold)
            out.append({"n": n, "rho": rho, "alpha_c": alpha_c})
    return out


def compute_finite_size_extrapolation(critical_alphas: list[dict]) -> list[dict]:
    rhos = sorted({r["rho"] for r in critical_alphas})
    out = []
    for rho in rhos:
        pts = [r for r in critical_alphas if r["rho"] == rho and r["alpha_c"] is not None]
        pts.sort(key=lambda r: r["n"])
        if len(pts) < 2:
            continue
        ns = np.array([r["n"] for r in pts], dtype=np.float64)
        alpha_cs = np.array([r["alpha_c"] for r in pts], dtype=np.float64)
        fit = finite_size_extrapolate(ns, alpha_cs)
        out.append({
            "rho": rho,
            "n_points_used": len(pts),
            "slope": fit.slope,
            "alpha_c_inf": fit.intercept,
            "alpha_c_inf_stderr": fit.intercept_stderr,
            "r_value": fit.r_value,
        })
    return out


def compute_capacity_vs_rho_fit(extrapolation: list[dict]) -> dict:
    extrapolation = sorted(extrapolation, key=lambda r: r["rho"])
    rhos = np.array([r["rho"] for r in extrapolation])
    alpha_c_inf = np.array([r["alpha_c_inf"] for r in extrapolation])
    fit = fit_capacity_vs_rho(rhos, alpha_c_inf)
    return {
        "alpha0": fit.alpha0,
        "linear_rmse": fit.linear_rmse,
        "linear_r2": fit.linear_r2,
        "power_k": fit.power_k,
        "power_rmse": fit.power_rmse,
        "power_r2": fit.power_r2,
        "better_model": fit.better_model,
        "spearman_rho": fit.spearman_rho,
        "spearman_p": fit.spearman_p,
    }


# ---------------------------------------------------------------------------
# Arcsin-law validation (pattern-generator sanity check, M4)
# ---------------------------------------------------------------------------

def run_arcsin_validation(
    rhos=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9),
    n: int = 2000,
    p: int = 200,
    n_trials: int = 10,
    seed: int = 999,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for rho in rhos:
        emp_corrs = []
        for _ in range(n_trials):
            patterns = generate_correlated_patterns(n, p, rho, rng)
            emp_corrs.append(empirical_pairwise_correlation(patterns))
        mean_emp, sem_emp = mean_and_sem(np.array(emp_corrs))
        theo = arcsin_law(rho)
        out.append({
            "rho": rho,
            "empirical_corr_mean": mean_emp,
            "empirical_corr_sem": sem_emp,
            "theoretical_corr": theo,
            "abs_error": abs(mean_emp - theo),
        })
    return out


def write_dicts_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
