"""Sweep runner: builds networks, runs the QS-method SIS sweep over a tau
grid tailored to each network's own QMF/HMF predictions, locates the
empirical epidemic threshold via susceptibility-peak interpolation, and
writes/reads all of it as CSV."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from .networks import build_network, compute_network_stats, NetworkStats
from .sis_simulation import run_qs_simulation
from .stats_utils import parabolic_peak, relative_error, exact_spearman

DEFAULT_NS = [300, 600, 1000]
DEFAULT_TOPOLOGIES = ["rr", "er", "ba"]
MEAN_DEGREE = 6.0
DELTA = 0.15  # recovery probability per step; kept small so the discrete
              # synchronous update approximates the continuous-time SIS
              # process the QMF/HMF thresholds are derived for.
N_STEPS = 3000
BURN_IN = 1200
N_REPEATS = 4
N_REALIZATIONS = 3  # independent network draws per (topology, N)
BUFFER_SIZE = 30
BUFFER_UPDATE_INTERVAL = 10


def build_tau_grid(qmf_tc: float, hmf_tc: float, n_wide: int = 10, n_dense: int = 8) -> np.ndarray:
    """tau grid spanning a wide log-range plus a dense linear band bracketing
    the two theoretical predictions, so the susceptibility peak is resolved
    regardless of which theory (if either) is closer to correct."""
    lo = min(qmf_tc, hmf_tc)
    hi = max(qmf_tc, hmf_tc)
    wide = np.geomspace(max(1e-6, 0.25 * lo), 3.5 * hi, n_wide)
    dense = np.linspace(max(1e-6, 0.6 * lo), 1.5 * hi, n_dense)
    return np.unique(np.concatenate([wide, dense]))


def dense_band(qmf_tc: float, hmf_tc: float) -> tuple[float, float]:
    """The [0.6*lo, 1.5*hi] band bracketing both theoretical predictions --
    used to restrict susceptibility-peak search to the region the transition
    is actually expected in. This matters: deep in the subcritical regime the
    QS method's reinjection mechanism keeps mean prevalence pinned near a
    small noise floor, and its fluctuations there can produce spuriously
    large susceptibility (chi = N*Var/Mean blows up when Mean is small and
    noisy) that has nothing to do with the real transition and can otherwise
    hijack a whole-grid argmax."""
    lo = min(qmf_tc, hmf_tc)
    hi = max(qmf_tc, hmf_tc)
    return 0.6 * lo, 1.5 * hi


def run_one_replicate(A, tau_grid: np.ndarray, delta: float, n_steps: int, burn_in: int,
                       seed: int, buffer_size: int = BUFFER_SIZE,
                       buffer_update_interval: int = BUFFER_UPDATE_INTERVAL) -> tuple[np.ndarray, np.ndarray]:
    mean_rhos = np.empty(len(tau_grid))
    sus = np.empty(len(tau_grid))
    for i, tau in enumerate(tau_grid):
        beta = float(tau * delta)
        result = run_qs_simulation(
            A, beta, delta, n_steps, burn_in, seed=seed * 100_003 + i,
            buffer_size=buffer_size, buffer_update_interval=buffer_update_interval,
        )
        mean_rhos[i] = result.mean_rho
        sus[i] = result.susceptibility
    return mean_rhos, sus


def run_network_sweep(topology: str, n: int, mean_degree: float, delta: float,
                       n_steps: int, burn_in: int, n_repeats: int, seed: int) -> dict:
    G = build_network(topology, n, mean_degree, seed=seed)
    stats = compute_network_stats(G, seed=seed)
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    tau_grid = build_tau_grid(stats.qmf_threshold, stats.hmf_threshold)
    band_lo, band_hi = dense_band(stats.qmf_threshold, stats.hmf_threshold)
    band_mask = (tau_grid >= band_lo) & (tau_grid <= band_hi)

    all_rho = np.empty((n_repeats, len(tau_grid)))
    all_sus = np.empty((n_repeats, len(tau_grid)))
    peak_taus = np.empty(n_repeats)

    for rep in range(n_repeats):
        rhos, sus = run_one_replicate(A, tau_grid, delta, n_steps, burn_in, seed=seed * 7 + rep)
        all_rho[rep] = rhos
        all_sus[rep] = sus
        tau_peak, _ = parabolic_peak(tau_grid[band_mask], sus[band_mask])
        peak_taus[rep] = tau_peak

    tau_c_empirical = float(peak_taus.mean())
    tau_c_sem = float(peak_taus.std(ddof=1) / np.sqrt(n_repeats)) if n_repeats > 1 else float("nan")

    sweep_rows = []
    for i, tau in enumerate(tau_grid):
        sweep_rows.append({
            "topology": topology,
            "n": n,
            "tau": float(tau),
            "mean_rho_mean": float(all_rho[:, i].mean()),
            "mean_rho_sem": float(all_rho[:, i].std(ddof=1) / np.sqrt(n_repeats)),
            "susceptibility_mean": float(all_sus[:, i].mean()),
            "susceptibility_sem": float(all_sus[:, i].std(ddof=1) / np.sqrt(n_repeats)),
        })

    return {
        "topology": topology,
        "n": n,
        "stats": stats,
        "tau_grid": tau_grid,
        "sweep_rows": sweep_rows,
        "peak_taus": peak_taus.tolist(),
        "tau_c_empirical": tau_c_empirical,
        "tau_c_sem": tau_c_sem,
    }


def summarize_network(result: dict) -> dict:
    stats: NetworkStats = result["stats"]
    tau_c = result["tau_c_empirical"]
    eps_qmf = relative_error(stats.qmf_threshold, tau_c)
    eps_hmf = relative_error(stats.hmf_threshold, tau_c)
    return {
        "topology": result["topology"],
        "n": result["n"],
        "mean_degree": stats.mean_degree,
        "mean_sq_degree": stats.mean_sq_degree,
        "heterogeneity_ratio": stats.heterogeneity_ratio,
        "lambda_max": stats.lambda_max,
        "qmf_threshold": stats.qmf_threshold,
        "hmf_threshold": stats.hmf_threshold,
        "tau_c_empirical": tau_c,
        "tau_c_sem": result["tau_c_sem"],
        "eps_qmf": eps_qmf,
        "eps_hmf": eps_hmf,
        "gap_hmf_minus_qmf": eps_hmf - eps_qmf,
    }


def run_full_experiment(ns=DEFAULT_NS, topologies=DEFAULT_TOPOLOGIES, mean_degree=MEAN_DEGREE,
                         delta=DELTA, n_steps=N_STEPS, burn_in=BURN_IN, n_repeats=N_REPEATS,
                         n_realizations=N_REALIZATIONS, seed=12345, verbose=True) -> tuple[list[dict], list[dict]]:
    """Sweeps every (topology, N) combination with `n_realizations`
    independent network draws each (distinct random graph instances, not
    just distinct epidemic-dynamics seeds on one fixed graph), so the
    heterogeneity-vs-accuracy-gap correlation test has more than one point
    per topology to work with -- otherwise it is testing 3 topology
    "clusters" rather than a real trend."""
    all_sweep_rows = []
    summaries = []
    combo_seed = seed
    for topology in topologies:
        for n in ns:
            for realization in range(n_realizations):
                if verbose:
                    print(f"  running topology={topology} n={n} realization={realization} ...")
                result = run_network_sweep(topology, n, mean_degree, delta, n_steps, burn_in,
                                            n_repeats, seed=combo_seed)
                for row in result["sweep_rows"]:
                    row["realization"] = realization
                all_sweep_rows.extend(result["sweep_rows"])
                summary = summarize_network(result)
                summary["realization"] = realization
                summaries.append(summary)
                combo_seed += 1
    return all_sweep_rows, summaries


def aggregate_by_topology_n(summaries: list[dict], topologies=DEFAULT_TOPOLOGIES,
                             ns=DEFAULT_NS) -> list[dict]:
    """Average the per-realization summaries down to one row per (topology,
    N), with tau_c_sem now reflecting realization-to-realization (structural)
    variability rather than just within-realization dynamics noise -- a more
    honest uncertainty than a single-realization SEM would give."""
    groups = defaultdict(list)
    for s in summaries:
        groups[(s["topology"], s["n"])].append(s)

    aggregated = []
    for topology in topologies:
        for n in ns:
            items = groups.get((topology, n), [])
            if not items:
                continue
            taus = np.array([it["tau_c_empirical"] for it in items])
            n_items = len(items)
            aggregated.append({
                "topology": topology,
                "n": n,
                "n_realizations": n_items,
                "mean_degree": float(np.mean([it["mean_degree"] for it in items])),
                "mean_sq_degree": float(np.mean([it["mean_sq_degree"] for it in items])),
                "heterogeneity_ratio": float(np.mean([it["heterogeneity_ratio"] for it in items])),
                "lambda_max": float(np.mean([it["lambda_max"] for it in items])),
                "qmf_threshold": float(np.mean([it["qmf_threshold"] for it in items])),
                "hmf_threshold": float(np.mean([it["hmf_threshold"] for it in items])),
                "tau_c_empirical": float(taus.mean()),
                "tau_c_sem": float(taus.std(ddof=1) / np.sqrt(n_items)) if n_items > 1 else float("nan"),
                "eps_qmf": float(np.mean([it["eps_qmf"] for it in items])),
                "eps_hmf": float(np.mean([it["eps_hmf"] for it in items])),
                "gap_hmf_minus_qmf": float(np.mean([it["gap_hmf_minus_qmf"] for it in items])),
            })
    return aggregated


def compute_heterogeneity_correlation(summaries: list[dict]) -> dict:
    heterogeneity = np.array([s["heterogeneity_ratio"] for s in summaries])
    gap = np.array([s["gap_hmf_minus_qmf"] for s in summaries])
    valid = ~np.isnan(gap)
    rho, p = exact_spearman(heterogeneity[valid], gap[valid])
    return {"n_points": int(valid.sum()), "spearman_rho": rho, "spearman_p": p}


def write_dicts_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_dicts_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = int(v)
                except ValueError:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
        return rows
