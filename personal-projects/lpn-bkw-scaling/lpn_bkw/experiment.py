"""Sweep harness: run the BKW attack across (n, b, tau) grids and aggregate."""

import random
import statistics

from . import theory
from .bkw import attack


def run_config(n, b, tau, trials, confidence_const=20.0, margin=1.3, seed=None):
    """Run `trials` independent attacks for one (n, b, tau) configuration."""
    rng = random.Random(seed)
    results = []
    for _ in range(trials):
        res = attack(n, b, tau, rng=rng, confidence_const=confidence_const, margin=margin)
        results.append(res)

    successes = [r["success"] for r in results]
    queries = [r["total_raw_queries"] for r in results]
    times = [r["wall_time_sec"] for r in results]

    return {
        "n": n,
        "b": b,
        "a_levels": n // b,
        "tau": tau,
        "confidence_const": confidence_const,
        "trials": trials,
        "success_rate": sum(successes) / trials,
        "mean_total_queries": statistics.mean(queries),
        "stdev_total_queries": statistics.pstdev(queries) if len(queries) > 1 else 0.0,
        "mean_wall_time_sec": statistics.mean(times),
        "stdev_wall_time_sec": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "theoretical_total_queries": theory.total_queries(n, b, tau, confidence_const, margin),
    }


def run_sweep(configs, trials, confidence_const=20.0, margin=1.3, seed=None):
    """configs: iterable of (n, b, tau) tuples."""
    out = []
    for i, (n, b, tau) in enumerate(configs):
        cfg_seed = None if seed is None else seed + i
        out.append(run_config(n, b, tau, trials, confidence_const, margin, cfg_seed))
    return out


def find_required_confidence_const(n, b, tau, candidate_consts, trials,
                                    target_success=0.9, margin=1.3, seed=None):
    """Smallest candidate confidence_const hitting >= target_success.

    Returns (required_const_or_None, list_of_(const, success_rate)_tried).
    None means no candidate in the list reached the target; the caller
    should look at the trace to see how close the largest candidate got.
    """
    trace = []
    for i, c in enumerate(sorted(candidate_consts)):
        cfg_seed = None if seed is None else seed + i
        result = run_config(n, b, tau, trials, confidence_const=c, margin=margin, seed=cfg_seed)
        trace.append((c, result["success_rate"]))
        if result["success_rate"] >= target_success:
            return c, trace
    return None, trace
