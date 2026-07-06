"""Aggregation helpers turning a flat list of experiment records (dicts, see
`experiment.run_single`) into summary tables used for the README and figures."""

from __future__ import annotations

from collections import defaultdict

from .oracle import hoeffding_repetitions


def group_by(results, keys):
    groups = defaultdict(list)
    for r in results:
        groups[tuple(r[k] for k in keys)].append(r)
    return groups


def success_rate_table(results, by=("noise_rate", "strategy")):
    """dict[(key...)] -> fraction of runs where the learned hypothesis was
    exactly equivalent to the target."""
    out = {}
    for key, rows in group_by(results, by).items():
        out[key] = sum(1 for r in rows if r["success"]) / len(rows)
    return out


def mean_field_table(results, field, by=("noise_rate", "strategy")):
    """dict[(key...)] -> mean of `field` over rows where it is not None."""
    out = {}
    for key, rows in group_by(results, by).items():
        vals = [r[field] for r in rows if r.get(field) is not None]
        out[key] = sum(vals) / len(vals) if vals else float("nan")
    return out


def repetition_factor_table(results, by=("noise_rate", "strategy")):
    """dict[(key...)] -> mean(raw_queries / distinct_queries), the empirically
    realized average number of repeated sub-queries per distinct membership
    query actually issued by the redundancy wrapper."""
    out = {}
    for key, rows in group_by(results, by).items():
        ratios = [r["raw_queries"] / r["distinct_queries"] for r in rows if r["distinct_queries"] > 0]
        out[key] = sum(ratios) / len(ratios) if ratios else float("nan")
    return out


def union_bound_success_prediction(results, strategy, delta_q, by_noise_rate=True):
    """For each noise rate, predict a *lower bound* on learning-success
    probability via a union bound over the mean number of distinct membership
    queries L* issued: P(all correct) >= 1 - Q * delta_q.

    This is conservative (ignores that a single wrong answer often gets
    self-corrected by later table refinement), so we expect empirical success
    rates to sit at or above this curve.
    """
    distinct_by_noise = mean_field_table(
        [r for r in results if r["strategy"] == strategy],
        "distinct_queries", by=("noise_rate",),
    )
    return {
        (noise_rate,): max(0.0, 1.0 - q_mean * delta_q)
        for (noise_rate,), q_mean in distinct_by_noise.items()
    }


def sorted_noise_rates(results):
    return sorted({r["noise_rate"] for r in results})


def theoretical_repetitions_curve(noise_rates, delta_q):
    return {p: hoeffding_repetitions(p, delta_q) for p in noise_rates}
