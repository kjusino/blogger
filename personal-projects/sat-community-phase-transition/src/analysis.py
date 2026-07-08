"""Aggregation and statistics over the raw per-instance results CSV."""

import csv
import statistics

CLASSICAL_THRESHOLD = 4.267  # widely cited asymptotic 3-SAT satisfiability threshold


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            rows.append({
                "n_vars": int(raw["n_vars"]),
                "n_communities": int(raw["n_communities"]),
                "alpha": float(raw["alpha"]),
                "mu": float(raw["mu"]),
                "trial": int(raw["trial"]),
                "n_clauses": int(raw["n_clauses"]),
                "satisfiable": bool(int(raw["satisfiable"])),
                "hit_cap": bool(int(raw["hit_cap"])),
                "decisions": int(raw["decisions"]),
                "backtracks": int(raw["backtracks"]),
                "modularity_q": float(raw["modularity_q"]) if raw["modularity_q"] not in ("", "None") else None,
                "runtime_sec": float(raw["runtime_sec"]),
            })
    return rows


def group_by(rows, *keys):
    groups = {}
    for row in rows:
        k = tuple(row[key] for key in keys)
        groups.setdefault(k, []).append(row)
    return groups


def aggregate_by_alpha_mu(rows):
    """Returns dict[(alpha, mu)] -> {p_sat, median_decisions, mean_q, n}."""
    groups = group_by(rows, "alpha", "mu")
    out = {}
    for key, group in groups.items():
        decisions = [r["decisions"] for r in group]
        qs = [r["modularity_q"] for r in group if r["modularity_q"] is not None]
        out[key] = {
            "p_sat": sum(r["satisfiable"] for r in group) / len(group),
            "median_decisions": statistics.median(decisions),
            "mean_decisions": statistics.mean(decisions),
            "mean_q": statistics.mean(qs) if qs else None,
            "n": len(group),
        }
    return out


def sorted_unique(rows, key):
    return sorted({r[key] for r in rows})


def nearest_alpha(alphas, target):
    return min(alphas, key=lambda a: abs(a - target))


def mann_whitney_u(sample_a, sample_b):
    """Two-sided Mann-Whitney U test. Returns (u_statistic, p_value).

    Uses scipy if available; falls back to a normal-approximation
    implementation so this module has no hard scipy dependency.
    """
    try:
        from scipy.stats import mannwhitneyu
        result = mannwhitneyu(sample_a, sample_b, alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    except ImportError:
        return _mann_whitney_u_normal_approx(sample_a, sample_b)


def _mann_whitney_u_normal_approx(sample_a, sample_b):
    import math

    n_a, n_b = len(sample_a), len(sample_b)
    all_vals = sample_a + sample_b
    order = sorted(range(len(all_vals)), key=lambda i: all_vals[i])
    ranks = [0.0] * len(all_vals)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and all_vals[order[j]] == all_vals[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    rank_sum_a = sum(ranks[:n_a])
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    u = min(u_a, n_a * n_b - u_a)
    mean_u = n_a * n_b / 2.0
    std_u = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12.0)
    if std_u == 0:
        return u, 1.0
    z = (u - mean_u) / std_u
    p = 2 * (1 - _normal_cdf(abs(z)))
    return u, p


def _normal_cdf(x):
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def pearson_correlation(xs, ys):
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(xs, ys)
        return float(r), float(p)
    except ImportError:
        n = len(xs)
        mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        std_x = statistics.pstdev(xs)
        std_y = statistics.pstdev(ys)
        r = cov / (n * std_x * std_y) if std_x and std_y else 0.0
        return r, None
