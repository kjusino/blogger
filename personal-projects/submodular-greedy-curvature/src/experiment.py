"""Sweep over weighted-coverage instances of varying redundancy intensity,
run greedy and exact brute-force optimization, and check whether the
curvature-refined approximation bound tracks greedy's realized performance
better than the classical worst-case (1-1/e) bound.
"""

import numpy as np
from scipy import stats

from src.greedy import greedy
from src.optimal import brute_force_opt
from src.submodular import make_grouped_redundancy_instance
from src.theory import WORST_CASE_BOUND, curvature_bound

DEFAULT_N_VALUES = (10, 14, 18)
DEFAULT_REDUNDANCY_MULTS = (
    0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.2, 2.0, 3.5, 6.0, 12.0, 30.0,
)
DEFAULT_SEEDS_PER_CONFIG = 6
DEFAULT_K_FRACS = (0.25, 0.4, 0.6)
DEFAULT_N_GROUPS = 10
DEFAULT_GROUP_MEMBERSHIP_PROB = 0.35
WEIGHT_LOW, WEIGHT_HIGH = 1.0, 10.0
VALIDITY_TOLERANCE = 1e-6


def _instance_id(n, mult, seed):
    return f"n{n}_mult{mult:.4f}_s{seed}"


def run_sweep(
    n_values=DEFAULT_N_VALUES,
    redundancy_mults=DEFAULT_REDUNDANCY_MULTS,
    seeds_per_config=DEFAULT_SEEDS_PER_CONFIG,
    k_fracs=DEFAULT_K_FRACS,
    n_groups=DEFAULT_N_GROUPS,
    group_membership_prob=DEFAULT_GROUP_MEMBERSHIP_PROB,
    base_seed=1000,
):
    """Run the full (n, redundancy_mult, seed, k) grid and return trial
    records. `redundancy_mult` is the design knob; the *realized* curvature
    (measured per instance via `WeightedCoverageFunction.curvature()`) is
    what the analysis actually keys on, since the two are related but not in
    closed form for this instance family."""
    records = []
    seed_counter = base_seed

    for n in n_values:
        for mult in redundancy_mults:
            for _rep in range(seeds_per_config):
                seed_counter += 1
                rng = np.random.default_rng(seed_counter)
                f = make_grouped_redundancy_instance(
                    n, n_groups, group_membership_prob, mult,
                    WEIGHT_LOW, WEIGHT_HIGH, rng,
                )
                c = f.curvature()
                inst_id = _instance_id(n, mult, seed_counter)

                for k_frac in k_fracs:
                    k = max(1, min(n, round(k_frac * n)))
                    opt_val, _ = brute_force_opt(f, k)
                    _, trace = greedy(f, k)
                    greedy_val = trace[-1]
                    ratio = greedy_val / opt_val if opt_val > 0 else 1.0

                    records.append({
                        "instance_id": inst_id,
                        "n": n,
                        "redundancy_mult": mult,
                        "seed": seed_counter,
                        "curvature": c,
                        "k": k,
                        "k_over_n": k / n,
                        "opt_val": opt_val,
                        "greedy_val": greedy_val,
                        "ratio": ratio,
                        "curvature_bound": curvature_bound(c),
                        "worst_case_bound": WORST_CASE_BOUND,
                    })

    return records


def compute_summary(records):
    """Compute the four preregistered success metrics (M1-M4) from records."""
    if not records:
        raise ValueError("no records to summarize")

    ratios = np.array([r["ratio"] for r in records])
    curvatures = np.array([r["curvature"] for r in records])
    curvature_bounds = np.array([r["curvature_bound"] for r in records])

    # --- M1: validity. The curvature bound is a proven theorem; it must
    # never be violated (up to floating-point tolerance).
    violations = ratios < (curvature_bounds - VALIDITY_TOLERANCE)
    m1_validity_rate = 1.0 - violations.mean()
    m1_pass = m1_validity_rate >= 0.99

    # --- M2: informativeness. Higher curvature should mean a harder
    # instance, i.e. a larger gap (1 - ratio) between greedy and optimal.
    gap = 1.0 - ratios
    rho, p_value = stats.spearmanr(curvatures, gap)
    m2_pass = bool(not np.isnan(rho) and rho > 0 and p_value < 0.01)

    # --- M3: the curvature-refined bound should characterize each
    # instance's *worst-observed-case* (min ratio across its k's) more
    # tightly than the constant worst-case bound does.
    by_instance = {}
    for r in records:
        by_instance.setdefault(r["instance_id"], []).append(r)

    curvature_dev, constant_dev = [], []
    per_instance_rows = []
    for inst_id, rows in by_instance.items():
        min_ratio = min(row["ratio"] for row in rows)
        c = rows[0]["curvature"]
        cb = rows[0]["curvature_bound"]
        curvature_dev.append(min_ratio - cb)
        constant_dev.append(min_ratio - WORST_CASE_BOUND)
        per_instance_rows.append({
            "instance_id": inst_id,
            "n": rows[0]["n"],
            "redundancy_mult": rows[0]["redundancy_mult"],
            "curvature": c,
            "curvature_bound": cb,
            "min_ratio": min_ratio,
            "mean_ratio": float(np.mean([row["ratio"] for row in rows])),
        })

    mae_curvature = float(np.mean(np.abs(curvature_dev)))
    mae_constant = float(np.mean(np.abs(constant_dev)))
    m3_pass = mae_curvature < mae_constant

    # --- M4: tightness at extremes. Near-modular (low curvature) instances
    # should be solved almost exactly by greedy.
    low_c_mask = curvatures < 0.05 + 1e-9
    high_c_mask = curvatures > 0.9 - 1e-9
    low_c_mean_ratio = float(ratios[low_c_mask].mean()) if low_c_mask.any() else None
    high_c_mean_ratio = float(ratios[high_c_mask].mean()) if high_c_mask.any() else None
    m4_pass = low_c_mean_ratio is not None and low_c_mean_ratio >= 0.999

    summary = {
        "n_trials": len(records),
        "n_instances": len(by_instance),
        "m1_validity": {
            "description": "fraction of trials with ratio >= curvature_bound - tol",
            "validity_rate": float(m1_validity_rate),
            "n_violations": int(violations.sum()),
            "pass": bool(m1_pass),
        },
        "m2_informativeness": {
            "description": "Spearman rho(curvature, 1-ratio), one-sided p < 0.01",
            "spearman_rho": float(rho) if not np.isnan(rho) else None,
            "p_value": float(p_value) if not np.isnan(p_value) else None,
            "pass": m2_pass,
        },
        "m3_superior_predictor": {
            "description": (
                "mean|min_ratio - curvature_bound| vs mean|min_ratio - worst_case_bound| "
                "per instance (lower = tighter characterization of worst-observed case)"
            ),
            "mae_curvature_bound": mae_curvature,
            "mae_constant_bound": mae_constant,
            "pass": m3_pass,
        },
        "m4_tightness_at_extremes": {
            "description": "mean ratio for curvature<0.05 (near-modular) vs curvature>0.9",
            "low_curvature_mean_ratio": low_c_mean_ratio,
            "high_curvature_mean_ratio": high_c_mean_ratio,
            "n_low_curvature_trials": int(low_c_mask.sum()),
            "n_high_curvature_trials": int(high_c_mask.sum()),
            "pass": m4_pass,
        },
        "worst_case_bound": WORST_CASE_BOUND,
        "overall_pass": bool(m1_pass and m2_pass and m3_pass and m4_pass),
    }
    return summary, per_instance_rows
