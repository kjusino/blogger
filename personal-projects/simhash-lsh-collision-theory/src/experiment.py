"""Three falsifiable experiments comparing empirical SimHash/LSH behavior
against the closed-form theory in theory.py."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import theory
from .data import (
    planted_neighbor_dataset,
    promise_preserving_dataset,
    random_unit_vector,
    vector_at_angle,
)
from .hyperplane_hash import empirical_single_bit_collision_rate
from .lsh_index import LSHIndex


def run_single_hash_experiment(
    thetas: list[float],
    dims: list[int],
    num_trials: int,
    seed: int,
) -> pd.DataFrame:
    """Experiment 1: for each (theta, dim), estimate Pr[single random
    hyperplane agrees on u, v at angle theta] by Monte Carlo, and compare to
    theory.single_hash_collision_prob(theta)."""
    rng = np.random.default_rng(seed)
    rows = []
    for dim in dims:
        u = random_unit_vector(dim, rng)
        for theta in thetas:
            v = vector_at_angle(u, theta, rng)
            p_hat, stderr = empirical_single_bit_collision_rate(u, v, num_trials, rng)
            p_theory = theory.single_hash_collision_prob(theta)
            rows.append(
                {
                    "dim": dim,
                    "theta": theta,
                    "empirical_prob": p_hat,
                    "theory_prob": p_theory,
                    "stderr": stderr,
                    "abs_error": abs(p_hat - p_theory),
                    "z_score": abs(p_hat - p_theory) / stderr if stderr > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def run_and_or_threshold_experiment(
    k: int,
    L: int,
    thetas: list[float],
    num_trials: int,
    dim: int,
    seed: int,
) -> pd.DataFrame:
    """Experiment 2: for each theta, estimate Pr[u, v collide in >=1 of L
    independent length-k tables] by repeated random re-instantiation of the
    LSH index, and compare to theory.or_of_bands_prob(theta, k, L)."""
    rng = np.random.default_rng(seed)
    rows = []
    u = random_unit_vector(dim, rng)
    for theta in thetas:
        v = vector_at_angle(u, theta, rng)
        hits = 0
        for _ in range(num_trials):
            index = LSHIndex(dim=dim, k=k, L=L, rng=rng)
            index.index(u[None, :])
            if index.collides_with(v, target_id=0):
                hits += 1
        p_hat = hits / num_trials
        p_theory = theory.or_of_bands_prob(theta, k, L)
        stderr = math.sqrt(p_hat * (1.0 - p_hat) / num_trials)
        rows.append(
            {
                "theta": theta,
                "empirical_recall": p_hat,
                "theory_recall": p_theory,
                "stderr": stderr,
                "abs_error": abs(p_hat - p_theory),
            }
        )
    return pd.DataFrame(rows)


def find_empirical_threshold_angle(df: pd.DataFrame) -> float:
    """Linear-interpolate the theta at which empirical_recall crosses 0.5,
    from a DataFrame sorted by increasing theta (recall is monotonically
    decreasing in theta)."""
    df_sorted = df.sort_values("theta").reset_index(drop=True)
    recalls = df_sorted["empirical_recall"].to_numpy()
    thetas = df_sorted["theta"].to_numpy()
    for i in range(len(recalls) - 1):
        if recalls[i] >= 0.5 >= recalls[i + 1]:
            r0, r1 = recalls[i], recalls[i + 1]
            t0, t1 = thetas[i], thetas[i + 1]
            if r0 == r1:
                return t0
            frac = (r0 - 0.5) / (r0 - r1)
            return t0 + frac * (t1 - t0)
    # Fallback: nearest point to 0.5 if the grid never bracket 0.5.
    idx = int(np.argmin(np.abs(recalls - 0.5)))
    return float(thetas[idx])


def k_of_n(n: int, p2: float) -> int:
    """Signature length k(n) = ceil(log(n) / log(1/p2)), chosen so that the
    expected number of "far" (non-neighbor) points colliding with a query in
    a single table is O(1): n * p2^k(n) ~ 1. This is the standard
    Indyk-Motwani (1998) construction, not an arbitrary choice."""
    return max(1, math.ceil(math.log(n) / math.log(1.0 / p2)))


def L_of_n(n: int, rho: float, cap: int = 400) -> int:
    """Number of tables L(n) = ceil(n^rho), chosen so that the probability a
    near neighbor (single-hash collision prob p1) is found across L tables
    stays a constant (independent of n): L(n) * p1^k(n) ~ n^rho * n^(-rho) = 1.
    Capped defensively so a single run stays computationally bounded; if the
    cap binds, the measured exponent will read as an underestimate of rho.
    """
    return min(cap, max(1, math.ceil(n**rho)))


def run_scaling_experiment(
    near_theta: float,
    far_theta: float,
    dim: int,
    n_list: list[int],
    trials_per_n: int,
    seed: int,
    k: int | None = None,
    L: int | None = None,
    dataset_mode: str = "iid",
) -> pd.DataFrame:
    """Experiment 3: fix a near/far angle pair (near_theta collides often,
    far_theta collides rarely). For each background size n, use the
    Indyk-Motwani prescription k(n), L(n) (see k_of_n/L_of_n above) so that
    the *probability* of retrieving the near neighbor stays roughly constant
    across n, then measure how the mean LSH candidate-set size (the query
    cost) grows with n. Its fitted exponent is compared against the
    theoretical rho = ln(1/p1)/ln(1/p2).

    `k` and `L` are accepted for API symmetry/testing (to force a fixed,
    non-adaptive k, L) but default to the adaptive k_of_n/L_of_n schedule.

    `dataset_mode`:
      - "iid" (default): background is i.i.d. random unit vectors. Realistic,
        but the (near_theta, far_theta)-near-neighbor "promise" is only
        approximate -- see `planted_neighbor_dataset`'s docstring.
      - "promise": every background point is placed at EXACTLY far_theta
        (see `promise_preserving_dataset`), isolating the theorem's n^rho
        prediction from that confound.
    """
    rng = np.random.default_rng(seed)
    p1 = theory.single_hash_collision_prob(near_theta)
    p2 = theory.single_hash_collision_prob(far_theta)
    rho_theory = theory.rho_exponent(p1, p2)

    if dataset_mode not in {"iid", "promise"}:
        raise ValueError(f"unknown dataset_mode: {dataset_mode}")

    rows = []
    for n in n_list:
        k_n = k if k is not None else k_of_n(n, p2)
        L_n = L if L is not None else L_of_n(n, rho_theory)
        candidate_counts = []
        found_count = 0
        for _ in range(trials_per_n):
            if dataset_mode == "iid":
                query, dataset, planted_idx = planted_neighbor_dataset(n, dim, near_theta, rng)
            else:
                query, dataset, planted_idx = promise_preserving_dataset(
                    n, dim, near_theta, far_theta, rng
                )
            index = LSHIndex(dim=dim, k=k_n, L=L_n, rng=rng)
            index.index(dataset)
            candidates = index.query_candidates(query)
            candidate_counts.append(len(candidates))
            if planted_idx in candidates:
                found_count += 1
        rows.append(
            {
                "n": n,
                "k_n": k_n,
                "L_n": L_n,
                "mean_candidates": float(np.mean(candidate_counts)),
                "std_candidates": float(np.std(candidate_counts)),
                "near_neighbor_found_rate": found_count / trials_per_n,
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["p1"] = p1
    df.attrs["p2"] = p2
    df.attrs["rho_theory"] = rho_theory
    return df


def fit_power_law_exponent(n_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    """Fit y = c * n^rho via linear regression in log-log space.
    Returns (rho_hat, r_squared). Drops any y <= 0 (candidate count can be
    0 for small n, which is not representable in log-space)."""
    mask = y_values > 0
    log_n = np.log(n_values[mask])
    log_y = np.log(y_values[mask])
    if mask.sum() < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(log_n, log_y, 1)
    pred = slope * log_n + intercept
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(r_squared)
