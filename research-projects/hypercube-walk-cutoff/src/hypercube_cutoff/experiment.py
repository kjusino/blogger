"""The three sweeps: lumping validation, cutoff-location/data-collapse, and
Monte Carlo validation of the exact chain against literal simulated walkers."""

from __future__ import annotations

import numpy as np

from . import chain as ch
from . import theory as th
from . import simulate as sim


def interp_crossing(t_values: np.ndarray, tv_values: np.ndarray, target: float) -> float:
    """Linear-interpolate the t at which a monotonically decreasing TV curve
    crosses `target`. Raises if the curve never spans `target`."""
    if not np.all(np.diff(tv_values) <= 1e-9):
        raise ValueError("tv_values must be non-increasing for interp_crossing")
    if target > tv_values[0] or target < tv_values[-1]:
        raise ValueError(f"target {target} outside range [{tv_values[-1]}, {tv_values[0]}] of tv_values")
    # tv_values is decreasing; reverse to get an increasing xp for np.interp
    return float(np.interp(target, tv_values[::-1], t_values[::-1]))


def validate_lumping(n_values, t_grid_fn=None, num_trials: int = 20000, seed: int = 0) -> list[dict]:
    """For small n, compare the exact lumped (birth-death) chain, the exact
    brute-force 2^n chain, and Monte Carlo simulation of literal bit-vector
    trajectories. All three should agree (the first two to machine precision,
    the third within statistical error) -- this is the empirical check that
    the lumping argument in `chain.py`'s docstring is correct, not just a
    plausible-sounding symmetry argument.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_values:
        t_star = th.cutoff_time(n) if n >= 2 else 1.0
        t_values = np.unique(np.clip(
            np.round(np.array([0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]) * t_star).astype(int), 0, None
        ))

        exact_lumped = ch.tv_curve(n, t_values)
        exact_full, _ = ch.full_chain_tv_curve(n, t_values)

        bv_weights = sim.simulate_bitvector_trajectories(n, int(t_values.max()), num_trials, rng)
        for i, t in enumerate(t_values):
            mc_tv, mc_ci = sim.empirical_tv_distance(
                bv_weights[t], n, n_bootstrap=400, rng=rng
            )
            rows.append({
                "n": n,
                "t": int(t),
                "exact_lumped_tv": float(exact_lumped[i]),
                "exact_bruteforce_tv": float(exact_full[i]),
                "lumped_vs_bruteforce_abs_diff": abs(float(exact_lumped[i]) - float(exact_full[i])),
                "mc_bitvector_tv": float(mc_tv),
                "mc_ci_lo": float(mc_ci[0]),
                "mc_ci_hi": float(mc_ci[1]),
                "mc_within_ci": bool(mc_ci[0] - 1e-6 <= exact_lumped[i] <= mc_ci[1] + 1e-6),
            })
    return rows


def cutoff_scaling_sweep(n_values, c_values) -> tuple[list[dict], list[dict]]:
    """Exact (no Monte Carlo) sweep of the lumped chain across n, checking
    whether the empirical half-mixing time and window converge to the
    asymptotic (n ln n)/2 location / Theta(n) window, whether the rescaled
    exact-TV curves collapse onto each other (a true, empirically-defined
    universal profile), and how that universal profile compares to
    `theory.limiting_profile` -- the profile predicted by the classical
    Diaconis-Shahshahani chi-square (Cauchy-Schwarz) bound.

    These are two logically distinct questions and the answers turn out to
    differ (see README): the cutoff location/window match the chi-square-bound
    theory well, but the bound's *profile shape* is not asymptotically tight,
    so `chi_square_bound_gap` does not shrink with n even though
    `self_collapse_error` (distance to the largest-n curve in the sweep) does.

    Returns (curve_rows, summary_rows): curve_rows has one row per (n, c);
    summary_rows has one row per n with the fitted half-mixing time, window,
    and both collapse-error metrics.
    """
    c_values = np.asarray(c_values, dtype=float)
    n_values = list(n_values)
    curve_rows = []
    summary_rows = []

    exact_by_n = {}
    for n in n_values:
        t_values = np.clip(np.round(th.unrescale(n, c_values)).astype(int), 0, None)
        exact_by_n[n] = (t_values, np.minimum(ch.tv_curve(n, t_values), 1.0))

    profile = np.minimum(th.limiting_profile(c_values), 1.0)
    reference_tv = exact_by_n[max(n_values)][1]

    for n in n_values:
        t_star = th.cutoff_time(n)
        t_values, exact_tv = exact_by_n[n]
        bound_tv = np.minimum(th.tv_upper_bound(n, t_values), 1.0)

        for c, t, etv, btv, prof in zip(c_values, t_values, exact_tv, bound_tv, profile):
            curve_rows.append({
                "n": n, "c": float(c), "t": int(t),
                "exact_tv": float(etv),
                "chi_square_bound": float(btv),
                "limiting_profile": float(prof),
            })

        t_half = interp_crossing(t_values, exact_tv, 0.5)
        rel_err_cutoff = abs(t_half - t_star) / t_star

        t_hi = interp_crossing(t_values, exact_tv, 0.75)
        t_lo = interp_crossing(t_values, exact_tv, 0.25)
        window = t_lo - t_hi

        chi_square_bound_gap = float(np.max(np.abs(exact_tv - profile)))
        self_collapse_error = float(np.max(np.abs(exact_tv - reference_tv)))

        summary_rows.append({
            "n": n,
            "t_star_theory": float(t_star),
            "t_half_empirical": float(t_half),
            "rel_err_half_vs_cutoff_time": float(rel_err_cutoff),
            "window": float(window),
            "window_over_n": float(window / n),
            "chi_square_bound_gap": chi_square_bound_gap,
            "self_collapse_error": self_collapse_error,
        })
    return curve_rows, summary_rows


def monte_carlo_validation(n: int, c_values, num_trials: int, seed: int = 0) -> list[dict]:
    """At a single moderate n, compare exact TV against Monte Carlo estimates
    from both the literal bit-vector simulator and the weight-only simulator,
    across the cutoff window."""
    rng_bv = np.random.default_rng(seed)
    rng_w = np.random.default_rng(seed + 1)
    rng_boot = np.random.default_rng(seed + 2)

    c_values = np.asarray(c_values, dtype=float)
    t_values = np.clip(np.round(th.unrescale(n, c_values)).astype(int), 0, None)
    t_max = int(t_values.max())

    exact_tv = ch.tv_curve(n, t_values)
    bv_weights = sim.simulate_bitvector_trajectories(n, t_max, num_trials, rng_bv)
    w_weights = sim.simulate_weight_trajectories(n, t_max, num_trials, rng_w)

    rows = []
    for c, t, etv in zip(c_values, t_values, exact_tv):
        mc_bv_tv, mc_bv_ci = sim.empirical_tv_distance(bv_weights[t], n, n_bootstrap=500, rng=rng_boot)
        mc_w_tv, mc_w_ci = sim.empirical_tv_distance(w_weights[t], n, n_bootstrap=500, rng=rng_boot)
        rows.append({
            "n": n, "c": float(c), "t": int(t),
            "exact_tv": float(etv),
            "mc_bitvector_tv": float(mc_bv_tv),
            "mc_bitvector_ci_lo": float(mc_bv_ci[0]),
            "mc_bitvector_ci_hi": float(mc_bv_ci[1]),
            "mc_weight_tv": float(mc_w_tv),
            "mc_weight_ci_lo": float(mc_w_ci[0]),
            "mc_weight_ci_hi": float(mc_w_ci[1]),
            "bitvector_within_ci": bool(mc_bv_ci[0] <= etv <= mc_bv_ci[1]),
            "weight_within_ci": bool(mc_w_ci[0] <= etv <= mc_w_ci[1]),
        })
    return rows
