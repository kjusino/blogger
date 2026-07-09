"""Grid-sweep orchestration: at each (sigma_w2, sigma_b2) grid point, compute
the mean-field theory summary and three independent empirical measurements,
then package everything into a flat list of dicts ready to write to CSV/JSON
and to plot.
"""

import csv
import json

import numpy as np

from meanfield import analyze_point, tanh, tanh_prime
from network import propagate_correlation, propagate_gradient_norms, train_classifier


def fit_log_decay_rate(values, floor=1e-7):
    """Fit log(|values|) ~ a - rate * layer_index by least squares, using
    only points above `floor` (below that, float64 noise dominates and would
    bias the fit). Returns (rate, chi_hat) where chi_hat = exp(-rate) is the
    per-layer multiplicative factor implied by the fit -- directly comparable
    to a theoretical chi_1."""
    values = np.asarray(values, dtype=float)
    idx = np.arange(len(values))
    mask = np.abs(values) > floor
    if mask.sum() < 3:
        return np.nan, np.nan
    y = np.log(np.abs(values[mask]))
    x = idx[mask]
    A = np.stack([np.ones_like(x, dtype=float), x], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    rate = -coef[1]
    chi_hat = float(np.exp(-rate))
    return float(rate), chi_hat


def empirical_correlation_chi(sigma_w2, sigma_b2, q_star, depth=60, width=400, seed=0):
    c_hat = propagate_correlation(depth, width, sigma_w2, sigma_b2, seed, q_star)
    # decay of the deviation from the *empirical* asymptote (last few layers'
    # mean), which is a good proxy for c* without re-running mean-field code
    c_inf = float(np.mean(c_hat[-5:]))
    rate, chi_hat = fit_log_decay_rate(c_hat - c_inf, floor=1e-6)
    return chi_hat, c_inf, c_hat


def empirical_gradient_chi(sigma_w2, sigma_b2, depth=60, width=200, seed=0, n_draws=4):
    """Average the fitted per-layer growth/decay rate over a few independent
    network draws to reduce finite-width noise in the fit."""
    chis = []
    curves = []
    for k in range(n_draws):
        g = propagate_gradient_norms(depth, width, sigma_w2, sigma_b2, seed=seed + k)
        rate, chi_hat = fit_log_decay_rate(g, floor=1e-12)
        if not np.isnan(chi_hat):
            chis.append(chi_hat)
        curves.append(g)
    chi_hat_mean = float(np.mean(chis)) if chis else np.nan
    return chi_hat_mean, np.mean(np.stack(curves), axis=0)


DEPTH_STAIRCASE = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96]


def empirical_max_trainable_depth(
    sigma_w2, sigma_b2, width=64, seed=0, depths=DEPTH_STAIRCASE, n_seeds=2, acc_threshold=0.8
):
    """Walk the depth staircase in increasing order. At each depth, train
    `n_seeds` independent random-init/data-shuffle draws and average their
    final training accuracy (a single training run at moderate depth is
    noisy enough -- one unlucky init can fail even a genuinely trainable
    depth -- that averaging is needed to get a monotonic-ish staircase).
    Stops at the first depth whose mean accuracy falls below
    `acc_threshold`; returns the last depth that cleared it (0 if even the
    shallowest network failed)."""
    best = 0
    for depth in depths:
        accs = [
            train_classifier(
                depth=depth, width=width, sigma_w2=sigma_w2, sigma_b2=sigma_b2, seed=seed + 1000 * s
            )["final_acc"]
            for s in range(n_seeds)
        ]
        if np.mean(accs) >= acc_threshold:
            best = depth
        else:
            break
    return best


def run_grid_sweep(sigma_w2_grid, sigma_b2_grid, seed=0, log=print):
    rows = []
    total = len(sigma_w2_grid) * len(sigma_b2_grid)
    n_done = 0
    for sigma_b2 in sigma_b2_grid:
        for sigma_w2 in sigma_w2_grid:
            theory = analyze_point(sigma_w2, sigma_b2, tanh, tanh_prime)

            corr_chi_hat, c_inf, _ = empirical_correlation_chi(
                sigma_w2, sigma_b2, theory["q_star"], seed=seed
            )
            grad_chi_hat, _ = empirical_gradient_chi(sigma_w2, sigma_b2, seed=seed)
            l_star = empirical_max_trainable_depth(sigma_w2, sigma_b2, seed=seed)

            rows.append(
                {
                    "sigma_w2": sigma_w2,
                    "sigma_b2": sigma_b2,
                    "q_star": theory["q_star"],
                    "chi1_theory": theory["chi1"],
                    "xi_c_theory": theory["xi_c"],
                    "c_star_theory": theory["c_star"],
                    "phase": theory["phase"],
                    "chi1_empirical_corr": corr_chi_hat,
                    "c_inf_empirical": c_inf,
                    "chi1_empirical_grad": grad_chi_hat,
                    "max_trainable_depth": l_star,
                }
            )
            n_done += 1
            log(
                f"[{n_done}/{total}] sw2={sigma_w2:.3f} sb2={sigma_b2:.3f} "
                f"chi1_theory={theory['chi1']:.4f} L*={l_star}"
            )
    return rows


def write_csv(rows, path):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _log_log_pearson(xi, l_star):
    finite = np.isfinite(xi) & (l_star > 0)
    if finite.sum() <= 2:
        return float("nan"), int(finite.sum())
    log_xi = np.log(np.clip(xi[finite], 1e-9, None))
    log_l = np.log(l_star[finite].astype(float))
    return float(np.corrcoef(log_xi, log_l)[0, 1]), int(finite.sum())


def summarize(rows):
    xi = np.array([r["xi_c_theory"] for r in rows])
    l_star = np.array([r["max_trainable_depth"] for r in rows])
    phase = np.array([r["phase"] for r in rows])

    pearson_log, n_finite = _log_log_pearson(xi, l_star)
    pearson_log_ordered, n_ordered = _log_log_pearson(xi[phase == "ordered"], l_star[phase == "ordered"])
    pearson_log_chaotic, n_chaotic = _log_log_pearson(xi[phase == "chaotic"], l_star[phase == "chaotic"])

    chi1_theory = np.array([r["chi1_theory"] for r in rows])
    chi1_corr_emp = np.array([r["chi1_empirical_corr"] for r in rows])
    chi1_grad_emp = np.array([r["chi1_empirical_grad"] for r in rows])

    mask_c = np.isfinite(chi1_corr_emp)
    corr_rel_err = float(
        np.median(np.abs(chi1_corr_emp[mask_c] - chi1_theory[mask_c]) / chi1_theory[mask_c])
    )
    pearson_corr = float(np.corrcoef(chi1_theory[mask_c], chi1_corr_emp[mask_c])[0, 1])

    mask_g = np.isfinite(chi1_grad_emp)
    grad_rel_err = float(
        np.median(np.abs(chi1_grad_emp[mask_g] - chi1_theory[mask_g]) / chi1_theory[mask_g])
    )
    pearson_grad = float(np.corrcoef(chi1_theory[mask_g], chi1_grad_emp[mask_g])[0, 1])

    return {
        "n_grid_points": len(rows),
        "forward_correlation_chi1": {
            "median_relative_error": corr_rel_err,
            "pearson_r_theory_vs_empirical": pearson_corr,
        },
        "backprop_gradient_chi1": {
            "median_relative_error": grad_rel_err,
            "pearson_r_theory_vs_empirical": pearson_grad,
        },
        "trainable_depth_vs_correlation_length": {
            "n_points_used": n_finite,
            "pearson_r_log_xi_vs_log_Lstar": pearson_log,
            "ordered_phase": {"n_points_used": n_ordered, "pearson_r": pearson_log_ordered},
            "chaotic_phase": {"n_points_used": n_chaotic, "pearson_r": pearson_log_chaotic},
        },
    }


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
