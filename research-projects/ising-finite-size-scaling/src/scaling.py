"""Finite-size scaling analysis: Binder-cumulant Tc estimation and data collapse.

Finite-size scaling ansatz near a continuous phase transition:
    m(T, L)   = L^(-beta/nu) * f_m((T - Tc) * L^(1/nu))
    chi(T, L) = L^(gamma/nu) * f_chi((T - Tc) * L^(1/nu))
so plotting the rescaled variables m*L^(beta/nu) and chi*L^(-gamma/nu) against
the rescaled temperature x = (T - Tc)*L^(1/nu) should collapse curves from
different L onto one universal curve f_m / f_chi, *using only the exact
theoretical Tc and exponents* (no fitting to the simulation data itself).
"""
import numpy as np


def rescaled_temperature(T, Tc, L, nu):
    return (np.asarray(T, dtype=float) - Tc) * (L ** (1.0 / nu))


def rescaled_magnetization(m, L, beta, nu):
    return np.asarray(m, dtype=float) * (L ** (beta / nu))


def rescaled_susceptibility(chi, L, gamma, nu):
    return np.asarray(chi, dtype=float) * (L ** (-gamma / nu))


def binder_crossing(T_grid, binder_by_L, L_values):
    """Estimate Tc from pairwise crossings of Binder-cumulant curves.

    For each pair of adjacent system sizes, linearly interpolate both U4(T)
    curves on a common fine grid and find the T where their difference
    changes sign (the crossing point), which is, to leading finite-size
    corrections, independent of L and equal to Tc. Returns the mean crossing
    over all adjacent pairs plus the individual crossings for inspection.
    """
    T_grid = np.asarray(T_grid, dtype=float)
    fine_T = np.linspace(T_grid.min(), T_grid.max(), 20000)
    crossings = []
    pairs = []
    for i in range(len(L_values) - 1):
        L1, L2 = L_values[i], L_values[i + 1]
        u1 = np.interp(fine_T, T_grid, binder_by_L[L1])
        u2 = np.interp(fine_T, T_grid, binder_by_L[L2])
        diff = u1 - u2
        sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(sign_changes) == 0:
            continue
        # If multiple crossings exist, take the one nearest the midpoint of
        # the grid (the finite-size-scaling crossing is expected near Tc,
        # not at the grid edges).
        mid = 0.5 * (T_grid.min() + T_grid.max())
        idx = sign_changes[np.argmin(np.abs(fine_T[sign_changes] - mid))]
        crossing_T = fine_T[idx]
        crossings.append(crossing_T)
        pairs.append((L1, L2))

    if not crossings:
        return {"Tc_estimate": None, "crossings": [], "pairs": []}
    return {
        "Tc_estimate": float(np.mean(crossings)),
        "Tc_std": float(np.std(crossings)),
        "crossings": [float(c) for c in crossings],
        "pairs": pairs,
    }


def collapse_rmse(x_by_L, y_by_L, L_values, n_grid=200):
    """Quantify how well curves from different L overlap once rescaled.

    For each adjacent pair of L, interpolate both curves onto a common grid
    restricted to their overlapping x-range and compute the RMSE between
    them. Returns the mean RMSE across adjacent pairs (lower = better
    collapse) plus the per-pair values. Comparable RMSE computed on the raw
    (T, observable) curves (no rescaling) gives a baseline to show the
    rescaling genuinely improves the overlap, not just relabels the axes.
    """
    rmses = []
    for i in range(len(L_values) - 1):
        L1, L2 = L_values[i], L_values[i + 1]
        x1, y1 = x_by_L[L1], y_by_L[L1]
        x2, y2 = x_by_L[L2], y_by_L[L2]
        lo = max(x1.min(), x2.min())
        hi = min(x1.max(), x2.max())
        if hi <= lo:
            continue
        grid = np.linspace(lo, hi, n_grid)
        order1 = np.argsort(x1)
        order2 = np.argsort(x2)
        yi1 = np.interp(grid, x1[order1], y1[order1])
        yi2 = np.interp(grid, x2[order2], y2[order2])
        scale = 0.5 * (np.abs(yi1).max() + np.abs(yi2).max())
        scale = scale if scale > 0 else 1.0
        rmse = np.sqrt(np.mean(((yi1 - yi2) / scale) ** 2))
        rmses.append(rmse)
    return {
        "mean_rmse": float(np.mean(rmses)) if rmses else None,
        "pair_rmses": [float(r) for r in rmses],
    }
