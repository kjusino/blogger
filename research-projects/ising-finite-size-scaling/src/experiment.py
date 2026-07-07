"""Orchestrates the full study: finite-size scaling grid, Onsager validation,
and a Metropolis-vs-Wolff critical-slowing-down comparison.
"""
import time

import numpy as np

from . import theory
from .metropolis import run_metropolis
from .wolff import run_wolff
from .observables import (
    specific_heat,
    susceptibility,
    binder_cumulant,
    mean_abs_magnetization,
    mean_energy,
    integrated_autocorrelation_time,
)
from .scaling import (
    rescaled_temperature,
    rescaled_magnetization,
    rescaled_susceptibility,
    binder_crossing,
    collapse_rmse,
)


def run_fss_grid(L_values, T_grid, n_equil, n_sample, sample_interval, n_seeds):
    """Sweep Metropolis over every (L, T, seed), aggregate observables per (L, T).

    Returns a list of per-(L, T) row dicts (mean +/- seed-to-seed std for each
    observable) suitable for writing straight to CSV, plus the same data
    organized as per-L arrays for the scaling analysis.
    """
    rows = []
    by_L = {L: {"T": [], "m": [], "chi": [], "c": [], "u4": [], "e": []} for L in L_values}

    for L in L_values:
        N = L * L
        for T in T_grid:
            m_seeds, chi_seeds, c_seeds, u4_seeds, e_seeds = [], [], [], [], []
            for seed in range(n_seeds):
                result = run_metropolis(
                    L=L,
                    T=T,
                    n_equil=n_equil,
                    n_sample=n_sample,
                    sample_interval=sample_interval,
                    seed=1000 * L + seed,
                )
                m_seeds.append(mean_abs_magnetization(result["magnetization"]))
                chi_seeds.append(susceptibility(result["magnetization"], T, N))
                c_seeds.append(specific_heat(result["energy"], T, N))
                u4_seeds.append(binder_cumulant(result["magnetization"]))
                e_seeds.append(mean_energy(result["energy"]))

            row = {
                "L": L,
                "T": T,
                "m_mean": float(np.mean(m_seeds)),
                "m_std": float(np.std(m_seeds)),
                "chi_mean": float(np.mean(chi_seeds)),
                "chi_std": float(np.std(chi_seeds)),
                "c_mean": float(np.mean(c_seeds)),
                "c_std": float(np.std(c_seeds)),
                "u4_mean": float(np.mean(u4_seeds)),
                "u4_std": float(np.std(u4_seeds)),
                "e_mean": float(np.mean(e_seeds)),
                "e_std": float(np.std(e_seeds)),
            }
            rows.append(row)
            by_L[L]["T"].append(T)
            by_L[L]["m"].append(row["m_mean"])
            by_L[L]["chi"].append(row["chi_mean"])
            by_L[L]["c"].append(row["c_mean"])
            by_L[L]["u4"].append(row["u4_mean"])
            by_L[L]["e"].append(row["e_mean"])

    for L in L_values:
        for key in by_L[L]:
            by_L[L][key] = np.array(by_L[L][key])

    return rows, by_L


def analyze_scaling(by_L, L_values):
    """Binder-cumulant Tc estimate plus data-collapse RMSE (raw vs. rescaled)."""
    T_grid = by_L[L_values[0]]["T"]
    binder_by_L = {L: by_L[L]["u4"] for L in L_values}
    tc_result = binder_crossing(T_grid, binder_by_L, L_values)
    Tc = tc_result["Tc_estimate"] if tc_result["Tc_estimate"] is not None else theory.T_C

    raw_T_by_L = {L: by_L[L]["T"] for L in L_values}
    raw_m_by_L = {L: by_L[L]["m"] for L in L_values}
    raw_collapse = collapse_rmse(raw_T_by_L, raw_m_by_L, L_values)

    x_by_L, y_by_L = {}, {}
    chi_x_by_L, chi_y_by_L = {}, {}
    for L in L_values:
        x_by_L[L] = rescaled_temperature(by_L[L]["T"], Tc, L, theory.NU)
        y_by_L[L] = rescaled_magnetization(by_L[L]["m"], L, theory.BETA, theory.NU)
        chi_x_by_L[L] = x_by_L[L]
        chi_y_by_L[L] = rescaled_susceptibility(by_L[L]["chi"], L, theory.GAMMA, theory.NU)

    rescaled_collapse = collapse_rmse(x_by_L, y_by_L, L_values)
    rescaled_chi_collapse = collapse_rmse(chi_x_by_L, chi_y_by_L, L_values)

    return {
        "tc_estimate": tc_result,
        "tc_relative_error": (
            abs(tc_result["Tc_estimate"] - theory.T_C) / theory.T_C
            if tc_result["Tc_estimate"] is not None
            else None
        ),
        "raw_collapse_rmse": raw_collapse["mean_rmse"],
        "rescaled_magnetization_collapse_rmse": rescaled_collapse["mean_rmse"],
        "rescaled_susceptibility_collapse_rmse": rescaled_chi_collapse["mean_rmse"],
        "collapse_improvement_factor": (
            raw_collapse["mean_rmse"] / rescaled_collapse["mean_rmse"]
            if rescaled_collapse["mean_rmse"]
            else None
        ),
        "x_by_L": x_by_L,
        "y_by_L": y_by_L,
        "chi_y_by_L": chi_y_by_L,
    }


def onsager_validation(L_large, T_values, n_equil, n_sample, sample_interval, n_seeds):
    """Compare large-L Metropolis energy/magnetization against the exact
    infinite-lattice Onsager solution, at a set of temperatures spanning both
    sides of Tc."""
    rows = []
    for T in T_values:
        e_seeds, m_seeds = [], []
        for seed in range(n_seeds):
            result = run_metropolis(
                L=L_large,
                T=T,
                n_equil=n_equil,
                n_sample=n_sample,
                sample_interval=sample_interval,
                seed=5000 + seed,
            )
            e_seeds.append(mean_energy(result["energy"]))
            m_seeds.append(mean_abs_magnetization(result["magnetization"]))
        e_sim = float(np.mean(e_seeds))
        m_sim = float(np.mean(m_seeds))
        e_exact = theory.onsager_energy(T)
        m_exact = theory.onsager_magnetization(T)
        rows.append(
            {
                "T": T,
                "L": L_large,
                "energy_sim": e_sim,
                "energy_exact": e_exact,
                "energy_abs_error": abs(e_sim - e_exact),
                "magnetization_sim": m_sim,
                "magnetization_exact": m_exact,
                "magnetization_abs_error": abs(m_sim - m_exact),
            }
        )
    return rows


def autocorrelation_comparison(L, T_values, n_equil, n_sample):
    """For each T, measure integrated autocorrelation time (in sweeps/cluster
    steps) and wall-clock time per independent sample, for Metropolis vs
    Wolff."""
    rows = []
    for T in T_values:
        t0 = time.perf_counter()
        met = run_metropolis(L=L, T=T, n_equil=n_equil, n_sample=n_sample, sample_interval=1, seed=777)
        met_wall = time.perf_counter() - t0
        met_tau = integrated_autocorrelation_time(met["magnetization"])
        met_sec_per_indep_sample = met_wall / n_sample * (2 * met_tau)

        t0 = time.perf_counter()
        wolf = run_wolff(L=L, T=T, n_equil=n_equil, n_sample=n_sample, sample_interval=1, seed=777)
        wolf_wall = time.perf_counter() - t0
        wolf_tau = integrated_autocorrelation_time(wolf["magnetization"])
        wolf_sec_per_indep_sample = wolf_wall / n_sample * (2 * wolf_tau)

        rows.append(
            {
                "T": T,
                "L": L,
                "metropolis_tau_sweeps": float(met_tau),
                "wolff_tau_steps": float(wolf_tau),
                "metropolis_wall_seconds": met_wall,
                "wolff_wall_seconds": wolf_wall,
                "metropolis_seconds_per_independent_sample": met_sec_per_indep_sample,
                "wolff_seconds_per_independent_sample": wolf_sec_per_indep_sample,
                "mean_wolff_cluster_size": float(np.mean(wolf["cluster_size"])),
            }
        )
    return rows
