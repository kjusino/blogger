"""Thermodynamic observables and autocorrelation analysis from MC sample series.

All observables here take per-spin (intensive) energy/magnetization sample
arrays as produced by metropolis.run_metropolis / wolff.run_wolff, plus N =
L*L, and return per-spin susceptibility/specific heat, matching the standard
finite-lattice conventions used in the Ising MC literature (in particular,
using <|m|> rather than <m> so the susceptibility estimator stays well
defined below T_c on a finite lattice where <m> -> 0 by symmetry).
"""
import numpy as np


def specific_heat(energy_samples, T, N):
    """c = N * Var(e) / T^2, where e is energy per spin."""
    return N * np.var(energy_samples, ddof=0) / (T ** 2)


def susceptibility(mag_samples, T, N):
    """chi = N * (<m^2> - <|m|>^2) / T, where m is magnetization per spin."""
    abs_m = np.abs(mag_samples)
    return N * (np.mean(mag_samples ** 2) - np.mean(abs_m) ** 2) / T


def binder_cumulant(mag_samples):
    """U4 = 1 - <m^4> / (3 <m^2>^2)."""
    m2 = np.mean(mag_samples ** 2)
    m4 = np.mean(mag_samples ** 4)
    return 1.0 - m4 / (3.0 * m2 ** 2)


def mean_abs_magnetization(mag_samples):
    return float(np.mean(np.abs(mag_samples)))


def mean_energy(energy_samples):
    return float(np.mean(energy_samples))


def integrated_autocorrelation_time(series, c=5, max_lag=None):
    """Sokal's windowed integrated autocorrelation time estimate.

    tau_int = 0.5 + sum_{t=1}^{M} rho(t), where rho is the normalized
    autocorrelation function and the summation window M is the smallest value
    satisfying M >= c * tau_int(M) (self-consistent window), which bounds the
    relative bias of truncating the (in principle infinite) sum. Falls back
    to the full series length if the window condition is never met (e.g. for
    extremely long correlation times relative to the sample size).
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    x = x - x.mean()
    var = np.mean(x ** 2)
    if var == 0.0:
        return 0.5
    if max_lag is None:
        max_lag = n - 1

    rho = np.empty(max_lag)
    for t in range(1, max_lag + 1):
        rho[t - 1] = np.mean(x[: n - t] * x[t:]) / var

    tau = 0.5
    for t in range(1, max_lag + 1):
        tau += rho[t - 1]
        if t >= c * max(tau, 0.5):
            break
    # tau_int >= 0.5 always for a genuine stationary series; small-sample
    # noise (especially for fast-mixing series where rho oscillates in sign)
    # can otherwise push the truncated sum below that floor.
    return max(tau, 0.5)
