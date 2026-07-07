"""Exact theoretical results for the 2D square-lattice Ising model (J=1, k_B=1).

These are closed-form results independent of any simulation, used purely as
ground truth to validate the Monte Carlo code:

  * T_c: Onsager's exact critical temperature (Onsager, 1944).
  * onsager_energy(T): exact infinite-lattice internal energy per spin at any T
    (Onsager 1944; see e.g. Pathria & Beale, "Statistical Mechanics", the
    "exact energy of the 2D Ising model" section).
  * onsager_magnetization(T): exact infinite-lattice spontaneous magnetization
    per spin for T < T_c (Yang, 1952).

Critical exponents for the 2D Ising universality class are also exact and
known in closed form (beta=1/8, gamma=7/4, nu=1) and are used in scaling.py.
"""
import numpy as np
from scipy import integrate

J_DEFAULT = 1.0

T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ~= 2.269185314213022

BETA = 1.0 / 8.0
GAMMA = 7.0 / 4.0
NU = 1.0
ALPHA = 0.0  # logarithmic divergence, not a power law


def _complete_elliptic_K(k):
    """Complete elliptic integral of the first kind, K(k) = int_0^{pi/2} dtheta / sqrt(1 - k^2 sin^2 theta).

    Uses scipy.special.ellipkm1(p) = K(sqrt(1-p)) (the "parameter" m = 1-p
    convention, numerically stable as k -> 1, which happens exactly at T=T_c).
    Cross-checked in tests against a direct quadrature of the integral above
    at moduli away from the k=1 singularity, to pin down the convention.
    """
    from scipy import special

    k = np.clip(k, 0.0, 1.0)
    m = k ** 2
    return float(special.ellipkm1(1.0 - m))


def onsager_energy(T, J=J_DEFAULT):
    """Exact internal energy per spin u(T) for the infinite 2D Ising lattice.

    u(T) = -J*coth(2K) * [1 + (2/pi)*(2*tanh(2K)^2 - 1)*K1(kappa)]
    with K = J/T, kappa = 2*sinh(2K)/cosh(2K)^2.
    """
    T = np.asarray(T, dtype=float)
    scalar_input = (T.ndim == 0)
    T = np.atleast_1d(T)
    out = np.empty_like(T)
    for idx, Ti in enumerate(T):
        K = J / Ti
        kappa = 2.0 * np.sinh(2 * K) / np.cosh(2 * K) ** 2
        coeff = 2.0 * np.tanh(2 * K) ** 2 - 1.0
        if kappa >= 1.0 - 1e-9:
            # At T=T_c, kappa=1 exactly and K1(kappa) diverges logarithmically,
            # but coeff -> 0 there too (tanh(2K_c)^2 -> 1/2 exactly), and the
            # product coeff*K1 -> 0 in the limit (coeff ~ (T-T_c), K1 ~ -ln|T-T_c|).
            # Evaluating both factors as floats this close to the singularity is
            # numerically unstable, so use the analytic limit directly.
            bracket = 1.0
        else:
            K1 = _complete_elliptic_K(kappa)
            bracket = 1.0 + (2.0 / np.pi) * coeff * K1
        out[idx] = -J / np.tanh(2 * K) * bracket
    return float(out[0]) if scalar_input else out


def onsager_magnetization(T, J=J_DEFAULT, Tc=None):
    """Exact spontaneous magnetization per spin (Yang, 1952) for T < T_c, else 0.

    m(T) = [1 - sinh(2J/T)^-4] ^ (1/8)   for T < T_c
    m(T) = 0                              for T >= T_c
    """
    if Tc is None:
        Tc = T_C
    T = np.asarray(T, dtype=float)
    scalar_input = (T.ndim == 0)
    T = np.atleast_1d(T)
    out = np.zeros_like(T)
    below = T < Tc
    s = np.sinh(2 * J / T[below])
    inner = np.clip(1.0 - s ** -4, 0.0, None)
    out[below] = inner ** (1.0 / 8.0)
    return float(out[0]) if scalar_input else out
