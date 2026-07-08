"""Discrete-time stochastic SIS epidemic simulation on a network, with the
quasi-stationary (QS) reinjection method for measuring the endemic state on
a finite graph without falling into the (always-absorbing, for finite N)
disease-free state.

Model (synchronous, discrete-time SIS -- see Chakrabarti, Wang, Wang, Leskovec,
Faloutsos, "Epidemic Thresholds in Real Networks", ACM TISSEC 2008, and Van
Mieghem's discrete-time NIMFA formulation): at each step, given the infection
state I(t) in {0,1}^n,

  - each infected node i recovers independently with probability delta
  - each susceptible node i with n_i(t) infected neighbors becomes infected
    independently with probability 1 - (1-beta)^{n_i(t)}

both applied using I(t) (synchronous / parallel update, not sequential). For
small beta, delta this is a first-order-accurate discretization of the
continuous-time SIS contact process and shares its linear-stability epidemic
threshold beta_c/delta = 1/lambda_max(A) (QMF) at leading order.

QS method (Ferreira, Sander, Boguna, "Epidemic thresholds of the
susceptible-infected-susceptible model on networks: A comparison of
numerical and theoretical results", Phys. Rev. E 86, 041125 (2012)): a small
rolling buffer of previously-visited *active* configurations is maintained;
whenever the epidemic goes extinct (I(t) = 0), the state is replaced by a
uniformly-random configuration drawn from that buffer, instead of allowing
the chain to sit in the absorbing state. This produces a stationary
distribution conditioned on survival (the quasi-stationary distribution)
whose mean prevalence and its fluctuations are the standard observables used
to locate the epidemic threshold numerically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


def sis_step(A: sp.spmatrix, infected: np.ndarray, beta: float, delta: float,
             rng: np.random.Generator) -> np.ndarray:
    """One synchronous SIS update. `infected` is a bool array of shape (n,)."""
    infected_f = infected.astype(float)
    n_infected_neighbors = A @ infected_f
    p_infect = 1.0 - (1.0 - beta) ** n_infected_neighbors

    susceptible = ~infected
    new_infections = susceptible & (rng.random(infected.shape[0]) < p_infect)
    recoveries = infected & (rng.random(infected.shape[0]) < delta)

    return (infected & ~recoveries) | new_infections


@dataclass
class QSResult:
    mean_rho: float
    var_rho: float
    susceptibility: float  # N * Var(rho) / Mean(rho)
    rho_series: np.ndarray
    n_reinjections: int


def run_qs_simulation(A: sp.spmatrix, beta: float, delta: float, n_steps: int,
                       burn_in: int, seed: int, initial_frac: float = 0.2,
                       buffer_size: int = 30, buffer_update_interval: int = 10) -> QSResult:
    """Run one QS-method SIS trajectory and return prevalence statistics
    over the post-burn-in window."""
    n = A.shape[0]
    rng = np.random.default_rng(seed)

    infected = rng.random(n) < initial_frac
    if not infected.any():
        infected[rng.integers(n)] = True

    buffer: list[np.ndarray] = [infected.copy()]
    rho_series = np.empty(n_steps - burn_in if n_steps > burn_in else 0)
    n_reinjections = 0
    write_idx = 0

    for t in range(n_steps):
        infected = sis_step(A, infected, beta, delta, rng)

        if not infected.any():
            n_reinjections += 1
            infected = buffer[rng.integers(len(buffer))].copy()

        if t % buffer_update_interval == 0 and infected.any():
            if len(buffer) < buffer_size:
                buffer.append(infected.copy())
            else:
                buffer[rng.integers(buffer_size)] = infected.copy()

        if t >= burn_in:
            rho_series[write_idx] = infected.sum() / n
            write_idx += 1

    mean_rho = float(rho_series.mean())
    var_rho = float(rho_series.var(ddof=1))
    susceptibility = (n * var_rho / mean_rho) if mean_rho > 0 else float("nan")

    return QSResult(
        mean_rho=mean_rho,
        var_rho=var_rho,
        susceptibility=susceptibility,
        rho_series=rho_series,
        n_reinjections=n_reinjections,
    )


def run_repeated_qs(A: sp.spmatrix, beta: float, delta: float, n_steps: int,
                     burn_in: int, n_repeats: int, seed: int,
                     **kwargs) -> list[QSResult]:
    """Independent QS runs (fresh RNG stream per repeat) for estimating the
    run-to-run variability of the QS observables at one (network, beta)."""
    return [
        run_qs_simulation(A, beta, delta, n_steps, burn_in, seed=seed * 1_000_003 + r, **kwargs)
        for r in range(n_repeats)
    ]
