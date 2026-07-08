import networkx as nx
import numpy as np
import pytest

from src.sis_simulation import run_qs_simulation, sis_step


def _complete_graph_csr(n):
    G = nx.complete_graph(n)
    return nx.to_scipy_sparse_array(G, format="csr", dtype=float)


def test_sis_step_disease_free_state_stays_disease_free():
    A = _complete_graph_csr(10)
    infected = np.zeros(10, dtype=bool)
    rng = np.random.default_rng(0)
    for _ in range(5):
        infected = sis_step(A, infected, beta=0.9, delta=0.1, rng=rng)
    assert not infected.any()


def test_sis_step_zero_beta_full_delta_causes_deterministic_recovery():
    A = _complete_graph_csr(10)
    infected = np.ones(10, dtype=bool)
    rng = np.random.default_rng(0)
    new_infected = sis_step(A, infected, beta=0.0, delta=1.0, rng=rng)
    assert not new_infected.any()


def test_sis_step_beta_one_delta_zero_infects_everyone_next_step():
    # On a complete graph, every susceptible node has >=1 infected neighbor
    # as soon as one node is infected; beta=1 makes p_infect = 1 exactly
    # (1 - (1-1)^n = 1 for n >= 1), so infection is deterministic here.
    n = 8
    A = _complete_graph_csr(n)
    infected = np.zeros(n, dtype=bool)
    infected[0] = True
    rng = np.random.default_rng(0)
    new_infected = sis_step(A, infected, beta=1.0, delta=0.0, rng=rng)
    assert new_infected.all()


def test_qs_simulation_produces_positive_prevalence_above_threshold():
    G = nx.random_regular_graph(6, 200, seed=1)
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    # tau = beta/delta = 1/6 well above QMF threshold ~1/lambda_max ~1/6... use
    # a comfortably supercritical tau to avoid flakiness near threshold.
    delta = 0.2
    beta = 0.5 * delta  # tau = 0.5, several x the QMF threshold (~1/6=0.167)
    result = run_qs_simulation(A, beta, delta, n_steps=1500, burn_in=500, seed=42)
    assert len(result.rho_series) == 1000
    assert result.mean_rho > 0.05


def test_qs_simulation_deep_subcritical_gives_low_prevalence():
    G = nx.random_regular_graph(6, 200, seed=1)
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    delta = 0.2
    beta = 0.02 * delta  # tau = 0.02, far below the QMF threshold (~0.167)
    result = run_qs_simulation(A, beta, delta, n_steps=1500, burn_in=500, seed=42)
    assert result.mean_rho < 0.15


def test_qs_reinjection_recovers_deterministically_from_guaranteed_extinction():
    # delta=1, beta=0 => every infected node recovers next step with
    # certainty and no new infections are ever possible, so the epidemic
    # goes extinct on literally every step and QS reinjection must fire
    # every single step, always restoring the one configuration ever
    # stored in the buffer (the initial state).
    G = nx.random_regular_graph(4, 30, seed=0)
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    result = run_qs_simulation(A, beta=0.0, delta=1.0, n_steps=50, burn_in=10, seed=7,
                                initial_frac=0.3, buffer_size=5, buffer_update_interval=1)
    assert result.n_reinjections == 50
    # every post-burn-in prevalence reading equals the (constant) reinjected fraction
    assert np.allclose(result.rho_series, result.rho_series[0])
    assert result.mean_rho > 0
