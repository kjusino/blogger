import numpy as np
import pytest

from src.network import (
    braess_network,
    pigou_network,
    three_parallel_links,
    two_link_pigou_style,
)
from src.solvers import price_of_anarchy, solve_equilibrium, solve_optimum, total_cost
from src.theory import poa_bound, two_link_equilibrium_cost, two_link_optimum_cost


def test_pigou_example_matches_the_textbook_four_thirds():
    r = price_of_anarchy(pigou_network())
    assert r["equilibrium_cost"] == pytest.approx(1.0, abs=1e-4)
    assert r["optimum_cost"] == pytest.approx(0.75, abs=1e-4)
    assert r["poa"] == pytest.approx(4.0 / 3.0, abs=1e-3)


def test_braess_paradox_shortcut_makes_everyone_worse_off():
    r_no = price_of_anarchy(braess_network(with_shortcut=False))
    r_yes = price_of_anarchy(braess_network(with_shortcut=True))
    assert r_no["equilibrium_cost"] == pytest.approx(1.5, abs=1e-4)
    assert r_yes["equilibrium_cost"] == pytest.approx(2.0, abs=1e-4)
    assert r_yes["equilibrium_cost"] > r_no["equilibrium_cost"]
    # the network *without* the shortcut is already socially optimal
    assert r_no["equilibrium_cost"] == pytest.approx(r_no["optimum_cost"], abs=1e-4)


def test_equilibrium_flow_satisfies_conservation():
    net = three_parallel_links()
    f = solve_equilibrium(net)
    A = net.incidence_matrix()
    b = net.conservation_rhs()
    assert np.allclose(A @ f, b, atol=1e-5)
    assert np.all(f >= -1e-8)


def test_equilibrium_condition_equal_latency_on_used_edges():
    """Wardrop equilibrium: all edges carrying flow have equal latency, and
    it is <= the latency any unused edge would have at zero flow."""
    net = three_parallel_links()
    f = solve_equilibrium(net)
    latencies = [e.cost(f[i]) for i, e in enumerate(net.edges)]
    used = [lat for lat, flow in zip(latencies, f) if flow > 1e-4]
    assert max(used) - min(used) < 1e-3
    for lat, flow in zip(latencies, f):
        if flow <= 1e-4:
            assert lat >= min(used) - 1e-3


def test_optimum_flow_satisfies_conservation():
    net = three_parallel_links()
    f = solve_optimum(net)
    A = net.incidence_matrix()
    b = net.conservation_rhs()
    assert np.allclose(A @ f, b, atol=1e-5)
    assert np.all(f >= -1e-8)


def test_optimum_cost_never_exceeds_equilibrium_cost():
    for net in [pigou_network(), three_parallel_links(),
                braess_network(with_shortcut=True), braess_network(with_shortcut=False)]:
        r = price_of_anarchy(net)
        assert r["optimum_cost"] <= r["equilibrium_cost"] + 1e-6


@pytest.mark.parametrize("p,b", [(1, 0.4), (1, 1.0), (2, 0.6), (3, 1.0), (4, 0.9)])
def test_generic_solver_matches_two_link_closed_form(p, b):
    net = two_link_pigou_style(p, b)
    r = price_of_anarchy(net)
    assert r["equilibrium_cost"] == pytest.approx(two_link_equilibrium_cost(p, b), abs=1e-3)
    assert r["optimum_cost"] == pytest.approx(two_link_optimum_cost(p, b), abs=1e-3)


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_poa_at_worst_case_b_hits_the_theoretical_bound(p):
    # b=1.0 is the worst-case constant for every degree (see theory.py docstring)
    net = two_link_pigou_style(p, 1.0)
    r = price_of_anarchy(net)
    assert r["poa"] == pytest.approx(poa_bound(p), abs=1e-3)


def test_total_cost_matches_manual_sum():
    net = three_parallel_links()
    f = np.array([0.5, 0.3, 0.2])
    manual = sum(e.cost.flow_cost(f[i]) for i, e in enumerate(net.edges))
    assert total_cost(net, f) == pytest.approx(manual)
