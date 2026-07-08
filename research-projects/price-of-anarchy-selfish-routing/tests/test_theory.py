import pytest

from src.theory import (
    poa_bound,
    two_link_equilibrium_cost,
    two_link_optimum_cost,
    two_link_poa,
    worst_case_two_link_poa,
)


def test_poa_bound_rejects_degree_below_one():
    with pytest.raises(ValueError):
        poa_bound(0)


def test_poa_bound_p1_is_exactly_four_thirds():
    assert poa_bound(1) == pytest.approx(4.0 / 3.0, abs=1e-12)


def test_poa_bound_is_increasing_in_degree():
    values = [poa_bound(p) for p in range(1, 8)]
    assert all(values[i] < values[i + 1] for i in range(len(values) - 1))


def test_poa_bound_p_infinity_diverges_slowly():
    # sanity: even at large p the bound stays finite and grows roughly like p/ln(p)
    assert poa_bound(50) > poa_bound(10) > poa_bound(1)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 6])
def test_worst_case_two_link_matches_closed_form_bound(p):
    """Independent from-scratch numeric derivation must agree with the
    closed-form Roughgarden formula -- this is the core cross-check that
    the hardcoded beta(p) formula is actually correct."""
    _, best_poa = worst_case_two_link_poa(p)
    assert best_poa == pytest.approx(poa_bound(p), abs=1e-6)


def test_two_link_poa_is_never_below_one():
    for p in [1, 2, 3]:
        for b in [0.0, 0.3, 0.9, 1.0, 1.5, 4.0]:
            assert two_link_poa(p, b) >= 1.0 - 1e-9


def test_two_link_equilibrium_cost_matches_hand_derivation_at_b_below_one():
    # b=0.5, p=1: equilibrium splits so both links cost 0.5 -> total cost 0.5
    assert two_link_equilibrium_cost(1, 0.5) == pytest.approx(0.5)


def test_two_link_equilibrium_cost_saturates_above_one():
    # for b >= 1 all flow uses edge 1 (cost x^p at x=1 is 1 < b)
    assert two_link_equilibrium_cost(2, 3.0) == pytest.approx(1.0)
    assert two_link_equilibrium_cost(2, 1.0) == pytest.approx(1.0)


def test_two_link_optimum_beats_equilibrium_pigou_case():
    # classic Pigou instance: p=1, b=1 -> equilibrium 1.0, optimum 0.75
    assert two_link_equilibrium_cost(1, 1.0) == pytest.approx(1.0)
    assert two_link_optimum_cost(1, 1.0) == pytest.approx(0.75)
    assert two_link_poa(1, 1.0) == pytest.approx(4.0 / 3.0)


def test_two_link_optimum_never_exceeds_equilibrium():
    for p in [1, 2, 3, 4]:
        for b in [0.1, 0.5, 1.0, 2.0, p + 1.0]:
            assert two_link_optimum_cost(p, b) <= two_link_equilibrium_cost(p, b) + 1e-9
