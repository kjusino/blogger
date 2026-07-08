import math

import pytest

from ramanujan_spectra.theory import (
    alon_boppana_bound,
    close_to_bound,
    is_ramanujan,
    within_epsilon_of_bound,
)


def test_alon_boppana_bound_known_values():
    assert alon_boppana_bound(2) == pytest.approx(2.0)
    assert alon_boppana_bound(3) == pytest.approx(2 * math.sqrt(2))
    assert alon_boppana_bound(5) == pytest.approx(4.0)
    assert alon_boppana_bound(10) == pytest.approx(6.0)


def test_alon_boppana_bound_rejects_small_d():
    with pytest.raises(ValueError):
        alon_boppana_bound(1)
    with pytest.raises(ValueError):
        alon_boppana_bound(0)


def test_is_ramanujan_petersen_graph():
    # Petersen graph: d=3, lambda(G) = 2 <= 2*sqrt(2) ~= 2.828
    assert is_ramanujan(2.0, 3)


def test_is_ramanujan_complete_bipartite_is_not():
    # K_{d,d}: lambda(G) = d, strictly above 2*sqrt(d-1) for d >= 2
    for d in (3, 5, 8):
        assert not is_ramanujan(float(d), d)


def test_is_ramanujan_exactly_at_bound():
    bound = alon_boppana_bound(6)
    assert is_ramanujan(bound, 6)
    assert is_ramanujan(bound - 1e-3, 6)
    assert not is_ramanujan(bound + 1e-3, 6)


def test_within_epsilon_of_bound():
    bound = alon_boppana_bound(4)
    assert within_epsilon_of_bound(bound, 4, eps=0.0)
    assert within_epsilon_of_bound(bound - 5.0, 4, eps=0.0)  # far below is still "within"
    assert not within_epsilon_of_bound(bound + 0.1, 4, eps=0.05)
    assert within_epsilon_of_bound(bound + 0.1, 4, eps=0.2)


def test_close_to_bound_is_two_sided():
    bound = alon_boppana_bound(4)
    assert close_to_bound(bound, 4, eps=0.0)
    assert close_to_bound(bound + 0.05, 4, eps=0.1)
    assert close_to_bound(bound - 0.05, 4, eps=0.1)
    assert not close_to_bound(bound - 5.0, 4, eps=0.1)  # far below is NOT "close"
    assert not close_to_bound(bound + 5.0, 4, eps=0.1)
