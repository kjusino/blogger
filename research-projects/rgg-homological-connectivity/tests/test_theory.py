import math

import pytest

from src.theory import penrose_threshold_radius, threshold_ratio


def test_penrose_threshold_matches_closed_form():
    n = 100
    expected = math.sqrt(math.log(n) / (math.pi * n))
    assert penrose_threshold_radius(n) == pytest.approx(expected)


def test_penrose_threshold_decreases_as_n_grows():
    # Not monotone in principle for tiny n, but should be for any reasonable range.
    values = [penrose_threshold_radius(n) for n in [50, 100, 500, 1000, 5000]]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_penrose_threshold_rejects_n_leq_one():
    with pytest.raises(ValueError):
        penrose_threshold_radius(1)
    with pytest.raises(ValueError):
        penrose_threshold_radius(0)


def test_threshold_ratio_is_one_when_r_equals_theory():
    n = 200
    r_c = penrose_threshold_radius(n)
    assert threshold_ratio(r_c, n) == pytest.approx(1.0)


def test_threshold_ratio_scales_linearly_with_r():
    n = 200
    r_c = penrose_threshold_radius(n)
    assert threshold_ratio(2 * r_c, n) == pytest.approx(2.0)
