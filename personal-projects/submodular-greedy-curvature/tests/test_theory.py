import math

import pytest

from src.theory import WORST_CASE_BOUND, curvature_bound


def test_worst_case_bound_value():
    assert WORST_CASE_BOUND == pytest.approx(1 - 1 / math.e)
    assert WORST_CASE_BOUND == pytest.approx(0.6321, abs=1e-4)


def test_curvature_bound_at_zero_is_one():
    assert curvature_bound(0.0) == pytest.approx(1.0)
    assert curvature_bound(1e-12) == pytest.approx(1.0, abs=1e-6)


def test_curvature_bound_at_one_equals_worst_case_bound():
    assert curvature_bound(1.0) == pytest.approx(WORST_CASE_BOUND, abs=1e-9)


def test_curvature_bound_is_monotonically_decreasing():
    xs = [i / 100 for i in range(101)]
    ys = [curvature_bound(x) for x in xs]
    assert all(a >= b - 1e-9 for a, b in zip(ys, ys[1:]))


def test_curvature_bound_stays_in_unit_interval():
    for c in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        b = curvature_bound(c)
        assert 0.0 <= b <= 1.0
