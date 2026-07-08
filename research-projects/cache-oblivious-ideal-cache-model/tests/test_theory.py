import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.theory import PREDICTED_EXPONENT, fit_power_law, leading_constant


def test_fit_power_law_recovers_known_exponent_and_constant():
    c, a = 3.5, 2.0
    xs = [1, 2, 4, 8, 16, 32, 64]
    ys = [c * (x ** a) for x in xs]
    fit = fit_power_law(xs, ys)
    assert fit.slope == pytest.approx(a, abs=1e-9)
    assert pytest.approx(c, rel=1e-9) == pow(2.718281828459045, fit.intercept)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
    assert fit.n_points == len(xs)


def test_fit_power_law_negative_slope():
    xs = [1, 2, 4, 8, 16]
    ys = [100.0 / x for x in xs]
    fit = fit_power_law(xs, ys)
    assert fit.slope == pytest.approx(-1.0, abs=1e-9)


def test_fit_power_law_requires_at_least_three_points():
    with pytest.raises(ValueError):
        fit_power_law([1, 2], [1.0, 2.0])


def test_fit_power_law_rejects_nonpositive_values():
    with pytest.raises(ValueError):
        fit_power_law([1, 2, -1], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        fit_power_law([1, 2, 3], [1.0, 0.0, 3.0])


def test_fit_result_within_tolerance():
    xs = [1, 2, 4, 8, 16, 32]
    ys = [1.0 * (x ** 3.05) for x in xs]  # slightly off a clean cube
    fit = fit_power_law(xs, ys)
    assert fit.within(3.0, tolerance=0.3)
    assert not fit.within(3.0, tolerance=0.01)


def test_leading_constant_naive_recovers_known_c():
    n, B, M = 100, 8, 4096
    c_true = 1.3
    misses = c_true * (n ** 3) / B
    got = leading_constant(n, B, M, misses, "naive")
    assert got == pytest.approx(c_true)


def test_leading_constant_oblivious_recovers_known_c():
    n, B, M = 100, 8, 4096
    c_true = 2.7
    misses = c_true * (n ** 3) / (B * (M ** 0.5))
    got = leading_constant(n, B, M, misses, "oblivious")
    assert got == pytest.approx(c_true)


def test_leading_constant_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        leading_constant(10, 4, 64, 100.0, "quantum")


def test_predicted_exponent_table_has_no_naive_M_entry():
    # naive's M-scaling is a step function, not a power law -- it must
    # never be silently graded against a slope prediction.
    assert ("M", "naive") not in PREDICTED_EXPONENT


@pytest.mark.parametrize(
    "key,expected",
    [
        (("n", "naive"), 3.0),
        (("n", "blocked"), 3.0),
        (("n", "oblivious"), 3.0),
        (("B", "naive"), -1.0),
        (("B", "blocked"), -1.0),
        (("B", "oblivious"), -1.0),
        (("M", "blocked"), -0.5),
        (("M", "oblivious"), -0.5),
    ],
)
def test_predicted_exponent_values(key, expected):
    assert PREDICTED_EXPONENT[key] == expected
