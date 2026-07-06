import math

import pytest

from src.theory import (
    PREDICTED_EXPONENT,
    PREDICTED_M,
    collision_tester_predicted_m,
    naive_learner_predicted_m,
    paired_collision_probability,
)


def test_collision_predicted_m_scales_as_sqrt_n():
    m1 = collision_tester_predicted_m(100, 0.2)
    m2 = collision_tester_predicted_m(400, 0.2)
    # n quadruples -> sqrt(n) doubles
    assert m2 / m1 == pytest.approx(2.0, rel=1e-9)


def test_naive_learner_predicted_m_scales_linearly_in_n():
    m1 = naive_learner_predicted_m(100, 0.2)
    m2 = naive_learner_predicted_m(400, 0.2)
    assert m2 / m1 == pytest.approx(4.0, rel=1e-9)


def test_predicted_m_scales_as_inverse_epsilon_squared():
    m1 = collision_tester_predicted_m(100, 0.1)
    m2 = collision_tester_predicted_m(100, 0.2)
    assert m1 / m2 == pytest.approx(4.0, rel=1e-9)


def test_predicted_m_registry_matches_functions():
    assert PREDICTED_M["collision"](100, 0.2) == collision_tester_predicted_m(100, 0.2)
    assert PREDICTED_M["naive_learner"](100, 0.2) == naive_learner_predicted_m(100, 0.2)
    assert PREDICTED_EXPONENT["collision"] == 0.5
    assert PREDICTED_EXPONENT["naive_learner"] == 1.0


def test_paired_collision_probability_matches_manual_derivation():
    n, epsilon = 50, 0.25
    # Manual: half the mass at (1+2e)/n, half at (1-2e)/n.
    high, low = (1 + 2 * epsilon) / n, (1 - 2 * epsilon) / n
    expected = (n / 2) * high ** 2 + (n / 2) * low ** 2
    assert paired_collision_probability(n, epsilon) == pytest.approx(expected)


def test_paired_collision_probability_reduces_to_uniform_at_epsilon_zero():
    n = 80
    assert paired_collision_probability(n, 0.0) == pytest.approx(1.0 / n)


def test_paired_collision_probability_exceeds_uniform_for_positive_epsilon():
    n = 80
    assert paired_collision_probability(n, 0.1) > 1.0 / n
