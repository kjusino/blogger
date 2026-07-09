import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from meanfield import (
    chi1,
    correlation_length,
    correlation_map,
    critical_sigma_w2,
    fixed_point_c,
    fixed_point_q,
    gaussian_expectation,
    q_map,
    tanh,
    tanh_prime,
)


def test_gaussian_expectation_matches_known_moments():
    assert gaussian_expectation(lambda z: np.ones_like(z)) == pytest.approx(1.0, abs=1e-10)
    assert gaussian_expectation(lambda z: z) == pytest.approx(0.0, abs=1e-10)
    assert gaussian_expectation(lambda z: z ** 2) == pytest.approx(1.0, abs=1e-10)
    assert gaussian_expectation(lambda z: z ** 4) == pytest.approx(3.0, abs=1e-8)  # E[Z^4]=3 for N(0,1)


def test_linear_activation_fixed_point_matches_closed_form():
    # For phi(x) = x, q* solves q = sigma_w2 * q + sigma_b2 exactly:
    # q* = sigma_b2 / (1 - sigma_w2), valid whenever sigma_w2 < 1.
    identity = lambda x: x
    for sigma_w2, sigma_b2 in [(0.3, 0.5), (0.7, 1.0), (0.9, 0.1)]:
        q_star = fixed_point_q(sigma_w2, sigma_b2, identity)
        expected = sigma_b2 / (1 - sigma_w2)
        assert q_star == pytest.approx(expected, rel=1e-4)


def test_linear_activation_chi1_equals_sigma_w2():
    # phi'(x) = 1 everywhere for the identity, so chi_1 = sigma_w2 * E[1] = sigma_w2 exactly.
    identity_prime = lambda x: np.ones_like(x)
    for sigma_w2 in [0.2, 0.5, 0.9]:
        q_star = fixed_point_q(sigma_w2, 0.3, lambda x: x)
        assert chi1(q_star, sigma_w2, identity_prime) == pytest.approx(sigma_w2, rel=1e-6)


def test_tanh_zero_bias_critical_point_is_sigma_w2_equals_one():
    # Classic result (Poole et al. 2016): with sigma_b2=0, tanh's near-origin
    # behavior is linear (tanh'(0)=1), so q*=0 is the fixed point for any
    # sigma_w2 <= 1, and chi_1 = sigma_w2 exactly at that fixed point --
    # meaning the critical line crosses sigma_w2=1 exactly at sigma_b2=0.
    for sigma_w2 in [0.3, 0.6, 0.99]:
        q_star = fixed_point_q(sigma_w2, 0.0, tanh)
        assert q_star == pytest.approx(0.0, abs=1e-6)
        assert chi1(q_star, sigma_w2, tanh_prime) == pytest.approx(sigma_w2, abs=1e-4)

    sw2_crit = critical_sigma_w2(0.0, tanh, tanh_prime)
    assert sw2_crit == pytest.approx(1.0, abs=1e-2)


def test_correlation_length_diverges_at_criticality_and_is_positive():
    assert correlation_length(1.0) == np.inf
    assert correlation_length(0.5) > 0
    assert correlation_length(2.0) > 0
    # symmetric divergence: chi_1 = 1-eps and chi_1 = 1+eps give similar xi_c
    xi_ordered = correlation_length(1.0 - 1e-4)
    xi_chaotic = correlation_length(1.0 + 1e-4)
    assert xi_ordered == pytest.approx(xi_chaotic, rel=0.05)


def test_correlation_length_shrinks_further_from_criticality():
    # farther from chi_1=1 (in log space) means faster decay/growth, i.e.
    # shorter correlation length
    assert correlation_length(0.9) > correlation_length(0.5)
    assert correlation_length(1.1) > correlation_length(2.0)


def test_correlation_map_fixes_c_equals_one():
    # f(1) = 1 is an identity of the correlation map for any activation and
    # any (sigma_w2, sigma_b2): two identical inputs stay identical.
    for sigma_w2, sigma_b2 in [(0.8, 0.05), (1.9861, 0.1), (3.2, 0.1)]:
        q_star = fixed_point_q(sigma_w2, sigma_b2, tanh)
        assert correlation_map(1.0, q_star, sigma_w2, sigma_b2, tanh) == pytest.approx(1.0, abs=1e-6)


def test_ordered_phase_correlation_fixed_point_is_one():
    sigma_w2, sigma_b2 = 0.8, 0.05  # chi_1 < 1
    q_star = fixed_point_q(sigma_w2, sigma_b2, tanh)
    assert chi1(q_star, sigma_w2, tanh_prime) < 1.0
    c_star = fixed_point_c(q_star, sigma_w2, sigma_b2, tanh)
    assert c_star == pytest.approx(1.0, abs=1e-3)


def test_chaotic_phase_correlation_fixed_point_is_below_one():
    sigma_w2, sigma_b2 = 3.2, 0.1  # chi_1 > 1
    q_star = fixed_point_q(sigma_w2, sigma_b2, tanh)
    assert chi1(q_star, sigma_w2, tanh_prime) > 1.0
    c_star = fixed_point_c(q_star, sigma_w2, sigma_b2, tanh)
    assert 0.0 < c_star < 0.99


def test_q_map_is_monotonic_in_sigma_b2():
    q1 = q_map(0.5, 1.0, 0.1, tanh)
    q2 = q_map(0.5, 1.0, 0.5, tanh)
    assert q2 > q1


def test_critical_sigma_w2_increases_with_sigma_b2():
    # matches the qualitative shape of Poole et al. Fig. 2: larger bias
    # variance pushes the ordered/chaotic boundary to larger sigma_w2
    sw2_a = critical_sigma_w2(0.0, tanh, tanh_prime)
    sw2_b = critical_sigma_w2(0.2, tanh, tanh_prime)
    sw2_c = critical_sigma_w2(0.4, tanh, tanh_prime)
    assert sw2_a < sw2_b < sw2_c
