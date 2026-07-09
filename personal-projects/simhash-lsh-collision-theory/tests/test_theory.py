import math

import pytest

from src import theory


def test_single_hash_collision_prob_endpoints():
    assert theory.single_hash_collision_prob(0.0) == pytest.approx(1.0)
    assert theory.single_hash_collision_prob(math.pi) == pytest.approx(0.0)


def test_single_hash_collision_prob_midpoint():
    # Orthogonal vectors: a random hyperplane agrees exactly half the time.
    assert theory.single_hash_collision_prob(math.pi / 2) == pytest.approx(0.5)


def test_single_hash_collision_prob_monotonic_decreasing():
    thetas = [i * math.pi / 20 for i in range(21)]
    probs = [theory.single_hash_collision_prob(t) for t in thetas]
    assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))


def test_single_hash_collision_prob_rejects_out_of_range():
    with pytest.raises(ValueError):
        theory.single_hash_collision_prob(-0.1)
    with pytest.raises(ValueError):
        theory.single_hash_collision_prob(math.pi + 0.1)


def test_banded_collision_prob_k1_matches_single_hash():
    theta = 0.7
    assert theory.banded_collision_prob(theta, 1) == pytest.approx(
        theory.single_hash_collision_prob(theta)
    )


def test_banded_collision_prob_is_power_of_single_hash():
    theta = 1.1
    k = 5
    expected = theory.single_hash_collision_prob(theta) ** k
    assert theory.banded_collision_prob(theta, k) == pytest.approx(expected)


def test_or_of_bands_prob_l1_matches_banded():
    theta, k = 0.9, 4
    assert theory.or_of_bands_prob(theta, k, 1) == pytest.approx(
        theory.banded_collision_prob(theta, k)
    )


def test_or_of_bands_prob_increases_with_L():
    theta, k = 0.9, 4
    p_small_L = theory.or_of_bands_prob(theta, k, 5)
    p_large_L = theory.or_of_bands_prob(theta, k, 50)
    assert p_large_L > p_small_L


def test_or_of_bands_prob_bounds():
    for theta in [0.1, 0.5, 1.0, 2.0, 3.0]:
        p = theory.or_of_bands_prob(theta, k=6, L=15)
        assert 0.0 <= p <= 1.0


def test_threshold_similarity_self_consistent():
    """Plugging threshold_similarity(k, L) back into the banded/OR formula
    should give recall exactly 0.5, by construction of the closed form."""
    for k, L in [(1, 1), (4, 8), (8, 20), (10, 50)]:
        s_star = theory.threshold_similarity(k, L)
        recall_at_threshold = 1.0 - (1.0 - s_star**k) ** L
        assert recall_at_threshold == pytest.approx(0.5, abs=1e-9)


def test_threshold_angle_consistent_with_threshold_similarity():
    k, L = 8, 20
    theta_star = theory.threshold_angle(k, L)
    s_star = theory.threshold_similarity(k, L)
    assert theta_star == pytest.approx(math.pi * (1.0 - s_star))
    recall = theory.or_of_bands_prob(theta_star, k, L)
    assert recall == pytest.approx(0.5, abs=1e-9)


def test_rho_exponent_known_values():
    # p1 = p2 => infinite/ill-defined discrimination; instead check a clean case:
    # p1 = 0.5, p2 = 0.25 => rho = ln(2)/ln(4) = 0.5
    assert theory.rho_exponent(0.5, 0.25) == pytest.approx(0.5)


def test_rho_exponent_requires_p1_greater_than_p2():
    with pytest.raises(ValueError):
        theory.rho_exponent(0.2, 0.5)
    with pytest.raises(ValueError):
        theory.rho_exponent(0.5, 0.5)


def test_rho_exponent_in_unit_interval_for_typical_inputs():
    # For LSH families with the "monotonicity" property (p1 > p2 as
    # similarity decreases), rho in (0, 1) is expected for well-separated
    # near/far thresholds.
    rho = theory.rho_exponent(p1=0.9, p2=0.3)
    assert 0.0 < rho < 1.0
