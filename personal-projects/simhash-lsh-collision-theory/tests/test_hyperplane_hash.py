import math

import numpy as np
import pytest

from src import theory
from src.data import random_unit_vector, vector_at_angle
from src.hyperplane_hash import (
    bits_to_int,
    empirical_single_bit_collision_rate,
    hash_bits,
    random_hyperplanes,
)


def test_random_hyperplanes_shape():
    rng = np.random.default_rng(0)
    planes = random_hyperplanes(k=7, dim=13, rng=rng)
    assert planes.shape == (7, 13)


def test_hash_bits_deterministic_given_same_hyperplanes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 10))
    planes = random_hyperplanes(k=4, dim=10, rng=rng)
    bits1 = hash_bits(X, planes)
    bits2 = hash_bits(X, planes)
    np.testing.assert_array_equal(bits1, bits2)


def test_hash_bits_shape_and_dtype():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((6, 10))
    planes = random_hyperplanes(k=3, dim=10, rng=rng)
    bits = hash_bits(X, planes)
    assert bits.shape == (6, 3)
    assert bits.dtype == bool


def test_identical_vector_always_matches_itself():
    rng = np.random.default_rng(2)
    u = random_unit_vector(20, rng)
    planes = random_hyperplanes(k=16, dim=20, rng=rng)
    bits_u = hash_bits(u[None, :], planes)
    bits_u_again = hash_bits(u[None, :], planes)
    np.testing.assert_array_equal(bits_u, bits_u_again)


def test_bits_to_int_distinguishes_patterns():
    bits = np.array([[True, False], [False, True], [True, False]])
    keys = bits_to_int(bits)
    assert keys[0] == keys[2]
    assert keys[0] != keys[1]


def test_bits_to_int_matches_manual_binary_encoding():
    bits = np.array([[True, True, False], [False, False, True]])
    keys = bits_to_int(bits)
    # bit i contributes 2**i regardless of ordering convention, as long as
    # it is applied consistently; check internal self-consistency instead
    # of a hardcoded convention.
    assert keys[0] == (1 << 0) + (1 << 1)
    assert keys[1] == (1 << 2)


@pytest.mark.parametrize("theta", [0.3, math.pi / 2, 2.5])
def test_empirical_single_bit_collision_matches_theory(theta):
    """Statistical test: with enough trials, the empirical collision rate
    should land within a small number of standard errors of the closed-form
    theory. Uses a fixed seed so this is deterministic, not flaky."""
    rng = np.random.default_rng(42)
    dim = 30
    u = random_unit_vector(dim, rng)
    v = vector_at_angle(u, theta, rng)
    p_hat, stderr = empirical_single_bit_collision_rate(u, v, num_trials=50000, rng=rng)
    p_theory = theory.single_hash_collision_prob(theta)
    z = abs(p_hat - p_theory) / stderr if stderr > 0 else abs(p_hat - p_theory)
    assert z < 5.0, f"theta={theta}: empirical={p_hat}, theory={p_theory}, z={z}"


def test_empirical_collision_rate_is_scale_invariant_to_vector_norm():
    """Sign(r.x) only depends on direction, not magnitude, so scaling u, v
    by different positive constants must not change the collision rate."""
    rng = np.random.default_rng(3)
    dim = 15
    u = random_unit_vector(dim, rng)
    v = vector_at_angle(u, 1.0, rng)
    planes = random_hyperplanes(k=200, dim=dim, rng=rng)
    bits_u = hash_bits(u[None, :], planes)
    bits_scaled = hash_bits((5.0 * u)[None, :], planes)
    np.testing.assert_array_equal(bits_u, bits_scaled)
