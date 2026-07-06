import numpy as np
import pytest

from src.distributions import (
    FAMILIES,
    block_perturbation,
    paired_perturbation,
    tv_distance,
    uniform,
)


def test_uniform_sums_to_one_and_is_flat():
    p = uniform(50)
    assert p.shape == (50,)
    assert np.isclose(p.sum(), 1.0)
    assert np.allclose(p, p[0])


def test_uniform_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        uniform(0)


def test_tv_distance_zero_for_identical_distributions():
    p = uniform(20)
    assert tv_distance(p, p) == pytest.approx(0.0)


@pytest.mark.parametrize("n", [10, 50, 200])
@pytest.mark.parametrize("epsilon", [0.0, 0.05, 0.2, 0.4])
def test_paired_perturbation_has_exact_tv_distance(n, epsilon):
    rng = np.random.default_rng(0)
    p = paired_perturbation(n, epsilon, rng)
    assert np.isclose(p.sum(), 1.0)
    assert np.all(p >= 0)
    assert tv_distance(p, uniform(n)) == pytest.approx(epsilon, abs=1e-9)


def test_paired_perturbation_rejects_odd_n():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        paired_perturbation(11, 0.1, rng)


def test_paired_perturbation_rejects_out_of_range_epsilon():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        paired_perturbation(10, 0.6, rng)


@pytest.mark.parametrize("n,block_size", [(20, 1), (20, 5), (100, 25)])
@pytest.mark.parametrize("epsilon", [0.05, 0.2])
def test_block_perturbation_has_exact_tv_distance(n, block_size, epsilon):
    rng = np.random.default_rng(1)
    p = block_perturbation(n, epsilon, block_size, rng)
    assert np.isclose(p.sum(), 1.0)
    assert np.all(p >= -1e-12)
    assert tv_distance(p, uniform(n)) == pytest.approx(epsilon, abs=1e-9)


def test_block_perturbation_rejects_invalid_block_size():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        block_perturbation(10, 0.1, 0, rng)
    with pytest.raises(ValueError):
        block_perturbation(10, 0.1, 10, rng)


def test_block_perturbation_rejects_epsilon_too_large_for_block():
    rng = np.random.default_rng(0)
    # block_size=1, n=10 => low = 1/10 - epsilon/9, which goes negative once
    # epsilon > 0.9.
    with pytest.raises(ValueError):
        block_perturbation(10, 0.95, 1, rng)


def test_families_registry_all_produce_valid_epsilon_far_distributions():
    rng = np.random.default_rng(2)
    n, epsilon = 40, 0.15
    for name, factory in FAMILIES.items():
        p = factory(n, epsilon, rng)
        assert np.isclose(p.sum(), 1.0), name
        assert np.all(p >= 0), name
        assert tv_distance(p, uniform(n)) == pytest.approx(epsilon, abs=1e-9), name


def test_paired_perturbation_permutes_across_draws():
    rng = np.random.default_rng(3)
    p1 = paired_perturbation(20, 0.2, rng)
    p2 = paired_perturbation(20, 0.2, rng)
    # Same multiset of values, but (almost certainly) different arrangement.
    assert np.allclose(np.sort(p1), np.sort(p2))
    assert not np.allclose(p1, p2)
