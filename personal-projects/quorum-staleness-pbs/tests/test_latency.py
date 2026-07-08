import numpy as np
import pytest

from pbs.latency import heterogeneous_model, homogeneous_model


def test_homogeneous_exponential_mean_matches_theory():
    rng = np.random.default_rng(0)
    model = homogeneous_model(n_replicas=4, family="exponential", rate=2.0)
    samples = model.sample(200_000, rng)
    assert samples.shape == (200_000, 4)
    assert np.isclose(samples.mean(), 0.5, rtol=0.02)


def test_homogeneous_lognormal_mean_matches_theory():
    rng = np.random.default_rng(1)
    model = homogeneous_model(n_replicas=4, family="lognormal", rate=2.0, sigma=0.8)
    samples = model.sample(300_000, rng)
    # mean-matched by construction to 1/rate regardless of sigma
    assert np.isclose(samples.mean(), 0.5, rtol=0.03)


def test_homogeneous_multipliers_all_ones():
    model = homogeneous_model(n_replicas=5, family="exponential", rate=1.0)
    assert np.array_equal(model.multipliers, np.ones(5))


def test_heterogeneous_sigma_zero_equals_homogeneous():
    rng = np.random.default_rng(2)
    model = heterogeneous_model(n_replicas=6, family="exponential", rate=1.5, sigma_het=0.0, rng=rng)
    assert np.array_equal(model.multipliers, np.ones(6))


def test_heterogeneous_multipliers_are_persistent_across_samples():
    rng = np.random.default_rng(3)
    model = heterogeneous_model(n_replicas=6, family="exponential", rate=1.0, sigma_het=0.5, rng=rng)
    first_multipliers = model.multipliers.copy()
    model.sample(10, rng)
    model.sample(10, rng)
    assert np.array_equal(model.multipliers, first_multipliers)


def test_heterogeneous_multipliers_geometric_mean_near_one():
    rng = np.random.default_rng(4)
    model = heterogeneous_model(n_replicas=2000, family="exponential", rate=1.0, sigma_het=0.4, rng=rng)
    log_mult = np.log(model.multipliers)
    assert np.isclose(log_mult.mean(), 0.0, atol=0.05)


def test_invalid_family_rejected():
    with pytest.raises(ValueError):
        homogeneous_model(n_replicas=3, family="gaussian", rate=1.0)


def test_nonpositive_rate_rejected():
    with pytest.raises(ValueError):
        homogeneous_model(n_replicas=3, family="exponential", rate=0.0)


def test_negative_sigma_het_rejected():
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError):
        heterogeneous_model(n_replicas=3, family="exponential", rate=1.0, sigma_het=-0.1, rng=rng)
