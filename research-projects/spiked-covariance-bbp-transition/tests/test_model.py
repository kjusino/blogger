import numpy as np
import pytest

from bbp_transition.model import (
    make_spike_direction,
    sample_spiked_covariance_data,
    sample_covariance,
    top_eigenpair,
)


def test_spike_direction_is_unit_norm():
    rng = np.random.default_rng(0)
    v = make_spike_direction(10, rng)
    assert v.shape == (10,)
    assert np.linalg.norm(v) == pytest.approx(1.0)


def test_sample_shapes():
    rng = np.random.default_rng(0)
    X, v = sample_spiked_covariance_data(n=100, p=5, lam=1.0, rng=rng)
    assert X.shape == (100, 5)
    assert v.shape == (5,)
    assert np.linalg.norm(v) == pytest.approx(1.0)


def test_rejects_negative_lambda():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_spiked_covariance_data(n=10, p=3, lam=-0.5, rng=rng)


def test_rejects_non_unit_v():
    rng = np.random.default_rng(0)
    bad_v = np.array([1.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        sample_spiked_covariance_data(n=10, p=3, lam=1.0, rng=rng, v=bad_v)


def test_reproducible_with_seeded_rng():
    v_fixed = make_spike_direction(4, np.random.default_rng(42))
    X1, _ = sample_spiked_covariance_data(n=50, p=4, lam=0.5, rng=np.random.default_rng(7), v=v_fixed)
    X2, _ = sample_spiked_covariance_data(n=50, p=4, lam=0.5, rng=np.random.default_rng(7), v=v_fixed)
    np.testing.assert_array_equal(X1, X2)


def test_no_spike_recovers_identity_covariance_at_large_n():
    """lam=0 means Sigma=I; the sample covariance should concentrate around I."""
    rng = np.random.default_rng(1)
    p = 5
    X, _ = sample_spiked_covariance_data(n=200_000, p=p, lam=0.0, rng=rng)
    S = sample_covariance(X)
    np.testing.assert_allclose(S, np.eye(p), atol=0.02)


def test_spiked_covariance_matches_population_at_large_n():
    """The empirical covariance should converge to Sigma = I + lam v v^T."""
    rng = np.random.default_rng(2)
    p = 5
    lam = 3.0
    X, v = sample_spiked_covariance_data(n=500_000, p=p, lam=lam, rng=rng)
    S = sample_covariance(X)
    Sigma = np.eye(p) + lam * np.outer(v, v)
    np.testing.assert_allclose(S, Sigma, atol=0.05)


def test_top_eigenpair_matches_numpy_reference():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((6, 6))
    S = A @ A.T
    eig, vec = top_eigenpair(S)
    eigvals_ref = np.linalg.eigvalsh(S)
    assert eig == pytest.approx(eigvals_ref[-1])
    # eigenvector equation: S v = eig v
    np.testing.assert_allclose(S @ vec, eig * vec, atol=1e-8)


def test_top_eigenpair_eigenvector_is_unit_norm():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((8, 8))
    S = A @ A.T
    _, vec = top_eigenpair(S)
    assert np.linalg.norm(vec) == pytest.approx(1.0)
