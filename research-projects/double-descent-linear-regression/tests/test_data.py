import numpy as np

from src.data import generate_beta0, sample_dataset


def test_generate_beta0_norm():
    beta0 = generate_beta0(p=50, r2=3.0)
    assert beta0.shape == (50,)
    assert np.isclose(np.sum(beta0 ** 2), 3.0)


def test_generate_beta0_rejects_invalid_p():
    try:
        generate_beta0(p=0, r2=1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_sample_dataset_shapes():
    rng = np.random.default_rng(0)
    beta0 = generate_beta0(p=10, r2=1.0)
    X, y = sample_dataset(n=30, p=10, beta0=beta0, sigma2=0.5, rng=rng)
    assert X.shape == (30, 10)
    assert y.shape == (30,)


def test_sample_dataset_noise_variance_matches_sigma2():
    rng = np.random.default_rng(1)
    p = 5
    beta0 = np.zeros(p)  # zero signal isolates the noise term
    n = 20000
    _X, y = sample_dataset(n=n, p=p, beta0=beta0, sigma2=2.0, rng=rng)
    assert abs(np.var(y) - 2.0) < 0.1


def test_sample_dataset_rejects_dimension_mismatch():
    rng = np.random.default_rng(2)
    beta0 = generate_beta0(p=5, r2=1.0)
    try:
        sample_dataset(n=10, p=6, beta0=beta0, sigma2=1.0, rng=rng)
        assert False, "expected ValueError"
    except ValueError:
        pass
