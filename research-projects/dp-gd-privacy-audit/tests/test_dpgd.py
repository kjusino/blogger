import numpy as np

from src.data import make_dataset
from src.dpgd import clip_gradients, train_dpgd


def test_clipping_caps_huge_gradient_norm():
    # Construct a single huge-magnitude example whose raw gradient norm is
    # enormous, and confirm clipping caps it at exactly C (not the raw norm).
    C = 1.0
    grads = np.array([[1000.0, -2000.0, 500.0]])
    raw_norm = np.linalg.norm(grads[0])
    assert raw_norm > C  # sanity: raw norm is indeed huge

    clipped = clip_gradients(grads, C)
    clipped_norm = np.linalg.norm(clipped[0])
    assert np.isclose(clipped_norm, C)
    assert not np.isclose(clipped_norm, raw_norm)


def test_clipping_leaves_small_gradients_unchanged():
    C = 10.0
    grads = np.array([[0.1, 0.2, -0.1]])
    clipped = clip_gradients(grads, C)
    np.testing.assert_allclose(clipped, grads)


def test_zero_noise_canary_effect_is_exact_after_one_step():
    rng_data = np.random.default_rng(0)
    X, y = make_dataset(n=20, d=5, rng=rng_data)
    theta0 = np.zeros(5)
    C = 1.0
    lr = 0.1

    rng_in = np.random.default_rng(1)
    rng_out = np.random.default_rng(1)  # same seed; sigma=0 means noise is 0 regardless

    theta_in = train_dpgd(X, y, theta0, T=1, C=C, sigma=0.0, lr=lr, canary_in=True, rng=rng_in)
    theta_out = train_dpgd(X, y, theta0, T=1, C=C, sigma=0.0, lr=lr, canary_in=False, rng=rng_out)

    diff = theta_in - theta_out
    expected = np.zeros(5)
    expected[0] = -lr * C
    np.testing.assert_allclose(diff, expected, atol=1e-12)


def test_same_rng_seed_produces_identical_noise_draws():
    # Mechanics check: an rng initialized from the same seed produces
    # identical noise draws when run through the same sequence of calls.
    # (The actual audit in src/audit.py deliberately does NOT do this -- it
    # uses independent fresh randomness for IN and OUT worlds. This test is
    # only about reproducibility of the noise generation mechanism itself.)
    rng_data = np.random.default_rng(0)
    X, y = make_dataset(n=20, d=5, rng=rng_data)
    theta0 = np.zeros(5)

    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)

    theta_a = train_dpgd(X, y, theta0, T=3, C=1.0, sigma=1.0, lr=0.1, canary_in=True, rng=rng_a)
    theta_b = train_dpgd(X, y, theta0, T=3, C=1.0, sigma=1.0, lr=0.1, canary_in=True, rng=rng_b)

    np.testing.assert_array_equal(theta_a, theta_b)


def test_canary_in_vs_out_differ_with_noise():
    rng_data = np.random.default_rng(0)
    X, y = make_dataset(n=20, d=5, rng=rng_data)
    theta0 = np.zeros(5)

    rng_in = np.random.default_rng(5)
    rng_out = np.random.default_rng(6)
    theta_in = train_dpgd(X, y, theta0, T=2, C=1.0, sigma=0.5, lr=0.1, canary_in=True, rng=rng_in)
    theta_out = train_dpgd(X, y, theta0, T=2, C=1.0, sigma=0.5, lr=0.1, canary_in=False, rng=rng_out)
    assert not np.allclose(theta_in, theta_out)


def test_trajectory_shape():
    rng_data = np.random.default_rng(0)
    X, y = make_dataset(n=10, d=4, rng=rng_data)
    theta0 = np.zeros(4)
    rng = np.random.default_rng(2)
    theta_final, trajectory = train_dpgd(
        X, y, theta0, T=5, C=1.0, sigma=1.0, lr=0.1, canary_in=True, rng=rng, return_trajectory=True
    )
    assert trajectory.shape == (6, 4)
    np.testing.assert_array_equal(trajectory[-1], theta_final)
    np.testing.assert_array_equal(trajectory[0], theta0)
