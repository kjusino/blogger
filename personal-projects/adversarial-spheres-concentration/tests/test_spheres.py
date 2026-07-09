import numpy as np
import pytest

from src.spheres import sample_sphere, make_dataset


def test_sample_sphere_has_correct_radius():
    rng = np.random.default_rng(0)
    x = sample_sphere(500, 10, radius=2.5, rng=rng)
    norms = np.linalg.norm(x, axis=1)
    assert np.allclose(norms, 2.5, atol=1e-9)


def test_sample_sphere_is_isotropic():
    # each coordinate should have mean ~0 (no directional bias)
    rng = np.random.default_rng(1)
    x = sample_sphere(20000, 5, radius=1.0, rng=rng)
    means = x.mean(axis=0)
    assert np.all(np.abs(means) < 0.02)


def test_sample_sphere_marginal_variance_matches_uniform_sphere():
    # for a uniform point on S^{d-1} of radius R, each coordinate has variance R^2/d
    rng = np.random.default_rng(2)
    d, R = 8, 3.0
    x = sample_sphere(50000, d, radius=R, rng=rng)
    empirical_var = x.var(axis=0)
    expected = R**2 / d
    assert np.allclose(empirical_var, expected, rtol=0.1)


def test_make_dataset_shapes_and_labels():
    rng = np.random.default_rng(3)
    x, y = make_dataset(100, d=6, r_inner=1.0, r_outer=1.3, rng=rng)
    assert x.shape == (200, 6)
    assert y.shape == (200,)
    assert set(np.unique(y).tolist()) == {0.0, 1.0}
    norms = np.linalg.norm(x, axis=1)
    inner_norms = norms[y == 0]
    outer_norms = norms[y == 1]
    assert np.allclose(inner_norms, 1.0, atol=1e-9)
    assert np.allclose(outer_norms, 1.3, atol=1e-9)


def test_make_dataset_is_shuffled_not_block_ordered():
    rng = np.random.default_rng(4)
    _, y = make_dataset(50, d=4, r_inner=1.0, r_outer=1.2, rng=rng)
    # if it were block-ordered (all 0s then all 1s), the first half would be
    # entirely class 0
    assert not np.all(y[:50] == 0.0)
