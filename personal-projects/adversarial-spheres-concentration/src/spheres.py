"""Concentric-sphere dataset generation (Gilmer et al. 2018 "Adversarial Spheres" task)."""

import numpy as np


def sample_sphere(n, d, radius, rng):
    """n points drawn uniformly from the surface of S^{d-1} scaled to `radius`."""
    x = rng.normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return radius * x


def make_dataset(n_per_class, d, r_inner, r_outer, rng):
    """Concentric-sphere binary classification dataset.

    Class 0: uniform on S^{d-1} of radius r_inner.
    Class 1: uniform on S^{d-1} of radius r_outer.
    """
    x0 = sample_sphere(n_per_class, d, r_inner, rng)
    x1 = sample_sphere(n_per_class, d, r_outer, rng)
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    perm = rng.permutation(len(y))
    return x[perm], y[perm]
