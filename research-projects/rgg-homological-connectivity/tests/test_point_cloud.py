import numpy as np
import pytest

from src.point_cloud import sample_torus_points, torus_pairwise_delta


def test_sample_torus_points_shape_and_range():
    rng = np.random.default_rng(0)
    points = sample_torus_points(50, rng)
    assert points.shape == (50, 2)
    assert np.all(points >= 0.0) and np.all(points < 1.0)


def test_sample_torus_points_rejects_non_positive_n():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_torus_points(0, rng)


def test_torus_pairwise_delta_wraps_around():
    a = np.array([[0.01, 0.5]])
    b = np.array([[0.99, 0.5]])
    delta = torus_pairwise_delta(a, b)
    # Minimum-image displacement should be 0.02, not 0.98.
    assert np.isclose(np.abs(delta[0, 0, 0]), 0.02)


def test_torus_pairwise_delta_no_wrap_needed_for_nearby_points():
    a = np.array([[0.5, 0.5]])
    b = np.array([[0.52, 0.5]])
    delta = torus_pairwise_delta(a, b)
    assert np.isclose(delta[0, 0, 0], -0.02)
