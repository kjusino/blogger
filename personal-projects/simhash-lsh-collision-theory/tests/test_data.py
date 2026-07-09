import math

import numpy as np
import pytest

from src.data import (
    planted_neighbor_dataset,
    random_unit_vector,
    random_unit_vectors,
    vector_at_angle,
)


def test_random_unit_vector_has_unit_norm():
    rng = np.random.default_rng(0)
    v = random_unit_vector(50, rng)
    assert np.linalg.norm(v) == pytest.approx(1.0)


def test_random_unit_vectors_all_unit_norm():
    rng = np.random.default_rng(0)
    V = random_unit_vectors(20, 8, rng)
    norms = np.linalg.norm(V, axis=1)
    np.testing.assert_allclose(norms, np.ones(20), atol=1e-10)


@pytest.mark.parametrize("theta", [0.0, 0.01, 0.5, math.pi / 2, 2.0, math.pi - 0.01, math.pi])
def test_vector_at_angle_produces_exact_angle(theta):
    rng = np.random.default_rng(7)
    u = random_unit_vector(25, rng)
    v = vector_at_angle(u, theta, rng)
    assert np.linalg.norm(v) == pytest.approx(1.0)
    cos_actual = np.clip(np.dot(u, v), -1.0, 1.0)
    theta_actual = math.acos(cos_actual)
    assert theta_actual == pytest.approx(theta, abs=1e-6)


def test_vector_at_angle_is_reproducible_with_same_rng_state():
    rng1 = np.random.default_rng(11)
    rng2 = np.random.default_rng(11)
    u1 = random_unit_vector(10, rng1)
    u2 = random_unit_vector(10, rng2)
    v1 = vector_at_angle(u1, 1.2, rng1)
    v2 = vector_at_angle(u2, 1.2, rng2)
    np.testing.assert_allclose(v1, v2)


def test_planted_neighbor_dataset_shapes_and_angle():
    rng = np.random.default_rng(5)
    near_theta = 0.4
    query, dataset, planted_idx = planted_neighbor_dataset(
        n_background=50, dim=16, near_theta=near_theta, rng=rng
    )
    assert query.shape == (16,)
    assert dataset.shape == (51, 16)
    planted_vec = dataset[planted_idx]
    cos_actual = np.clip(
        np.dot(query, planted_vec) / (np.linalg.norm(query) * np.linalg.norm(planted_vec)),
        -1.0,
        1.0,
    )
    theta_actual = math.acos(cos_actual)
    assert theta_actual == pytest.approx(near_theta, abs=1e-6)


def test_planted_neighbor_is_closer_than_typical_background_in_high_dim():
    """Sanity check of the experimental setup: in higher dimension, i.i.d.
    random background vectors concentrate near angle pi/2, so a
    planted neighbor at a small angle should be closer than the background
    median."""
    rng = np.random.default_rng(9)
    dim = 64
    query, dataset, planted_idx = planted_neighbor_dataset(
        n_background=500, dim=dim, near_theta=0.2, rng=rng
    )
    q_norm = query / np.linalg.norm(query)
    d_norm = dataset / np.linalg.norm(dataset, axis=1, keepdims=True)
    sims = d_norm @ q_norm
    planted_sim = sims[planted_idx]
    background_sims = np.delete(sims, planted_idx)
    assert planted_sim > np.median(background_sims)
