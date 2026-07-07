import numpy as np
import pytest

from tda_phase_transitions import graph_models


def test_er_distance_matrix_shape_and_symmetry():
    m = graph_models.er_distance_matrix(50, seed=1)
    assert m.dist.shape == (50, 50)
    assert np.allclose(np.diag(m.dist), 0.0)
    assert np.allclose(m.dist, m.dist.T)
    assert m.dist.min() >= 0.0
    assert m.dist.max() <= 1.0


def test_er_thresholding_matches_expected_edge_count():
    n = 300
    m = graph_models.er_distance_matrix(n, seed=7)
    p = 0.05
    iu = np.triu_indices(n, k=1)
    num_edges = (m.dist[iu] <= p).sum()
    expected = p * len(iu[0])
    # Binomial with ~22K trials at p=0.05: std ~ sqrt(n*p*(1-p)) ~ 32
    assert abs(num_edges - expected) < 6 * np.sqrt(expected)


def test_er_requires_at_least_two_nodes():
    with pytest.raises(ValueError):
        graph_models.er_distance_matrix(1)


def test_rgg_distance_matrix_symmetry_and_bounds():
    m = graph_models.rgg_distance_matrix(40, seed=2)
    assert np.allclose(np.diag(m.dist), 0.0)
    assert np.allclose(m.dist, m.dist.T)
    # max toroidal distance in [0,1)^2 is sqrt(0.5^2 + 0.5^2)
    assert m.dist.max() <= np.sqrt(0.5)


def test_rgg_thresholding_matches_disc_area():
    n = 300
    m = graph_models.rgg_distance_matrix(n, seed=3)
    r = 0.1
    iu = np.triu_indices(n, k=1)
    num_edges = (m.dist[iu] <= r).sum()
    # On a torus, P(dist <= r) = pi r^2 exactly for r <= 0.5 (no boundary effects)
    expected = np.pi * r ** 2 * len(iu[0])
    assert abs(num_edges - expected) < 6 * np.sqrt(expected)


def test_chung_lu_weights_are_positive_and_bounded():
    m = graph_models.chung_lu_distance_matrix(200, gamma=2.5, w_min=1.0, w_max_ratio=10.0, seed=4)
    assert (m.weights >= 1.0 - 1e-9).all()
    assert (m.weights <= 10.0 + 1e-9).all()


def test_chung_lu_distance_matrix_symmetry():
    m = graph_models.chung_lu_distance_matrix(60, seed=5)
    assert np.allclose(np.diag(m.dist), 0.0)
    assert np.allclose(m.dist, m.dist.T)
    assert (m.dist >= 0).all()


def test_chung_lu_thresholding_matches_expected_probability():
    n = 400
    m = graph_models.chung_lu_distance_matrix(n, gamma=2.5, seed=6)
    theta = 0.3
    L = m.weights.sum()
    iu = np.triu_indices(n, k=1)
    prob = np.minimum(1.0, theta * m.weights[iu[0]] * m.weights[iu[1]] / L)
    expected_edges = prob.sum()
    variance = (prob * (1.0 - prob)).sum()
    num_edges = (m.dist[iu] <= theta).sum()
    # Poisson-binomial: compare against its own mean/variance rather than a
    # fixed relative tolerance, since heavy-tailed weights make a handful of
    # pairs near-certain (low variance) or near-even (high variance) edges.
    assert abs(num_edges - expected_edges) < 8 * np.sqrt(variance) + 5


def test_chung_lu_rejects_gamma_le_one():
    with pytest.raises(ValueError):
        graph_models.chung_lu_distance_matrix(20, gamma=1.0)
