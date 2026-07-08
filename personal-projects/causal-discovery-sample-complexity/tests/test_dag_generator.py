import numpy as np
import pytest

from src.dag_generator import (
    LinearSEM,
    generate_random_dag,
    generate_faithful_dag,
    _sample_edge_weight,
)


def test_generate_random_dag_respects_max_degree():
    rng = np.random.default_rng(0)
    for _ in range(20):
        sem = generate_random_dag(p=20, max_degree=3, rng=rng)
        assert sem.realized_max_degree() <= 3


def test_generate_random_dag_is_acyclic_by_construction():
    rng = np.random.default_rng(1)
    sem = generate_random_dag(p=15, max_degree=3, rng=rng)
    for j, parents in enumerate(sem.parents):
        assert all(i < j for i in parents), "parents must precede children in topological order"


def test_skeleton_is_symmetric_and_matches_edge_count():
    rng = np.random.default_rng(2)
    sem = generate_random_dag(p=12, max_degree=2, rng=rng)
    skel = sem.skeleton()
    assert np.array_equal(skel, skel.T)
    n_directed_edges = sum(len(p) for p in sem.parents)
    assert skel.sum() == 2 * n_directed_edges  # each edge counted twice (symmetric)
    assert not skel.diagonal().any()


def test_zero_max_degree_gives_no_edges():
    rng = np.random.default_rng(3)
    sem = generate_random_dag(p=10, max_degree=0, rng=rng)
    assert sem.skeleton().sum() == 0
    assert sem.weights == {}


def test_sample_shape_and_finiteness():
    rng = np.random.default_rng(4)
    sem = generate_random_dag(p=8, max_degree=2, rng=rng)
    X = sem.sample(n=500, rng=rng)
    assert X.shape == (500, 8)
    assert np.all(np.isfinite(X))


def test_sample_columns_are_standardized():
    rng = np.random.default_rng(5)
    sem = generate_random_dag(p=6, max_degree=2, rng=rng)
    X = sem.sample(n=2000, rng=rng)
    assert np.allclose(X.mean(axis=0), 0, atol=0.1)
    assert np.allclose(X.std(axis=0), 1, atol=0.1)


def test_analytic_covariance_matches_sample_covariance():
    rng = np.random.default_rng(6)
    sem = generate_random_dag(p=6, max_degree=2, rng=rng)
    sigma = sem.analytic_covariance()
    assert sigma.shape == (6, 6)
    assert np.allclose(sigma, sigma.T)
    # Compare (unstandardized) analytic covariance to a large raw sample's.
    n = 400_000
    X = np.zeros((n, sem.p))
    for j in range(sem.p):
        mean = np.zeros(n)
        for i in sem.parents[j]:
            mean += sem.weights[(i, j)] * X[:, i]
        X[:, j] = mean + rng.normal(0.0, 1.0, size=n)
    empirical = np.cov(X, rowvar=False)
    assert np.allclose(sigma, empirical, atol=0.05)


def test_edge_weight_magnitude_bounded_away_from_zero():
    rng = np.random.default_rng(7)
    weights = [_sample_edge_weight(rng) for _ in range(500)]
    assert all(0.4 <= abs(w) <= 1.2 for w in weights)
    assert any(w < 0 for w in weights) and any(w > 0 for w in weights)


def test_generate_faithful_dag_is_faithful_and_meets_margin():
    from src.pc_algorithm import estimate_skeleton_oracle, min_true_edge_margin

    rng = np.random.default_rng(8)
    min_margin = 0.1
    sem = generate_faithful_dag(p=10, max_degree=2, rng=rng, min_margin=min_margin)
    cov = sem.analytic_covariance()
    oracle = estimate_skeleton_oracle(cov, eps=1e-8)
    assert np.array_equal(oracle.skeleton, sem.skeleton())
    if sem.weights:
        margin = min_true_edge_margin(sem.skeleton(), oracle)
        assert margin >= min_margin


def test_generate_faithful_dag_raises_when_infeasible():
    rng = np.random.default_rng(9)
    with pytest.raises(RuntimeError):
        generate_faithful_dag(
            p=10, max_degree=2, rng=rng, min_margin=0.999,
            max_weight_attempts=2, max_structure_attempts=2,
        )
