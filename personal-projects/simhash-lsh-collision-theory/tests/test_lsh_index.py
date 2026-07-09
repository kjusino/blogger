import math

import numpy as np
import pytest

from src.data import random_unit_vector, vector_at_angle
from src.lsh_index import LSHIndex, brute_force_nearest


def test_identical_point_always_collides():
    rng = np.random.default_rng(0)
    dim = 20
    u = random_unit_vector(dim, rng)
    index = LSHIndex(dim=dim, k=10, L=5, rng=rng)
    index.index(u[None, :])
    assert index.collides_with(u, target_id=0)


def test_query_candidates_excludes_self_id_when_requested():
    rng = np.random.default_rng(1)
    dim = 10
    X = np.vstack([random_unit_vector(dim, rng) for _ in range(5)])
    index = LSHIndex(dim=dim, k=6, L=3, rng=rng)
    index.index(X)
    candidates = index.query_candidates(X[0], exclude_id=0)
    assert 0 not in candidates


def test_near_duplicate_collides_much_more_often_than_orthogonal_pair():
    """Statistical test: repeated fresh LSH indices should place a
    near-duplicate pair (small angle) in the same bucket far more often
    than a near-orthogonal pair, for a reasonably strict (large k) hash."""
    rng = np.random.default_rng(2)
    dim = 40
    k, L = 12, 4
    num_trials = 300

    u = random_unit_vector(dim, rng)
    near = vector_at_angle(u, 0.05, rng)
    far = vector_at_angle(u, math.pi / 2, rng)

    near_hits = 0
    far_hits = 0
    for _ in range(num_trials):
        index = LSHIndex(dim=dim, k=k, L=L, rng=rng)
        index.index(u[None, :])
        if index.collides_with(near, target_id=0):
            near_hits += 1
        if index.collides_with(far, target_id=0):
            far_hits += 1

    near_rate = near_hits / num_trials
    far_rate = far_hits / num_trials
    assert near_rate > far_rate
    assert near_rate > 0.5
    assert far_rate < 0.5


def test_brute_force_nearest_finds_exact_match():
    rng = np.random.default_rng(3)
    dim = 10
    query = random_unit_vector(dim, rng)
    dataset = np.vstack([random_unit_vector(dim, rng) for _ in range(20)])
    dataset = np.vstack([dataset, query])  # exact match at the last index
    best_idx, best_sim = brute_force_nearest(query, dataset)
    assert best_idx == 20
    assert best_sim == pytest.approx(1.0, abs=1e-9)


def test_lsh_recovers_planted_near_neighbor_with_high_probability():
    """End-to-end sanity check: with well-chosen (k, L), an LSH index should
    retrieve a genuinely near planted neighbor among a random background as
    a query candidate most of the time."""
    rng = np.random.default_rng(4)
    dim = 32
    n_background = 200
    from src.data import planted_neighbor_dataset

    hits = 0
    trials = 40
    for _ in range(trials):
        query, dataset, planted_idx = planted_neighbor_dataset(
            n_background, dim, near_theta=0.1, rng=rng
        )
        index = LSHIndex(dim=dim, k=6, L=10, rng=rng)
        index.index(dataset)
        candidates = index.query_candidates(query)
        if planted_idx in candidates:
            hits += 1
    assert hits / trials > 0.8
