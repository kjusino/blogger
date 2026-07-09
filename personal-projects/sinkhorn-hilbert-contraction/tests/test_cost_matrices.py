import numpy as np
import pytest

from src.cost_matrices import (
    build_cost,
    clustered_points_cost,
    grid_1d_cost,
    iid_random_cost,
    random_points_cost,
)


@pytest.mark.parametrize("family", ["random_points", "clustered_points", "grid_1d", "iid_random"])
def test_build_cost_shape_and_nonnegativity(family):
    rng = np.random.default_rng(0)
    C = build_cost(family, n=7, m=5, rng=rng)
    assert C.shape == (7, 5)
    assert np.all(C >= 0)
    assert np.all(np.isfinite(C))


def test_build_cost_unknown_family_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        build_cost("not_a_family", 3, 3, rng)


def test_random_points_cost_deterministic_given_rng_seed():
    C1 = random_points_cost(5, 5, np.random.default_rng(42))
    C2 = random_points_cost(5, 5, np.random.default_rng(42))
    np.testing.assert_array_equal(C1, C2)


def test_random_points_cost_varies_with_seed():
    C1 = random_points_cost(5, 5, np.random.default_rng(1))
    C2 = random_points_cost(5, 5, np.random.default_rng(2))
    assert not np.allclose(C1, C2)


def test_grid_1d_cost_is_deterministic_and_symmetric_structure():
    C = grid_1d_cost(6, 6)
    # Endpoints are 0 apart from themselves, 1 apart from the opposite endpoint.
    assert C[0, 0] == pytest.approx(0.0)
    assert C[-1, -1] == pytest.approx(0.0)
    assert C[0, -1] == pytest.approx(1.0)
    assert C[-1, 0] == pytest.approx(1.0)
    # Symmetric when n == m.
    np.testing.assert_allclose(C, C.T)


def test_clustered_points_cost_is_bimodal_within_vs_across_cluster():
    rng = np.random.default_rng(3)
    n_clusters = 2
    separation = 5.0
    C = clustered_points_cost(
        200, 200, rng, n_clusters=n_clusters, cluster_std=0.01, separation=separation
    )
    assert C.min() >= 0
    # Cluster assignment is random per point, so we can't assume any fixed
    # block structure by index -- instead split entries by a threshold well
    # between the within-cluster scale (~cluster_std^2) and the
    # across-cluster scale (~separation^2), and check both groups are
    # populated and well separated from each other.
    threshold = (separation**2) / 4.0
    small = C[C < threshold]
    large = C[C >= threshold]
    assert small.size > 0 and large.size > 0
    assert small.max() < 0.01
    assert large.min() > 1.0


def test_iid_random_cost_range():
    rng = np.random.default_rng(4)
    C = iid_random_cost(50, 50, rng)
    assert C.min() >= 0.0
    assert C.max() <= 1.0
    assert 0.3 < C.mean() < 0.7  # sanity check on Uniform(0,1) mean ~0.5
