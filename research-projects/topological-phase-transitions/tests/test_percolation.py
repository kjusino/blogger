import numpy as np

from tda_phase_transitions.percolation import UnionFind, percolation_curve, find_percolation_threshold


def test_union_find_basic_merges_and_sizes():
    uf = UnionFind(5)
    assert uf.union(0, 1) is True
    assert uf.union(0, 1) is False  # already merged
    assert uf.union(2, 3) is True
    sizes = sorted(uf.component_sizes().tolist())
    assert sizes == [1, 2, 2]
    uf.union(1, 2)
    sizes = sorted(uf.component_sizes().tolist())
    assert sizes == [1, 4]


def _path_graph_distance_matrix(n):
    # 0-1-2-...-(n-1), edge (i, i+1) has weight i+1; all other pairs far away
    dist = np.full((n, n), 1000.0)
    np.fill_diagonal(dist, 0.0)
    for i in range(n - 1):
        dist[i, i + 1] = dist[i + 1, i] = i + 1
    return dist


def test_percolation_curve_on_path_graph():
    n = 5
    dist = _path_graph_distance_matrix(n)
    thresholds = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    giant_frac, chi = percolation_curve(dist, thresholds)
    # After threshold t, edges 1..floor(t) are present, forming one chain
    # of length floor(t)+1 and n - floor(t) - 1 isolated singletons.
    expected_giant = np.array([1, 2, 3, 4, 5]) / n
    assert np.allclose(giant_frac, expected_giant)
    # susceptibility excludes the giant: sum of squares of the rest / n
    # t=0.5: giant=1, rest = four singletons -> chi = 4*1/5
    assert np.isclose(chi[0], 4 * 1 / n)
    # t=4.5: giant=5 (all merged), rest empty -> chi = 0
    assert np.isclose(chi[-1], 0.0)


def test_find_percolation_threshold_on_two_stage_merge():
    # Small cluster merges early (weight 1), then a big merge happens later
    # (weight 10) that should produce peak susceptibility right before it.
    n = 8
    dist = np.full((n, n), 1000.0)
    np.fill_diagonal(dist, 0.0)
    # Two clusters of size 4, formed by weight-1 edges (star from node 0 and node 4)
    for i in [1, 2, 3]:
        dist[0, i] = dist[i, 0] = 1.0
    for i in [5, 6, 7]:
        dist[4, i] = dist[i, 4] = 1.0
    # Merge the two clusters at weight 10
    dist[0, 4] = dist[4, 0] = 10.0

    thresholds = np.linspace(0, 20, 41)
    threshold = find_percolation_threshold(dist, thresholds)
    # susceptibility should peak somewhere between the two cluster merges
    # (two clusters of size 4 -> chi = 4^2/8 = 2) and before the final merge
    assert 1.0 <= threshold < 10.0
