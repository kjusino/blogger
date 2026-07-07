import numpy as np

from tda_phase_transitions import persistence


def _square_distance_matrix():
    # Corners of a unit square: sides = 1, diagonals = sqrt(2)
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def test_square_has_one_persistent_cycle():
    dist = _square_distance_matrix()
    dgms = persistence.compute_persistence(dist, maxdim=1, thresh=2.0)
    h1 = dgms[1]
    assert h1.shape[0] == 1
    birth, death = h1[0]
    assert np.isclose(birth, 1.0, atol=1e-6)
    assert np.isclose(death, np.sqrt(2), atol=1e-6)


def test_square_h0_matches_mst_edges():
    dist = _square_distance_matrix()
    dgms = persistence.compute_persistence(dist, maxdim=1, thresh=2.0)
    h0 = dgms[0]
    # 4 points -> 3 finite merges (MST) + 1 infinite bar for the root
    finite = h0[np.isfinite(h0[:, 1])]
    infinite = h0[~np.isfinite(h0[:, 1])]
    assert finite.shape[0] == 3
    assert infinite.shape[0] == 1
    # MST of the 4-cycle-with-diagonals uses the three unit-length sides
    assert np.allclose(np.sort(finite[:, 1]), [1.0, 1.0, 1.0], atol=1e-6)


def test_two_well_separated_clusters_merge_late():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=(0, 0), scale=0.02, size=(6, 2))
    cluster_b = rng.normal(loc=(10, 10), scale=0.02, size=(6, 2))
    pts = np.vstack([cluster_a, cluster_b])
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    dgms = persistence.compute_persistence(dist, maxdim=0, thresh=20.0)
    h0 = dgms[0]
    finite = np.sort(h0[np.isfinite(h0[:, 1])][:, 1])
    # 10 finite merges total; the last one (joining the two clusters) should
    # be far larger than all intra-cluster merges
    intra_cluster_max = finite[-2]
    cross_cluster = finite[-1]
    assert cross_cluster > 50 * intra_cluster_max


def test_betti_curve_counts_alive_bars():
    dgm = np.array([[0.0, 1.0], [0.5, 2.0], [1.5, np.inf]])
    thresholds = np.array([0.25, 0.75, 1.25, 1.75])
    counts = persistence.betti_curve(dgm, thresholds)
    assert np.array_equal(counts, [1.0, 2.0, 1.0, 2.0])


def test_betti_curve_empty_diagram():
    empty = np.zeros((0, 2))
    counts = persistence.betti_curve(empty, [0.0, 1.0])
    assert np.array_equal(counts, [0.0, 0.0])
