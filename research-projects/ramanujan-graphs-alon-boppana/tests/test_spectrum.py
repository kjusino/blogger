import math

import pytest

from ramanujan_spectra.graphs import (
    complete_bipartite_regular_adjacency,
    complete_graph_adjacency,
    networkx_regular_graph,
    petersen_graph_adjacency,
)
from ramanujan_spectra.spectrum import _dense_extremes, _sparse_extremes, extremal_eigenvalues


@pytest.mark.parametrize("d", [3, 5, 8])
def test_complete_graph_exact_spectrum(d):
    # K_{d+1}: spectrum {d (x1), -1 (x d)} => lambda2 = -1, lambda_min = -1
    spec = extremal_eigenvalues(complete_graph_adjacency(d), d)
    assert spec.lambda1 == pytest.approx(d, abs=1e-9)
    assert spec.lambda2 == pytest.approx(-1.0, abs=1e-9)
    assert spec.lambda_min == pytest.approx(-1.0, abs=1e-9)
    assert spec.lambda2_abs == pytest.approx(1.0, abs=1e-9)
    assert spec.bipartite_like is False


@pytest.mark.parametrize("d", [3, 5, 8])
def test_complete_bipartite_exact_spectrum(d):
    # K_{d,d}: spectrum {d (x1), -d (x1), 0 (x 2d-2)} => lambda(G) = d
    spec = extremal_eigenvalues(complete_bipartite_regular_adjacency(d), d)
    assert spec.lambda1 == pytest.approx(d, abs=1e-9)
    assert spec.lambda2 == pytest.approx(0.0, abs=1e-8)
    assert spec.lambda_min == pytest.approx(-d, abs=1e-9)
    assert spec.lambda2_abs == pytest.approx(d, abs=1e-9)
    assert spec.bipartite_like is True


def test_petersen_graph_exact_spectrum():
    # spectrum {3 (x1), 1 (x5), -2 (x4)}
    spec = extremal_eigenvalues(petersen_graph_adjacency(), 3)
    assert spec.lambda1 == pytest.approx(3.0, abs=1e-9)
    assert spec.lambda2 == pytest.approx(1.0, abs=1e-9)
    assert spec.lambda_min == pytest.approx(-2.0, abs=1e-9)
    assert spec.lambda2_abs == pytest.approx(2.0, abs=1e-9)
    assert spec.bipartite_like is False
    # this is the textbook example of a Ramanujan graph
    assert spec.lambda2_abs <= 2 * math.sqrt(3 - 1) + 1e-9


def test_extremal_eigenvalues_raises_if_not_d_regular():
    adj = complete_graph_adjacency(5)  # actually 5-regular
    with pytest.raises(ValueError):
        extremal_eigenvalues(adj, 4)  # wrong claimed degree


def test_dense_and_sparse_paths_agree_on_a_moderate_graph():
    d, n = 4, 400  # above DENSE_CUTOFF, exercises the sparse path
    adj = networkx_regular_graph(d, n, seed=7)

    dense_eigs = _dense_extremes(adj)
    lambda1_dense, lambda2_dense = float(dense_eigs[-1]), float(dense_eigs[-2])
    lambda_min_dense = float(dense_eigs[0])

    top_sparse, bottom_sparse = _sparse_extremes(adj, k=4)
    lambda1_sparse, lambda2_sparse = float(top_sparse[-1]), float(top_sparse[-2])
    lambda_min_sparse = float(bottom_sparse[0])

    assert lambda1_sparse == pytest.approx(lambda1_dense, abs=1e-6)
    assert lambda2_sparse == pytest.approx(lambda2_dense, abs=1e-6)
    assert lambda_min_sparse == pytest.approx(lambda_min_dense, abs=1e-6)


def test_extremal_eigenvalues_matches_manual_sparse_path_at_n_above_cutoff():
    d, n = 6, 500
    adj = networkx_regular_graph(d, n, seed=3)
    spec = extremal_eigenvalues(adj, d)
    dense_eigs = _dense_extremes(adj)
    assert spec.lambda1 == pytest.approx(float(dense_eigs[-1]), abs=1e-6)
    assert spec.lambda2 == pytest.approx(float(dense_eigs[-2]), abs=1e-6)
    assert spec.lambda_min == pytest.approx(float(dense_eigs[0]), abs=1e-6)
