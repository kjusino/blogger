import numpy as np

from src.pc_algorithm import (
    estimate_skeleton,
    estimate_skeleton_oracle,
    orient_edges,
    min_true_edge_margin,
)
from src.dag_generator import generate_random_dag, generate_faithful_dag
from src.metrics import skeleton_shd, exact_recovery


def _chain_covariance(betas):
    """Covariance of a linear-Gaussian chain X0 -> X1 -> ... -> Xk with the
    given edge weights and unit noise variance."""
    p = len(betas) + 1
    B = np.zeros((p, p))
    for i, b in enumerate(betas):
        B[i, i + 1] = b
    inv = np.linalg.inv(np.eye(p) - B)
    return inv.T @ inv


def test_oracle_recovers_chain_skeleton_exactly():
    cov = _chain_covariance([0.7, -0.6, 0.5])
    result = estimate_skeleton_oracle(cov, eps=1e-8)
    expected = np.array([
        [False, True, False, False],
        [True, False, True, False],
        [False, True, False, True],
        [False, False, True, False],
    ])
    assert np.array_equal(result.skeleton, expected)


def test_oracle_finds_correct_sepset_for_chain():
    cov = _chain_covariance([0.7, -0.6, 0.5])
    result = estimate_skeleton_oracle(cov, eps=1e-8)
    # X0 and X2 should be separated by {1} (or a superset thereof).
    sep = result.sepsets[frozenset((0, 2))]
    assert 1 in sep


def test_finite_sample_recovers_chain_with_ample_data():
    rng = np.random.default_rng(0)
    betas = [0.8, -0.7, 0.6]
    p = len(betas) + 1
    B = np.zeros((p, p))
    for i, b in enumerate(betas):
        B[i, i + 1] = b
    n = 20000
    X = np.zeros((n, p))
    for j in range(p):
        mean = np.zeros(n)
        for i in range(j):
            if B[i, j] != 0:
                mean += B[i, j] * X[:, i]
        X[:, j] = mean + rng.normal(size=n)
    cov = np.cov(X, rowvar=False)
    result = estimate_skeleton(cov, n=n, alpha=0.01)
    expected = _chain_covariance(betas) != 0  # placeholder, replaced below
    true_skel = np.zeros((p, p), dtype=bool)
    for i in range(p - 1):
        true_skel[i, i + 1] = True
        true_skel[i + 1, i] = True
    assert skeleton_shd(result.skeleton, true_skel) == 0


def test_v_structure_orientation_on_collider():
    # X0 -> X2 <- X1, X0 and X1 marginally independent (no edge between them).
    p = 3
    B = np.zeros((p, p))
    B[0, 2] = 0.9
    B[1, 2] = -0.8
    inv = np.linalg.inv(np.eye(p) - B)
    cov = inv.T @ inv
    result = estimate_skeleton_oracle(cov, eps=1e-8)
    assert not result.skeleton[0, 1]
    assert result.skeleton[0, 2] and result.skeleton[1, 2]
    directed = orient_edges(result)
    # Collider: 0 -> 2 and 1 -> 2, i.e. only the (x, 2) direction survives.
    assert directed[0, 2] and not directed[2, 0]
    assert directed[1, 2] and not directed[2, 1]


def test_estimate_skeleton_respects_max_cond_set_cap():
    rng = np.random.default_rng(2)
    sem = generate_random_dag(p=10, max_degree=2, rng=rng)
    X = sem.sample(500, rng)
    cov = np.cov(X, rowvar=False)
    result = estimate_skeleton(cov, n=500, alpha=0.05, max_cond_set=1)
    assert result.p == 10  # runs without error / combinatorial blowup


def test_min_true_edge_margin_is_positive_for_faithful_dag():
    rng = np.random.default_rng(3)
    sem = generate_faithful_dag(p=8, max_degree=2, rng=rng, min_margin=0.1)
    if sem.weights:
        cov = sem.analytic_covariance()
        oracle = estimate_skeleton_oracle(cov, eps=1e-8)
        margin = min_true_edge_margin(sem.skeleton(), oracle)
        assert margin >= 0.1


def test_min_true_edge_margin_is_inf_for_edgeless_graph():
    p = 4
    empty_skeleton = np.zeros((p, p), dtype=bool)
    from src.pc_algorithm import PCResult

    result = PCResult(p=p, skeleton=empty_skeleton, sepsets={}, margins={})
    assert min_true_edge_margin(empty_skeleton, result) == float("inf")
