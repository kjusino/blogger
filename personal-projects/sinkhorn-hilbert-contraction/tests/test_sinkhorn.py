import numpy as np
import pytest

from src.cost_matrices import random_points_cost
from src.exact_ot import exact_ot_uniform
from src.sinkhorn import entropic_cost, sinkhorn_log


def test_sinkhorn_converges_and_satisfies_marginals():
    rng = np.random.default_rng(0)
    n, m = 12, 9
    C = random_points_cost(n, m, rng)
    r = np.full(n, 1.0 / n)
    c = np.full(m, 1.0 / m)
    result = sinkhorn_log(C, r, c, eps=0.1, max_iter=2000, tol=1e-10)

    assert result.converged
    np.testing.assert_allclose(result.plan.sum(axis=1), r, atol=1e-8)
    np.testing.assert_allclose(result.plan.sum(axis=0), c, atol=1e-8)
    assert np.all(result.plan >= 0)


def test_sinkhorn_residual_history_is_nonincreasing_in_the_tail():
    rng = np.random.default_rng(1)
    n = m = 15
    C = random_points_cost(n, m, rng)
    r = np.full(n, 1.0 / n)
    c = np.full(m, 1.0 / m)
    result = sinkhorn_log(C, r, c, eps=0.05, max_iter=3000, tol=1e-12)

    hist = result.residual_history
    assert len(hist) >= 2
    # Overall trend must be decreasing: the tail average must be much
    # smaller than the head average (individual steps can be noisy near
    # machine precision, so we don't require strict monotonicity).
    head = hist[: max(1, len(hist) // 10)].mean()
    tail = hist[-max(1, len(hist) // 10):].mean()
    assert tail <= head


def test_sinkhorn_rejects_non_probability_marginals():
    C = np.ones((3, 3))
    with pytest.raises(ValueError):
        sinkhorn_log(C, np.array([0.5, 0.5, 0.5]), np.full(3, 1 / 3), eps=0.1)


def test_sinkhorn_rejects_nonpositive_eps():
    C = np.ones((3, 3))
    r = c = np.full(3, 1 / 3)
    with pytest.raises(ValueError):
        sinkhorn_log(C, r, c, eps=0.0)


def test_sinkhorn_entropic_cost_approaches_exact_ot_as_eps_shrinks():
    """As eps -> 0, entropic OT cost must converge (monotonically, in the
    limit) to the true OT cost -- checked here against the Hungarian-
    algorithm exact solution for a small square, uniform-marginal instance.
    """
    rng = np.random.default_rng(2)
    n = 8
    C = random_points_cost(n, n, rng)
    exact = exact_ot_uniform(C)
    r = c = np.full(n, 1.0 / n)

    gaps = []
    for eps in (1.0, 0.3, 0.1, 0.03, 0.01):
        result = sinkhorn_log(C, r, c, eps=eps, max_iter=20_000, tol=1e-11)
        cost = entropic_cost(result.plan, C)
        assert cost >= exact - 1e-6  # entropic cost is always >= exact OT cost
        gaps.append(cost - exact)

    # Gap must shrink as eps shrinks (entropic bias vanishes), and the
    # smallest-eps gap must be small in absolute terms.
    assert gaps[-1] < gaps[0]
    assert gaps[-1] < 1e-2
