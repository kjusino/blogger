import numpy as np
import pytest

from hypercube_cutoff import chain as ch


def test_stationary_weight_pmf_sums_to_one_and_matches_binomial():
    from scipy.stats import binom
    n = 30
    pmf = ch.stationary_weight_pmf(n)
    assert pmf.shape == (n + 1,)
    assert pmf.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pmf, binom.pmf(np.arange(n + 1), n, 0.5))


def test_weight_step_preserves_total_probability():
    n = 15
    dist = np.zeros(n + 1)
    dist[3] = 1.0
    for _ in range(20):
        dist = ch.weight_step(dist, n)
        assert dist.sum() == pytest.approx(1.0)
        assert np.all(dist >= -1e-12)


def test_weight_step_rejects_wrong_shape():
    with pytest.raises(ValueError):
        ch.weight_step(np.zeros(5), 10)


def test_tv_curve_at_t0_is_one_minus_pi0():
    n = 20
    pi0 = ch.stationary_weight_pmf(n)[0]
    result = ch.tv_curve(n, [0])
    assert result[0] == pytest.approx(1 - pi0)


def test_tv_curve_monotonically_decreasing():
    n = 50
    t_values = np.arange(0, 300, 10)
    tv = ch.tv_curve(n, t_values)
    assert np.all(np.diff(tv) <= 1e-9)


def test_tv_curve_converges_to_near_zero():
    n = 30
    tv = ch.tv_curve(n, [5000])
    assert tv[0] < 1e-6


def test_tv_curve_rejects_negative_t():
    with pytest.raises(ValueError):
        ch.tv_curve(10, [-1])


def test_tv_curve_rejects_bad_start_weight():
    with pytest.raises(ValueError):
        ch.tv_curve(10, [5], start_weight=-1)
    with pytest.raises(ValueError):
        ch.tv_curve(10, [5], start_weight=11)


def test_tv_curve_order_independent_of_input_order():
    n = 20
    t_values = [30, 5, 15, 0]
    tv = ch.tv_curve(n, t_values)
    tv_sorted = ch.tv_curve(n, sorted(t_values))
    for t, v in zip(t_values, tv):
        idx = sorted(t_values).index(t)
        assert v == pytest.approx(tv_sorted[idx])


def test_full_chain_step_preserves_probability():
    n = 6
    size = 1 << n
    dist = np.zeros(size)
    dist[0] = 1.0
    for _ in range(10):
        dist = ch.full_chain_step(dist, n)
        assert dist.sum() == pytest.approx(1.0)
        assert np.all(dist >= -1e-12)


def test_full_chain_rejects_n_too_large():
    with pytest.raises(ValueError):
        ch.full_chain_tv_curve(20, [1])


@pytest.mark.integration
def test_lumping_matches_full_chain_exactly():
    """The core claim of the whole project: the weight-only birth-death chain
    is an *exact* lumping of the full 2^n-state chain, not an approximation."""
    for n in [6, 8, 10]:
        t_values = [0, 1, 3, 7, 15, 25]
        exact_lumped = ch.tv_curve(n, t_values)
        exact_full, _ = ch.full_chain_tv_curve(n, t_values)
        np.testing.assert_allclose(exact_lumped, exact_full, atol=1e-9)


@pytest.mark.integration
def test_full_chain_weight_marginal_matches_birth_death_distribution():
    n = 8
    t_values = [0, 5, 12, 20]
    _, weight_marginals = ch.full_chain_tv_curve(n, t_values)

    dist = np.zeros(n + 1)
    dist[0] = 1.0
    step = 0
    for i, target_t in enumerate(t_values):
        while step < target_t:
            dist = ch.weight_step(dist, n)
            step += 1
        np.testing.assert_allclose(dist, weight_marginals[i], atol=1e-9)
