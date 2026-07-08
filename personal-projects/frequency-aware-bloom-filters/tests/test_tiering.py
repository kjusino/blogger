import numpy as np
import pytest

from fab.tiering import assign_k, hot_key_set, solve_k_cold, uniform_k


def test_hot_key_set_picks_top_fraction():
    scores = {f"k{i}": float(i) for i in range(100)}  # k99 is highest score
    hot = hot_key_set(scores, hot_fraction=0.1)
    assert len(hot) == 10
    assert "k99" in hot
    assert "k0" not in hot


def test_hot_key_set_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        hot_key_set({"a": 1.0}, hot_fraction=1.5)


def test_solve_k_cold_preserves_budget():
    hot_fraction, k_hot, k_base = 0.1, 10, 6.0
    k_cold = solve_k_cold(hot_fraction, k_hot, k_base)
    average = hot_fraction * k_hot + (1 - hot_fraction) * k_cold
    assert average == pytest.approx(k_base)


def test_solve_k_cold_clamped_at_one():
    # Extreme case: tiny hot fraction with a huge k_hot would otherwise
    # force k_cold negative; it must clamp to 1.
    k_cold = solve_k_cold(hot_fraction=0.01, k_hot=1000, k_base=2.0)
    assert k_cold == 1.0


def test_assign_k_hot_keys_get_more_hash_functions_than_cold():
    rng = np.random.default_rng(0)
    keys = [f"k{i}" for i in range(1000)]
    scores = {key: float(i) for i, key in enumerate(keys)}  # k999 hottest
    ks = assign_k(keys, scores, hot_fraction=0.1, k_hot=10, k_base=6.0, rng=rng)

    hot = hot_key_set(scores, 0.1)
    hot_ks = [ks[k] for k in hot]
    cold_ks = [ks[k] for k in keys if k not in hot]

    assert all(hk == 10 for hk in hot_ks)
    assert max(cold_ks) <= 10
    assert np.mean(cold_ks) < np.mean(hot_ks)


def test_assign_k_average_matches_budget_for_large_n():
    """Randomized rounding of the fractional k_cold should converge to the
    target average as n grows (law of large numbers)."""
    rng = np.random.default_rng(123)
    n = 20_000
    keys = [f"k{i}" for i in range(n)]
    scores = {key: float(rng.random()) for key in keys}
    k_base = 6.0
    ks = assign_k(keys, scores, hot_fraction=0.1, k_hot=10, k_base=k_base, rng=rng)

    achieved_average = np.mean(list(ks.values()))
    assert achieved_average == pytest.approx(k_base, abs=0.05)


def test_degenerate_hot_fraction_zero_is_effectively_uniform_cold():
    rng = np.random.default_rng(0)
    keys = [f"k{i}" for i in range(50)]
    scores = {key: float(i) for i, key in enumerate(keys)}
    ks = assign_k(keys, scores, hot_fraction=0.0, k_hot=10, k_base=4.0, rng=rng)
    assert all(k == 4 for k in ks.values())


def test_uniform_k_assigns_same_value_to_everyone():
    keys = [f"k{i}" for i in range(10)]
    ks = uniform_k(keys, k_base=7)
    assert set(ks.values()) == {7}
