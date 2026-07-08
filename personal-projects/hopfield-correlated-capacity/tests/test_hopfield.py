import numpy as np
import pytest

from src.hopfield import (
    hebbian_weights,
    run_async_dynamics_batch,
    overlap,
    corrupt,
)
from src.patterns import generate_correlated_patterns


def test_hebbian_weights_symmetric_and_zero_diagonal():
    rng = np.random.default_rng(0)
    patterns = generate_correlated_patterns(n=60, p=10, rho=0.2, rng=rng)
    w = hebbian_weights(patterns)
    assert w.shape == (60, 60)
    assert np.allclose(w, w.T)
    assert np.allclose(np.diag(w), 0.0)


def test_overlap_of_pattern_with_itself_is_one():
    rng = np.random.default_rng(0)
    patterns = generate_correlated_patterns(n=40, p=1, rho=0.0, rng=rng)
    m = overlap(patterns, patterns)
    assert m[0] == pytest.approx(1.0)


def test_corrupt_flips_expected_fraction():
    rng = np.random.default_rng(0)
    pattern = np.ones(200)
    corrupted = corrupt(pattern, flip_frac=0.05, rng=rng)
    n_flipped = np.sum(corrupted != pattern)
    assert n_flipped == 10  # 5% of 200


def test_single_pattern_low_load_retrieves_perfectly():
    """Trivial sanity check: a single stored pattern (alpha = 1/N, far below
    the classical capacity) corrupted by 5% bit flips should be retrieved
    essentially perfectly by asynchronous dynamics."""
    rng = np.random.default_rng(7)
    n = 200
    patterns = generate_correlated_patterns(n=n, p=1, rho=0.0, rng=rng)
    w = hebbian_weights(patterns)

    init = corrupt(patterns, flip_frac=0.05, rng=rng)
    final_states, sweeps_used = run_async_dynamics_batch(w, init, rng, max_sweeps=20)
    m = overlap(final_states, patterns)

    assert m[0] > 0.99
    assert sweeps_used <= 20


def test_batch_retrieval_independent_instances_converge():
    rng = np.random.default_rng(3)
    n = 150
    patterns = generate_correlated_patterns(n=n, p=3, rho=0.0, rng=rng)
    w = hebbian_weights(patterns)

    init = corrupt(patterns, flip_frac=0.05, rng=rng)
    final_states, _ = run_async_dynamics_batch(w, init, rng, max_sweeps=20)
    m = overlap(final_states, patterns)

    assert np.all(np.abs(m) > 0.95)


def test_high_overload_destroys_retrieval():
    """At a grossly excessive load (alpha well above the classical ~0.138
    capacity), retrieval should fail (mean |overlap| well below 1)."""
    rng = np.random.default_rng(11)
    n = 100
    p = 60  # alpha = 0.6, far above capacity
    patterns = generate_correlated_patterns(n=n, p=p, rho=0.0, rng=rng)
    w = hebbian_weights(patterns)

    idx = rng.choice(p, size=5, replace=False)
    test_patterns = patterns[idx]
    init = corrupt(test_patterns, flip_frac=0.05, rng=rng)
    final_states, _ = run_async_dynamics_batch(w, init, rng, max_sweeps=20)
    m = overlap(final_states, test_patterns)

    assert np.mean(np.abs(m)) < 0.9
