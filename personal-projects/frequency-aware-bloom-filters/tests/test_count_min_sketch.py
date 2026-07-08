from collections import Counter

import numpy as np
import pytest

from fab.count_min_sketch import CountMinSketch


def test_estimate_never_underestimates():
    rng = np.random.default_rng(0)
    stream = [f"k{i}" for i in rng.integers(0, 50, size=5000)]
    true_counts = Counter(stream)

    cms = CountMinSketch.from_stream(stream, width=64, depth=4, seed=1)
    for key, true_count in true_counts.items():
        assert cms.estimate(key) >= true_count


def test_estimate_error_within_theoretical_bound_with_high_probability():
    """Run many independent (sketch-seed) trials; the fraction of trials
    where the additive error exceeds eps*L should not wildly exceed the
    theoretical delta (a coarse but real statistical check, not just a
    restatement of the guarantee)."""
    rng = np.random.default_rng(0)
    width, depth = 200, 4
    stream_len = 4000
    stream = [f"k{i}" for i in rng.integers(0, 300, size=stream_len)]
    true_counts = Counter(stream)

    violations = 0
    n_trials = 30
    for seed in range(n_trials):
        cms = CountMinSketch.from_stream(stream, width=width, depth=depth, seed=seed)
        eps, _delta = cms.error_bound()
        for key, true_count in true_counts.items():
            if cms.estimate(key) > true_count + eps * stream_len:
                violations += 1
                break  # one violation is enough to flag this trial

    # Theoretical delta for depth=4 is e^-4 ~= 1.8%; allow slack for the
    # finite number of trials and the union bound over many keys per trial.
    assert violations / n_trials < 0.5


def test_deterministic_given_seed():
    stream = ["a", "b", "a", "c", "a", "b"]
    cms1 = CountMinSketch.from_stream(stream, width=32, depth=3, seed=5)
    cms2 = CountMinSketch.from_stream(stream, width=32, depth=3, seed=5)
    for key in ["a", "b", "c", "d"]:
        assert cms1.estimate(key) == cms2.estimate(key)


def test_frequent_item_estimated_higher_than_rare_item():
    rng = np.random.default_rng(3)
    stream = ["hot"] * 500 + ["cold"] * 5
    rng.shuffle(stream)
    cms = CountMinSketch.from_stream(list(stream), width=500, depth=4, seed=1)
    assert cms.estimate("hot") > cms.estimate("cold")


def test_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        CountMinSketch(width=0, depth=4)
    with pytest.raises(ValueError):
        CountMinSketch(width=10, depth=0)
