"""End-to-end integration test: build all three schemes (uniform, oracle-
weighted, CMS-weighted) on a small synthetic workload and check the core
research claim holds -- oracle-weighted beats uniform under skew, and
CMS-weighted (a realistic, budget-limited estimator) lands between the two,
recovering most but not necessarily all of the oracle's advantage.

Everything here is seeded for full reproducibility; no flaky randomness.
"""
from __future__ import annotations

import numpy as np

from fab.bloom_filter import BloomFilter
from fab.count_min_sketch import CountMinSketch
from fab.tiering import assign_k, uniform_k
from fab.theory import expected_weighted_fpr, load_factor
from fab.zipf import popularity_weights, sample_stream


def _run_scheme(keys, members, scores, hot_fraction, k_hot, k_base, num_bits, seed, rng):
    if scores is None:
        ks = uniform_k(keys, k_base=int(round(k_base)))
    else:
        ks = assign_k(keys, scores, hot_fraction, k_hot, k_base, rng)

    bf = BloomFilter(num_bits=num_bits, seed=seed)
    for m in members:
        bf.insert(m, ks[m])
    return bf, ks


def test_oracle_weighted_beats_uniform_and_cms_lands_between():
    rng = np.random.default_rng(2024)

    num_keys = 3000
    n_members = 600
    skew = 1.5
    k_base = 6.0
    hot_fraction = 0.1
    k_hot = 10
    num_bits = 6000  # ~10 bits/member, matches experiments/run_experiments.py

    weight_arr = popularity_weights(num_keys, skew, rng=rng)
    keys = [f"k{i}" for i in range(num_keys)]
    scores = dict(zip(keys, weight_arr))

    member_idx = rng.choice(num_keys, size=n_members, replace=False)
    members = [keys[i] for i in member_idx]

    # -- Uniform baseline --
    bf_uniform, ks_uniform = _run_scheme(
        keys, members, None, hot_fraction, k_hot, k_base, num_bits, seed=1, rng=rng
    )

    # -- Oracle-weighted (true Zipf weights known exactly) --
    bf_oracle, ks_oracle = _run_scheme(
        keys, members, scores, hot_fraction, k_hot, k_base, num_bits, seed=1, rng=rng
    )

    # -- CMS-weighted (frequency estimated from a sampled access log) --
    training_stream = sample_stream(weight_arr, length=20_000, rng=rng)
    cms = CountMinSketch.from_stream(training_stream, width=500, depth=4, seed=3)
    cms_scores = {key: cms.estimate(key) for key in keys}
    bf_cms, ks_cms = _run_scheme(
        keys, members, cms_scores, hot_fraction, k_hot, k_base, num_bits, seed=1, rng=rng
    )

    # Exact (non-sampled) expected FPR using the true query-weight
    # distribution, restricted to non-members (the false-positive universe).
    negatives = {key: w for key, w in scores.items() if key not in set(members)}

    def negatives_only(ks):
        return {key: k for key, k in ks.items() if key in negatives}

    load_uniform = load_factor(num_bits, bf_uniform.total_insertions)
    load_oracle = load_factor(num_bits, bf_oracle.total_insertions)
    load_cms = load_factor(num_bits, bf_cms.total_insertions)

    fpr_uniform = expected_weighted_fpr(negatives, negatives_only(ks_uniform), load_uniform)
    fpr_oracle = expected_weighted_fpr(negatives, negatives_only(ks_oracle), load_oracle)
    fpr_cms = expected_weighted_fpr(negatives, negatives_only(ks_cms), load_cms)

    # Core claim: knowing (or estimating) query-frequency and reallocating
    # hash functions accordingly beats a uniform allocation at equal memory.
    assert fpr_oracle < fpr_uniform
    assert fpr_cms < fpr_uniform

    # A realistic online estimator shouldn't beat the oracle; some gap is
    # expected, but it should be small relative to the full uniform->oracle
    # gap given a reasonably sized sketch (width=500) and training stream.
    assert fpr_cms >= fpr_oracle
    gap_closed = (fpr_uniform - fpr_cms) / (fpr_uniform - fpr_oracle)
    assert gap_closed > 0.5


def test_empirical_query_sampling_agrees_with_exact_expectation():
    """Sanity check that sampling queries proportional to popularity (as
    experiments/run_experiments.py does) converges to the same number the
    exact weighted-sum formula predicts."""
    rng = np.random.default_rng(11)
    num_keys = 500
    n_members = 100
    skew = 1.2
    k = 6
    num_bits = 3000

    weight_arr = popularity_weights(num_keys, skew, rng=rng)
    keys = [f"k{i}" for i in range(num_keys)]
    scores = dict(zip(keys, weight_arr))
    member_idx = rng.choice(num_keys, size=n_members, replace=False)
    members = set(keys[i] for i in member_idx)

    bf = BloomFilter(num_bits=num_bits, seed=5)
    for m in members:
        bf.insert(m, k)

    negative_keys = [key for key in keys if key not in members]
    negative_weights = np.array([scores[key] for key in negative_keys])
    negative_weights = negative_weights / negative_weights.sum()

    exact_fpr = sum(w * bf.query(key, k) for w, key in zip(negative_weights, negative_keys))

    sampled_idx = rng.choice(len(negative_keys), size=50_000, p=negative_weights)
    empirical_fpr = np.mean([bf.query(negative_keys[i], k) for i in sampled_idx])

    assert abs(exact_fpr - empirical_fpr) < 0.02
