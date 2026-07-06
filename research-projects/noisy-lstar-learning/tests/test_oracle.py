import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dfa import DFA
from src.oracle import (
    MembershipOracle,
    NoisyMembershipOracle,
    RedundantMembershipOracle,
    ExactEquivalenceOracle,
    hoeffding_repetitions,
)


def toy_dfa():
    # Accepts strings ending in 1. States: 0 = last-was-0-or-empty, 1 = last-was-1.
    transitions = [[0, 1], [0, 1]]
    return DFA(2, 2, transitions, start=0, accepting=frozenset({1}))


def test_membership_oracle_matches_target_and_counts_queries():
    dfa = toy_dfa()
    oracle = MembershipOracle(dfa)
    assert oracle.query([0, 1]) is True
    assert oracle.query([1, 0]) is False
    assert oracle.raw_queries == 2


def test_noisy_oracle_zero_noise_is_exact():
    dfa = toy_dfa()
    rng = random.Random(0)
    noisy = NoisyMembershipOracle(dfa, noise_rate=0.0, rng=rng)
    for word in ([], [0], [1], [0, 1], [1, 0, 1]):
        assert noisy.query_once(word) == dfa.accepts(word)


def test_noisy_oracle_flip_rate_matches_noise_rate_statistically():
    dfa = toy_dfa()
    rng = random.Random(1)
    noise_rate = 0.3
    noisy = NoisyMembershipOracle(dfa, noise_rate=noise_rate, rng=rng)
    word = [0, 1]  # truth = True
    n = 20000
    flips = sum(1 for _ in range(n) if noisy.query_once(word) != dfa.accepts(word))
    observed_rate = flips / n
    assert abs(observed_rate - noise_rate) < 0.02  # loose statistical tolerance


def test_noisy_oracle_rejects_noise_rate_half_or_above():
    dfa = toy_dfa()
    with pytest.raises(ValueError):
        NoisyMembershipOracle(dfa, noise_rate=0.5, rng=random.Random(0))


def test_hoeffding_repetitions_zero_noise_is_one_query():
    assert hoeffding_repetitions(noise_rate=0.0, delta=1e-6) == 1


def test_hoeffding_repetitions_increases_with_noise():
    k_low = hoeffding_repetitions(noise_rate=0.1, delta=1e-4)
    k_high = hoeffding_repetitions(noise_rate=0.4, delta=1e-4)
    assert k_high > k_low
    assert k_low % 2 == 1 and k_high % 2 == 1  # always odd, to avoid ties


def test_hoeffding_repetitions_matches_closed_form():
    p, delta = 0.2, 1e-5
    k = hoeffding_repetitions(p, delta)
    margin = 0.5 - p
    expected = math.ceil(math.log(1.0 / delta) / (2 * margin * margin))
    if expected % 2 == 0:
        expected += 1
    assert k == expected


def test_redundant_oracle_majority_vote_corrects_moderate_noise():
    dfa = toy_dfa()
    rng = random.Random(2)
    noisy = NoisyMembershipOracle(dfa, noise_rate=0.3, rng=rng)
    k = hoeffding_repetitions(0.3, delta=1e-6)
    redundant = RedundantMembershipOracle(noisy, repetitions=k)
    # Across many distinct words, majority-vote answers should match ground truth
    # far more often than a single noisy query would.
    words = [(), (0,), (1,), (0, 1), (1, 0), (1, 1), (0, 0), (0, 1, 1), (1, 0, 1)]
    correct = sum(1 for w in words if redundant.query(w) == dfa.accepts(list(w)))
    assert correct == len(words)


def test_redundant_oracle_caches_distinct_queries():
    dfa = toy_dfa()
    rng = random.Random(3)
    noisy = NoisyMembershipOracle(dfa, noise_rate=0.1, rng=rng)
    redundant = RedundantMembershipOracle(noisy, repetitions=5)
    redundant.query((0, 1))
    redundant.query((0, 1))
    redundant.query((0, 1))
    assert redundant.distinct_queries == 1
    assert redundant.raw_queries == 5  # only the first call issued raw queries


def test_exact_equivalence_oracle_finds_counterexample_and_counts_queries():
    target = toy_dfa()
    wrong = DFA(1, 2, [[0, 0]], start=0, accepting=frozenset())  # rejects everything
    oracle = ExactEquivalenceOracle(target)
    cx = oracle.query(wrong)
    assert cx is not None
    assert target.accepts(cx) != wrong.accepts(cx)
    assert oracle.num_queries == 1

    cx_none = oracle.query(target)
    assert cx_none is None
    assert oracle.num_queries == 2
