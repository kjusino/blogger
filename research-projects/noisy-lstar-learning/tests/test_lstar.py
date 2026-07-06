import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dfa import DFA, equivalent, random_dfa
from src.oracle import (
    MembershipOracle,
    ExactEquivalenceOracle,
    NoisyMembershipOracle,
    RedundantMembershipOracle,
    hoeffding_repetitions,
)
from src.lstar import learn, LStarNonConvergence


def dfa_contains_ab():
    transitions = [[1, 0], [1, 2], [2, 2]]
    return DFA(3, 2, transitions, start=0, accepting=frozenset({2}))


def dfa_even_zeros():
    transitions = [[1, 0], [0, 1]]
    return DFA(2, 2, transitions, start=0, accepting=frozenset({0}))


def dfa_divisible_by_3_binary():
    # Accepts binary strings (MSB first) representing a number divisible by 3.
    # state = (value so far) mod 3.
    transitions = []
    for s in range(3):
        row = [(2 * s) % 3, (2 * s + 1) % 3]  # append bit b: new = 2*s + b (mod 3)
        transitions.append(row)
    return DFA(3, 2, transitions, start=0, accepting=frozenset({0}))


@pytest.mark.parametrize("target_factory", [dfa_contains_ab, dfa_even_zeros, dfa_divisible_by_3_binary])
def test_lstar_learns_exact_language_noiselessly(target_factory):
    target = target_factory()
    membership = MembershipOracle(target)
    equivalence = ExactEquivalenceOracle(target)
    hypothesis, stats = learn(membership, equivalence, alphabet_size=target.alphabet_size)
    assert equivalent(hypothesis, target)
    assert stats["equivalence_queries"] >= 1


def test_lstar_learns_random_dfas_noiselessly():
    rng = random.Random(123)
    for _ in range(15):
        target = random_dfa(num_states=rng.choice([4, 6, 9]), alphabet_size=2, rng=rng)
        membership = MembershipOracle(target)
        equivalence = ExactEquivalenceOracle(target)
        hypothesis, stats = learn(membership, equivalence, alphabet_size=2)
        assert equivalent(hypothesis, target), f"failed on target with {target.num_states} states"


def test_lstar_hypothesis_is_minimal_or_smaller_equal_to_target():
    # L*'s hypothesis, once equivalent, should never exceed the target's state
    # count by much (it converges to the minimal DFA for the target language).
    target = dfa_divisible_by_3_binary()
    membership = MembershipOracle(target)
    equivalence = ExactEquivalenceOracle(target)
    hypothesis, stats = learn(membership, equivalence, alphabet_size=2)
    assert hypothesis.num_states <= target.num_states


def test_lstar_with_redundant_oracle_recovers_under_moderate_noise():
    target = dfa_contains_ab()
    rng = random.Random(99)
    noise_rate = 0.2
    noisy = NoisyMembershipOracle(target, noise_rate=noise_rate, rng=rng)
    k = hoeffding_repetitions(noise_rate, delta=1e-4)
    redundant = RedundantMembershipOracle(noisy, repetitions=k)
    equivalence = ExactEquivalenceOracle(target)
    hypothesis, stats = learn(redundant, equivalence, alphabet_size=2)
    assert equivalent(hypothesis, target)


def test_lstar_raises_nonconvergence_on_pathological_all_wrong_oracle():
    target = dfa_contains_ab()

    class AlwaysWrongOracle:
        def query(self, word):
            return not target.accepts(word)

    equivalence = ExactEquivalenceOracle(target)
    with pytest.raises(LStarNonConvergence):
        learn(AlwaysWrongOracle(), equivalence, alphabet_size=2,
              max_states=10, max_equivalence_queries=10)
