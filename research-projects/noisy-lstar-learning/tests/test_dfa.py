import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dfa import DFA, equivalent, equivalence_counterexample, random_dfa, is_empty


def dfa_contains_ab():
    """Alphabet {0,1} = {a,b}. Accepts strings containing substring "ab" (0 then 1)."""
    # states: 0 = start/no progress, 1 = saw 'a', 2 = saw "ab" (accepting, sink-accept)
    transitions = [
        [1, 0],  # state 0: on 'a'->1, on 'b'->0
        [1, 2],  # state 1: on 'a'->1, on 'b'->2
        [2, 2],  # state 2: sink accepting
    ]
    return DFA(3, 2, transitions, start=0, accepting=frozenset({2}))


def dfa_even_zeros():
    """Alphabet {0,1}. Accepts strings with an even number of 0s (including zero)."""
    transitions = [
        [1, 0],  # state 0 (even so far): on 0 -> 1 (odd), on 1 -> 0
        [0, 1],  # state 1 (odd so far): on 0 -> 0 (even), on 1 -> 1
    ]
    return DFA(2, 2, transitions, start=0, accepting=frozenset({0}))


def test_accepts_matches_definition():
    dfa = dfa_contains_ab()
    assert dfa.accepts([0, 1])
    assert dfa.accepts([1, 1, 0, 1])
    assert not dfa.accepts([])
    assert dfa.accepts([0, 0, 1, 1])
    assert not dfa.accepts([1, 1, 1])
    assert not dfa.accepts([0])


def test_even_zeros_definition():
    dfa = dfa_even_zeros()
    assert dfa.accepts([])
    assert dfa.accepts([0, 0])
    assert dfa.accepts([1, 1, 1])
    assert not dfa.accepts([0])
    assert not dfa.accepts([0, 0, 0])


def test_equivalent_reflexive():
    dfa = dfa_contains_ab()
    assert equivalent(dfa, dfa)


def test_equivalent_detects_difference():
    a = dfa_contains_ab()
    b = dfa_even_zeros()
    assert not equivalent(a, b)
    cx = equivalence_counterexample(a, b)
    assert cx is not None
    assert a.accepts(cx) != b.accepts(cx)


def test_equivalent_up_to_isomorphism():
    # Same language as dfa_contains_ab but with states renumbered / an extra
    # unreachable state, and a redundant duplicate accepting state.
    transitions = [
        [1, 0],
        [1, 2],
        [2, 2],
        [3, 3],  # unreachable dead state
    ]
    b = DFA(4, 2, transitions, start=0, accepting=frozenset({2}))
    a = dfa_contains_ab()
    assert equivalent(a, b)


def test_is_empty():
    transitions = [[0, 0]]
    empty_dfa = DFA(1, 2, transitions, start=0, accepting=frozenset())
    assert is_empty(empty_dfa)
    full_dfa = DFA(1, 2, transitions, start=0, accepting=frozenset({0}))
    assert not is_empty(full_dfa)


def test_random_dfa_is_well_formed_and_nontrivial():
    rng = random.Random(42)
    for _ in range(20):
        dfa = random_dfa(num_states=6, alphabet_size=2, rng=rng)
        assert dfa.num_states == 6
        assert not dfa.is_trivial()
        # every transition must land in range; DFA.__post_init__ already checks this,
        # so just constructing it without raising is part of the assertion.


def test_random_dfa_reproducible_with_seed():
    dfa_a = random_dfa(8, 2, random.Random(7))
    dfa_b = random_dfa(8, 2, random.Random(7))
    assert dfa_a.transitions == dfa_b.transitions
    assert dfa_a.accepting == dfa_b.accepting
