"""Deterministic finite automata: representation, generation, minimization-free
exact equivalence checking via the symmetric-difference product automaton."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field


@dataclass
class DFA:
    """A DFA over an integer alphabet {0, ..., alphabet_size-1}.

    States are ints 0..num_states-1. `transitions[state][symbol] = next_state`.
    """

    num_states: int
    alphabet_size: int
    transitions: list  # transitions[state][symbol] -> state
    start: int
    accepting: frozenset
    unreachable_ok: bool = field(default=False, repr=False)

    def __post_init__(self):
        for state, row in enumerate(self.transitions):
            if len(row) != self.alphabet_size:
                raise ValueError(
                    f"state {state} has {len(row)} transitions, expected {self.alphabet_size}"
                )
            for symbol, dst in enumerate(row):
                if not (0 <= dst < self.num_states):
                    raise ValueError(f"transition ({state},{symbol}) -> {dst} out of range")
        if not (0 <= self.start < self.num_states):
            raise ValueError("start state out of range")
        for s in self.accepting:
            if not (0 <= s < self.num_states):
                raise ValueError("accepting state out of range")

    def step(self, state: int, symbol: int) -> int:
        return self.transitions[state][symbol]

    def run(self, word) -> int:
        state = self.start
        for symbol in word:
            state = self.step(state, symbol)
        return state

    def accepts(self, word) -> bool:
        return self.run(word) in self.accepting

    def reachable_states(self) -> set:
        seen = {self.start}
        queue = deque([self.start])
        while queue:
            s = queue.popleft()
            for symbol in range(self.alphabet_size):
                t = self.transitions[s][symbol]
                if t not in seen:
                    seen.add(t)
                    queue.append(t)
        return seen

    def is_trivial(self) -> bool:
        """True iff the language is empty or all-strings (up to reachable states)."""
        reach = self.reachable_states()
        acc = reach & self.accepting
        return len(acc) == 0 or acc == reach


def random_dfa(num_states: int, alphabet_size: int, rng: random.Random,
               accept_prob: float = 0.5) -> DFA:
    """Generate a random DFA with a fully-defined transition function.

    Retries with fresh randomness if the reachable sub-automaton has a trivial
    (always-accept / always-reject) language, since that would make the
    learning problem degenerate.
    """
    for _ in range(1000):
        transitions = [
            [rng.randrange(num_states) for _ in range(alphabet_size)]
            for _ in range(num_states)
        ]
        accepting = frozenset(s for s in range(num_states) if rng.random() < accept_prob)
        dfa = DFA(num_states, alphabet_size, transitions, start=0, accepting=accepting)
        if not dfa.is_trivial():
            return dfa
    raise RuntimeError("failed to generate a non-trivial random DFA after 1000 attempts")


def product_symmetric_difference(a: DFA, b: DFA) -> DFA:
    """Build the product automaton accepting L(a) XOR L(b)."""
    if a.alphabet_size != b.alphabet_size:
        raise ValueError("alphabet size mismatch")
    alphabet_size = a.alphabet_size

    pair_to_id = {}
    id_to_pair = []

    def get_id(pair):
        if pair not in pair_to_id:
            pair_to_id[pair] = len(id_to_pair)
            id_to_pair.append(pair)
        return pair_to_id[pair]

    start_pair = (a.start, b.start)
    get_id(start_pair)

    transitions = []
    accepting = set()
    i = 0
    while i < len(id_to_pair):
        sa, sb = id_to_pair[i]
        row = []
        for symbol in range(alphabet_size):
            na, nb = a.step(sa, symbol), b.step(sb, symbol)
            row.append(get_id((na, nb)))
        transitions.append(row)
        if (sa in a.accepting) != (sb in b.accepting):
            accepting.add(i)
        i += 1

    return DFA(len(id_to_pair), alphabet_size, transitions, start=0,
               accepting=frozenset(accepting))


def is_empty(dfa: DFA) -> bool:
    """True iff dfa accepts no string (checked over reachable states only)."""
    reach = dfa.reachable_states()
    return len(reach & dfa.accepting) == 0


def find_witness(dfa: DFA) -> list:
    """Return a shortest word accepted by dfa, or None if L(dfa) is empty."""
    if dfa.start in dfa.accepting:
        return []
    visited = {dfa.start}
    queue = deque([(dfa.start, [])])
    while queue:
        state, path = queue.popleft()
        for symbol in range(dfa.alphabet_size):
            nxt = dfa.step(state, symbol)
            if nxt in dfa.accepting:
                return path + [symbol]
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [symbol]))
    return None


def equivalent(a: DFA, b: DFA) -> bool:
    """Exact language equivalence via the symmetric-difference product automaton."""
    return is_empty(product_symmetric_difference(a, b))


def equivalence_counterexample(a: DFA, b: DFA):
    """Return a word in the symmetric difference of L(a) and L(b), or None if equivalent."""
    return find_witness(product_symmetric_difference(a, b))
