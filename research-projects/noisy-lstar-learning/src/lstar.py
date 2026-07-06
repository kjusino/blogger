"""Angluin's L* algorithm for exact learning of regular languages from a
membership oracle and an equivalence oracle, via an observation table."""

from __future__ import annotations

from .dfa import DFA


class LStarNonConvergence(Exception):
    """Raised when the observation table grows past the configured resource
    caps without the equivalence oracle confirming a hypothesis. Under a noisy
    membership oracle this is a legitimate learning failure, not a bug."""


def _build_hypothesis(S, E, mem, alphabet_size: int) -> DFA:
    row_cache = {}

    def row(s):
        if s not in row_cache:
            row_cache[s] = tuple(mem(s + e) for e in E)
        return row_cache[s]

    row_to_id = {}
    rep = {}
    for s in S:
        r = row(s)
        if r not in row_to_id:
            row_to_id[r] = len(row_to_id)
            rep[r] = s

    num_states = len(row_to_id)
    transitions = [[0] * alphabet_size for _ in range(num_states)]
    for r, sid in row_to_id.items():
        s = rep[r]
        for a in range(alphabet_size):
            transitions[sid][a] = row_to_id[row(s + (a,))]

    idx_eps = E.index(())
    start = row_to_id[row(())]
    accepting = frozenset(sid for r, sid in row_to_id.items() if r[idx_eps])
    return DFA(num_states, alphabet_size, transitions, start=start, accepting=accepting)


def learn(membership_oracle, equivalence_oracle, alphabet_size: int,
          max_states: int = 60, max_equivalence_queries: int = 200):
    """Run L* to convergence.

    Returns (hypothesis, stats) where stats is a dict with S, E sizes and
    equivalence-query count. Raises LStarNonConvergence if resource caps are
    exceeded before the equivalence oracle accepts a hypothesis.
    """
    S = [()]
    E = [()]
    T = {}

    def mem(word):
        if word not in T:
            T[word] = membership_oracle.query(word)
        return T[word]

    def row(s):
        return tuple(mem(s + e) for e in E)

    equivalence_queries = 0

    while True:
        changed = True
        while changed:
            changed = False

            s_rows = {row(s): s for s in S}
            for s in list(S):
                for a in range(alphabet_size):
                    t = s + (a,)
                    if row(t) not in s_rows:
                        S.append(t)
                        s_rows[row(t)] = t
                        changed = True

            if len(S) > max_states:
                raise LStarNonConvergence(
                    f"observation table exceeded max_states={max_states} (|S|={len(S)})"
                )

            row_groups = {}
            for s in S:
                row_groups.setdefault(row(s), []).append(s)

            for group in row_groups.values():
                if len(group) < 2:
                    continue
                s1 = group[0]
                inconsistent = False
                for s2 in group[1:]:
                    for a in range(alphabet_size):
                        if row(s1 + (a,)) != row(s2 + (a,)):
                            for e in E:
                                if mem(s1 + (a,) + e) != mem(s2 + (a,) + e):
                                    new_e = (a,) + e
                                    if new_e not in E:
                                        E.append(new_e)
                                        changed = True
                                        inconsistent = True
                                    break
                            break
                    if inconsistent:
                        break
                if inconsistent:
                    break

        hypothesis = _build_hypothesis(S, E, mem, alphabet_size)

        if equivalence_queries >= max_equivalence_queries:
            raise LStarNonConvergence(
                f"exceeded max_equivalence_queries={max_equivalence_queries}"
            )
        equivalence_queries += 1
        counterexample = equivalence_oracle.query(hypothesis)

        if counterexample is None:
            return hypothesis, {
                "num_states": hypothesis.num_states,
                "S_size": len(S),
                "E_size": len(E),
                "equivalence_queries": equivalence_queries,
            }

        cx = tuple(counterexample)
        for i in range(len(cx) + 1):
            prefix = cx[:i]
            if prefix not in S:
                S.append(prefix)
