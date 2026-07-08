"""Generators for random and community-structured 3-SAT instances.

A CNF formula is represented as a list of clauses, where each clause is a
tuple of 3 nonzero ints (literals). A positive int `v` means variable `v`
appears un-negated; a negative int `-v` means variable `v` appears negated.
Variables are numbered 1..n_vars.
"""

import random


class CNF:
    """A 3-SAT CNF formula plus the community structure it was generated with."""

    def __init__(self, n_vars, clauses, communities=None):
        self.n_vars = n_vars
        self.clauses = clauses
        # `communities`: list of lists, each inner list the variable ids (1-indexed)
        # belonging to that planted community. None for uniform-random instances.
        self.communities = communities

    @property
    def n_clauses(self):
        return len(self.clauses)

    @property
    def alpha(self):
        """Clause-to-variable ratio, the standard control parameter for the
        random-3-SAT phase transition."""
        return self.n_clauses / self.n_vars


def _random_clause(rng, pool):
    """Draw one 3-SAT clause uniformly from a pool of variable ids, with
    independently randomized literal signs. Requires len(pool) >= 3."""
    vars_ = rng.sample(pool, 3)
    return tuple(v if rng.random() < 0.5 else -v for v in vars_)


def partition_variables(n_vars, n_communities, rng):
    """Split variables 1..n_vars into n_communities near-equal-sized groups."""
    if n_communities < 1:
        raise ValueError("n_communities must be >= 1")
    var_ids = list(range(1, n_vars + 1))
    rng.shuffle(var_ids)
    communities = [[] for _ in range(n_communities)]
    for i, v in enumerate(var_ids):
        communities[i % n_communities].append(v)
    return communities


def random_3sat(n_vars, n_clauses, rng=None):
    """Uniform-random 3-SAT: every clause drawn from all n_vars variables.
    Equivalent to the mu=1.0 case of community_3sat below."""
    if rng is None:
        rng = random.Random()
    if n_vars < 3:
        raise ValueError("n_vars must be >= 3")
    pool = list(range(1, n_vars + 1))
    clauses = [_random_clause(rng, pool) for _ in range(n_clauses)]
    return CNF(n_vars, clauses, communities=None)


def community_3sat(n_vars, n_clauses, n_communities, mu, rng=None):
    """Community-structured 3-SAT via a controllable mixing parameter.

    `mu` in [0, 1] is the probability that a given clause is drawn as a
    "random" clause (its 3 variables uniform over ALL variables). With
    probability (1 - mu) the clause is instead a "local" clause: one
    community is chosen uniformly, and all 3 variables are drawn from
    that community only (falling back to the union of two communities if
    the chosen one is too small to supply 3 distinct variables).

    mu=1.0 recovers uniform-random 3-SAT. mu=0.0 produces formulas whose
    variable-interaction graph is maximally explained by the planted
    partition (every clause lives inside one community).
    """
    if rng is None:
        rng = random.Random()
    if n_vars < 3:
        raise ValueError("n_vars must be >= 3")
    if not 0.0 <= mu <= 1.0:
        raise ValueError("mu must be in [0, 1]")

    communities = partition_variables(n_vars, n_communities, rng)
    full_pool = list(range(1, n_vars + 1))

    clauses = []
    for _ in range(n_clauses):
        if rng.random() < mu:
            clauses.append(_random_clause(rng, full_pool))
            continue

        idx = rng.randrange(n_communities)
        pool = communities[idx]
        if len(pool) < 3:
            # merge with a neighboring community so we can still draw 3
            # distinct variables; keeps the generator well-defined for
            # small n_vars / large n_communities.
            other = communities[(idx + 1) % n_communities]
            pool = list(pool) + list(other)
        clauses.append(_random_clause(rng, pool))

    return CNF(n_vars, clauses, communities=communities)


def decompose_by_community(cnf):
    """Split a formula whose every clause lies within a single planted
    community (e.g. a community_3sat instance generated with mu=0.0) into
    one independent sub-CNF per community, with variables remapped to a
    contiguous 1..k range.

    Raises ValueError if the formula has no planted communities, or if any
    clause spans more than one community (the formula is then not a
    disjoint union and cannot be decomposed this way).
    """
    if not cnf.communities:
        raise ValueError("cnf has no planted communities")

    membership = {}
    for idx, comm in enumerate(cnf.communities):
        for v in comm:
            membership[v] = idx

    clauses_per_community = [[] for _ in cnf.communities]
    for clause in cnf.clauses:
        comm_ids = {membership[abs(lit)] for lit in clause}
        if len(comm_ids) != 1:
            raise ValueError("clause spans multiple communities; formula is not decomposable")
        clauses_per_community[next(iter(comm_ids))].append(clause)

    sub_cnfs = []
    for idx, comm in enumerate(cnf.communities):
        var_list = sorted(comm)
        remap = {v: i + 1 for i, v in enumerate(var_list)}
        remapped_clauses = [
            tuple(remap[abs(lit)] if lit > 0 else -remap[abs(lit)] for lit in clause)
            for clause in clauses_per_community[idx]
        ]
        sub_cnfs.append(CNF(n_vars=len(var_list), clauses=remapped_clauses))
    return sub_cnfs
