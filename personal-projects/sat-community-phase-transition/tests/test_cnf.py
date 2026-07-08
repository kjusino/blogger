import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cnf import CNF, community_3sat, partition_variables, random_3sat


def test_random_3sat_shape():
    cnf = random_3sat(n_vars=20, n_clauses=80, rng=random.Random(0))
    assert cnf.n_vars == 20
    assert cnf.n_clauses == 80
    assert cnf.alpha == 4.0
    assert cnf.communities is None


def test_random_3sat_clauses_well_formed():
    cnf = random_3sat(n_vars=15, n_clauses=50, rng=random.Random(1))
    for clause in cnf.clauses:
        assert len(clause) == 3
        variables = [abs(lit) for lit in clause]
        assert len(set(variables)) == 3, "clause must have 3 distinct variables"
        for v in variables:
            assert 1 <= v <= 15
        for lit in clause:
            assert lit != 0


def test_random_3sat_rejects_too_few_variables():
    try:
        random_3sat(n_vars=2, n_clauses=5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_partition_variables_covers_all_and_is_disjoint():
    communities = partition_variables(n_vars=31, n_communities=4, rng=random.Random(2))
    assert len(communities) == 4
    all_vars = [v for c in communities for v in c]
    assert sorted(all_vars) == list(range(1, 32))
    sizes = sorted(len(c) for c in communities)
    assert sizes[-1] - sizes[0] <= 1, "community sizes should be near-equal"


def test_community_3sat_mu_zero_all_clauses_intra_community():
    n_vars, n_communities = 30, 3
    cnf = community_3sat(
        n_vars=n_vars, n_clauses=200, n_communities=n_communities, mu=0.0,
        rng=random.Random(3),
    )
    membership = {}
    for idx, comm in enumerate(cnf.communities):
        for v in comm:
            membership[v] = idx

    intra_or_merged = 0
    for clause in cnf.clauses:
        comms = {membership[abs(lit)] for lit in clause}
        # with mu=0 every clause is drawn from one community OR (if that
        # community was too small) the union of two adjacent communities.
        if len(comms) <= 2:
            intra_or_merged += 1
    assert intra_or_merged == len(cnf.clauses)


def test_community_3sat_mu_one_matches_random_distribution():
    # mu=1.0 should behave like uniform random 3-SAT: every variable should
    # appear roughly equally often across many clauses.
    n_vars = 12
    cnf = community_3sat(
        n_vars=n_vars, n_clauses=4000, n_communities=3, mu=1.0,
        rng=random.Random(4),
    )
    counts = [0] * (n_vars + 1)
    for clause in cnf.clauses:
        for lit in clause:
            counts[abs(lit)] += 1
    observed = counts[1:]
    mean = sum(observed) / len(observed)
    max_dev = max(abs(c - mean) for c in observed)
    assert max_dev / mean < 0.25, "mu=1.0 should draw near-uniformly over all variables"


def test_community_3sat_rejects_invalid_mu():
    try:
        community_3sat(n_vars=10, n_clauses=5, n_communities=2, mu=1.5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_cnf_alpha_property():
    cnf = CNF(n_vars=10, clauses=[(1, 2, 3)] * 43)
    assert cnf.alpha == 4.3
