import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cnf import CNF, community_3sat, random_3sat
from src.community import planted_modularity, variable_interaction_graph


def test_vig_node_count_matches_variables():
    cnf = random_3sat(n_vars=10, n_clauses=20, rng=random.Random(0))
    g = variable_interaction_graph(cnf)
    assert g.number_of_nodes() == 10


def test_vig_edges_come_from_clause_cooccurrence():
    cnf = CNF(n_vars=4, clauses=[(1, 2, 3)])
    g = variable_interaction_graph(cnf)
    assert g.has_edge(1, 2)
    assert g.has_edge(1, 3)
    assert g.has_edge(2, 3)
    assert not g.has_edge(1, 4)


def test_vig_edge_weight_counts_cooccurrences():
    # VIG edges are keyed by variable co-occurrence regardless of literal
    # sign, so (1,2,3) and (1,-2,3) both contribute a 1-2 co-occurrence.
    cnf = CNF(n_vars=4, clauses=[(1, 2, 3), (1, -2, 3), (1, 2, 4)])
    g = variable_interaction_graph(cnf)
    assert g[1][2]["weight"] == 3  # co-occurs in all three clauses
    assert g[1][3]["weight"] == 2  # co-occurs in the first two
    assert g[2][3]["weight"] == 2
    assert g[1][4]["weight"] == 1  # only in the third clause
    assert g[2][4]["weight"] == 1


def test_planted_modularity_is_none_without_communities():
    cnf = random_3sat(n_vars=10, n_clauses=20, rng=random.Random(0))
    assert planted_modularity(cnf) is None


def test_planted_modularity_higher_for_more_local_formulas():
    # At mu=0.0 (fully local) the planted partition should explain the VIG
    # far better than at mu=1.0 (fully random), on average.
    n_vars, n_clauses, n_communities = 60, 300, 4

    local_qs = []
    random_qs = []
    for seed in range(8):
        rng = random.Random(seed)
        local_cnf = community_3sat(n_vars, n_clauses, n_communities, mu=0.0, rng=rng)
        local_qs.append(planted_modularity(local_cnf))

        rng2 = random.Random(seed + 1000)
        random_cnf = community_3sat(n_vars, n_clauses, n_communities, mu=1.0, rng=rng2)
        random_qs.append(planted_modularity(random_cnf))

    mean_local = sum(local_qs) / len(local_qs)
    mean_random = sum(random_qs) / len(random_qs)
    assert mean_local > mean_random
    assert mean_local > 0.3  # strongly community-explained
    assert mean_random < 0.1  # near-zero for a random graph vs. an arbitrary partition


def test_planted_modularity_monotonic_in_mu_on_average():
    n_vars, n_clauses, n_communities = 60, 300, 4
    mus = [0.0, 0.3, 0.6, 1.0]
    mean_qs = []
    for mu in mus:
        qs = []
        for seed in range(6):
            rng = random.Random(seed * 17 + int(mu * 100))
            cnf = community_3sat(n_vars, n_clauses, n_communities, mu=mu, rng=rng)
            qs.append(planted_modularity(cnf))
        mean_qs.append(sum(qs) / len(qs))

    # increasing mu (more randomness) should not increase mean modularity
    for i in range(len(mean_qs) - 1):
        assert mean_qs[i] >= mean_qs[i + 1] - 0.05  # allow small sampling noise
