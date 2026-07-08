"""PC-stable algorithm for causal skeleton discovery + CPDAG orientation.

Implements the skeleton-discovery phase of Spirtes-Glymour-Scheines' PC
algorithm in the order-independent "PC-stable" variant of Colombo &
Maathuis (2014): at each conditioning-set size l, the neighbor sets used
for all tests are frozen at the *start* of the level, and edge deletions
are applied only after every pair has been tested at that level. This
avoids the original PC algorithm's dependence on variable ordering.

Followed by v-structure (collider) orientation and a minimal subset of
Meek's orientation rules (R1) to produce a CPDAG.
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .partial_correlation import fisher_z_test, partial_correlation


@dataclass
class PCResult:
    p: int
    skeleton: np.ndarray  # symmetric boolean adjacency
    sepsets: dict = field(default_factory=dict)  # {frozenset({i,j}): tuple(cond_set)}
    directed: np.ndarray = None  # directed[i,j] = True means i -> j oriented
    n_ci_tests: int = 0
    # {frozenset({i,j}): float}, min |partial correlation| seen across every
    # test where the pair was judged dependent (i.e. not separated). Only
    # meaningful when the search was run on a population covariance matrix
    # (see estimate_skeleton_oracle) -- it is then the weakest signal the
    # algorithm's own test sequence needs to detect to keep that edge.
    margins: dict = field(default_factory=dict)


def _neighbors(adj: np.ndarray, i: int) -> set:
    return set(np.nonzero(adj[i])[0].tolist())


def _skeleton_search(cov: np.ndarray, decide_fn, max_cond_set: int = None) -> PCResult:
    """PC-stable skeleton search, generic over the independence decision rule.

    decide_fn(r, cond_set_size) -> bool, True means "judged independent",
    given the partial correlation r already computed from cov.
    """
    p = cov.shape[0]
    adj = np.ones((p, p), dtype=bool)
    np.fill_diagonal(adj, False)
    sepsets = {}
    margins = {}
    n_tests = 0

    l = 0
    max_cond_set = p - 2 if max_cond_set is None else max_cond_set
    while True:
        frozen_adj = adj.copy()
        any_tested = False
        removed_this_level = set()
        # Iterate ORDERED pairs (i, j): the true separating set for an edge
        # can live in neighbors(i) or in neighbors(j), and those neighbor
        # sets differ in general, so both directions must be tried (PC /
        # PC-stable both test from both sides; testing only neighbors(i)
        # would miss separating sets only visible from j's side, silently
        # leaving spurious edges in the skeleton no matter how large n is).
        for i in range(p):
            for j in range(p):
                if i == j or not frozen_adj[i, j] or frozenset((i, j)) in removed_this_level:
                    continue
                neighbors_i = _neighbors(frozen_adj, i) - {j}
                if len(neighbors_i) < l:
                    continue
                any_tested = True
                for cond_set in combinations(sorted(neighbors_i), l):
                    n_tests += 1
                    r = partial_correlation(cov, i, j, cond_set)
                    if decide_fn(r, l):
                        removed_this_level.add(frozenset((i, j)))
                        sepsets[frozenset((i, j))] = cond_set
                        break
                    pair = frozenset((i, j))
                    margins[pair] = min(margins.get(pair, np.inf), abs(r))
        for pair in removed_this_level:
            i, j = tuple(pair)
            adj[i, j] = False
            adj[j, i] = False
        l += 1
        if not any_tested or l > max_cond_set:
            break

    return PCResult(p=p, skeleton=adj, sepsets=sepsets, n_ci_tests=n_tests, margins=margins)


def estimate_skeleton(cov: np.ndarray, n: int, alpha: float, max_cond_set: int = None) -> PCResult:
    """Finite-sample skeleton estimation from an empirical covariance matrix,
    using the Fisher z conditional-independence test."""
    decide_fn = lambda r, l: fisher_z_test(r, n, l, alpha)
    return _skeleton_search(cov, decide_fn, max_cond_set)


def estimate_skeleton_oracle(cov: np.ndarray, eps: float = 1e-6, max_cond_set: int = None) -> PCResult:
    """Population-level ("infinite sample") skeleton estimation: independence
    is declared when the population partial correlation is below eps. Used
    to verify that a generated SEM is faithful before running finite-sample
    experiments on it, and (via PCResult.margins) to measure how weak its
    weakest true-edge signal is."""
    decide_fn = lambda r, l: abs(r) < eps
    return _skeleton_search(cov, decide_fn, max_cond_set)


def min_true_edge_margin(sem_skeleton: np.ndarray, oracle_result: PCResult) -> float:
    """Weakest partial-correlation signal among the SEM's true edges, per
    the oracle run's own test sequence. Returns +inf if the SEM has no
    edges at all."""
    p = sem_skeleton.shape[0]
    margin = np.inf
    for i in range(p):
        for j in range(i + 1, p):
            if sem_skeleton[i, j]:
                pair_margin = oracle_result.margins.get(frozenset((i, j)), np.inf)
                margin = min(margin, pair_margin)
    return margin


def orient_edges(result: PCResult) -> np.ndarray:
    """Orient v-structures (colliders) then apply Meek's rule R1.

    Returns a p x p boolean matrix `directed` where directed[i, j] = True
    means the edge is oriented i -> j. An edge still undirected in the
    CPDAG has directed[i, j] == directed[j, i] == True (both directions
    "possible").
    """
    p = result.p
    skeleton = result.skeleton
    directed = skeleton.copy()  # start: every skeleton edge is bidirected (unoriented)

    adj_list = [set(np.nonzero(skeleton[i])[0].tolist()) for i in range(p)]

    # V-structures: i - k - j unshielded (i, j not adjacent), k not in sepset(i, j)
    # => orient as i -> k <- j.
    for k in range(p):
        neighbors_k = sorted(adj_list[k])
        for i, j in combinations(neighbors_k, 2):
            if skeleton[i, j]:
                continue  # shielded triple, skip
            sep = result.sepsets.get(frozenset((i, j)), ())
            if k not in sep:
                directed[k, i] = False
                directed[k, j] = False

    # Meek's rule R1: if i -> k and k - j unoriented and i, j not adjacent,
    # orient k -> j (avoids creating a new v-structure).
    changed = True
    while changed:
        changed = False
        for k in range(p):
            for j in adj_list[k]:
                if not (directed[k, j] and directed[j, k]):
                    continue  # k - j must currently be unoriented
                for i in adj_list[k]:
                    if i == j:
                        continue
                    i_to_k = directed[i, k] and not directed[k, i]
                    if i_to_k and not skeleton[i, j]:
                        directed[j, k] = False
                        changed = True
                        break

    return directed
