"""Bipartite graph generators and the local-search move used to *discover*
adversarial (hard-for-RANKING) instances rather than relying on one
hand-picked textbook example.
"""

from .matching import BipartiteGraph, max_matching_size


def complete_bipartite(n):
    """K_{n,n}: every algorithm gets a perfect matching trivially -- a
    sanity/control instance."""
    edges = [(i, j) for i in range(n) for j in range(n)]
    return BipartiteGraph(n, n, edges)


def random_bipartite(n, p, rng):
    """G(n,n,p): each of the n^2 possible edges present independently with
    probability p -- a "typical", non-adversarial control instance."""
    edges = [(i, j) for i in range(n) for j in range(n) if rng.random() < p]
    return BipartiteGraph(n, n, edges)


def staircase_graph(n):
    """A simple structured instance: right-vertex j is adjacent to left
    vertices 0..j. Has a perfect matching (i <- i), but greedily consuming
    low-indexed left vertices early starves later right vertices of low-rank
    options. Used only as a *starting point* for local search, not claimed
    to be the worst possible instance -- the search in `search.py` is what
    actually certifies how hard an instance is for a given algorithm.
    """
    edges = [(i, j) for j in range(n) for i in range(j + 1)]
    return BipartiteGraph(n, n, edges)


def perfect_matching_graph(n, rng):
    """A uniformly random permutation matching (i, sigma(i)) plus nothing
    else -- the unique-perfect-matching control instance used in tests."""
    perm = list(rng.permutation(n))
    edges = [(i, perm[i]) for i in range(n)]
    return BipartiteGraph(n, n, edges)


def random_edge_flip(graph, rng):
    """Propose toggling one uniformly random potential edge. Returns a new
    BipartiteGraph (input is left untouched)."""
    i = int(rng.integers(0, graph.n_left))
    j = int(rng.integers(0, graph.n_right))
    return graph.with_edge_toggled(i, j)


def perturb_keep_perfect_matching(graph, rng, max_attempts=40):
    """Propose a random edge flip and only return it if the result still
    admits a perfect matching (size == n_left == n_right). Retries up to
    `max_attempts` times; falls back to returning the input graph unchanged
    if no valid flip is found (rare, only near the empty/complete extremes).
    This is the move set for the adversarial search in `search.py`: we only
    want to explore instances where the *offline optimum* is a perfect
    matching, so a low online competitive ratio is attributable to the
    algorithm's online-ness, not to there being no good matching at all.
    """
    n = graph.n_left
    assert graph.n_left == graph.n_right, "perfect-matching search requires n_left == n_right"
    for _ in range(max_attempts):
        candidate = random_edge_flip(graph, rng)
        if max_matching_size(candidate) == n:
            return candidate
    return graph


def perturb_batch_keep_perfect_matching(graph, rng, batch_size, max_attempts=40):
    """Like `perturb_keep_perfect_matching`, but flips `batch_size` edges at
    once before checking the perfect-matching constraint. Large simultaneous
    batches let the search escape local optima that single-edge flips
    cannot (e.g. dense random graphs, whose competitive ratio is smooth in
    any *one* edge but only drops once enough shared/contested neighborhoods
    are built up together)."""
    n = graph.n_left
    assert graph.n_left == graph.n_right, "perfect-matching search requires n_left == n_right"
    for _ in range(max_attempts):
        candidate = graph
        for _ in range(batch_size):
            candidate = random_edge_flip(candidate, rng)
        if max_matching_size(candidate) == n:
            return candidate
    return graph


def random_dense_graph_with_perfect_matching(n, density, rng, max_attempts=500):
    """A random G(n,n,density) graph, resampled until it happens to admit a
    perfect matching (dense random graphs almost always do; this is just a
    guard against the rare unlucky draw). Used as one of several random
    restarts for the adversarial-instance search -- dense random graphs
    already have far more RANKING-relevant "choice" and contention than the
    fully-rigid `staircase_graph`, so they make far better starting points.
    """
    for _ in range(max_attempts):
        candidate = random_bipartite(n, density, rng)
        if max_matching_size(candidate) == n:
            return candidate
    raise RuntimeError(f"no perfect matching found after {max_attempts} draws "
                        f"at density {density} -- try a higher density")
