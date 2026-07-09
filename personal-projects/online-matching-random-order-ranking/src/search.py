"""Monte-Carlo evaluation of online-matching competitive ratios, and a
simulated-annealing local search that *discovers* hard instances for RANKING
under a fixed (adversarial) arrival order, instead of relying on one
memorized textbook worst-case graph.

Why search instead of citing a fixed construction: Karp-Vazirani-Vazirani
(1990) prove RANKING's competitive ratio is *exactly* 1-1/e in the worst
case over ALL bipartite graphs and rankings, but the specific graph that
witnesses tightness is one detail of their proof, not the theorem's content.
Searching for a hard instance ourselves and checking it against the
theorem's floor is a stronger, more falsifiable test of the *theorem* than
reproducing a graph from a paper: if our search ever finds an instance
driving RANKING's true (large-sample) mean ratio below 1-1/e, that is a
genuine problem (either in our implementation or, extremely unlikely, in the
theorem), not an artifact of picking the "wrong" graph.
"""

import numpy as np

from .graphs import (perturb_batch_keep_perfect_matching,
                      random_dense_graph_with_perfect_matching,
                      staircase_graph)
from .matching import (greedy_online_matching, matching_size,
                        max_matching_size, ranking_online_matching)

ONE_MINUS_INV_E = 1.0 - 1.0 / np.e  # ~0.6321, the n -> infinity worst-case ratio


def exact_finite_floor(n):
    """RANKING's proven *finite-n* competitive-ratio guarantee is
    1 - (1-1/n)^n, not the flat asymptotic constant 1-1/e. (1-1/n)^n
    increases monotonically to 1/e as n -> infinity, so this exact floor
    *decreases* monotonically to 1-1/e and is strictly above it for every
    finite n -- e.g. exact_finite_floor(4) ~= 0.684, not 0.632. Comparing
    empirical means against this sharper, n-dependent floor is a strictly
    stronger correctness check than comparing against ONE_MINUS_INV_E, and
    it is what KVV's theorem actually guarantees at finite n (the flat
    1-1/e constant is only the limiting case)."""
    return 1.0 - (1.0 - 1.0 / n) ** n


def ranking_ratios(graph, arrival_order, rng, n_trials, opt_size=None):
    """Draw `n_trials` independent random rankings, run RANKING under the
    given arrival order on each, and return the list of competitive ratios
    (achieved / optimal)."""
    if opt_size is None:
        opt_size = max_matching_size(graph)
    if opt_size == 0:
        return [1.0] * n_trials
    ratios = []
    for _ in range(n_trials):
        rank = rng.permutation(graph.n_left)
        match = ranking_online_matching(graph, rank, arrival_order)
        ratios.append(matching_size(match) / opt_size)
    return ratios


def mean_ranking_ratio(graph, arrival_order, rng, n_trials, opt_size=None):
    ratios = ranking_ratios(graph, arrival_order, rng, n_trials, opt_size)
    return float(np.mean(ratios)), ratios


def ranking_ratio_single(graph, arrival_order, rng, opt_size):
    """One RANKING trial (fresh random rank) under the given arrival order.
    Used when both the rank *and* the arrival order are re-randomized per
    trial (the random-order-model evaluation in `experiment.py`)."""
    if opt_size == 0:
        return 1.0
    rank = rng.permutation(graph.n_left)
    match = ranking_online_matching(graph, rank, arrival_order)
    return matching_size(match) / opt_size


def greedy_ratio(graph, arrival_order, opt_size=None):
    """Greedy has no internal randomness -- one arrival order gives one
    ratio."""
    if opt_size is None:
        opt_size = max_matching_size(graph)
    if opt_size == 0:
        return 1.0
    match = greedy_online_matching(graph, arrival_order)
    return matching_size(match) / opt_size


def greedy_ratios_random_orders(graph, rng, n_trials, opt_size=None):
    """Greedy's ratio under `n_trials` independent uniformly random arrival
    orders (used for the greedy-under-ROM comparison)."""
    if opt_size is None:
        opt_size = max_matching_size(graph)
    n_right = graph.n_right
    ratios = []
    for _ in range(n_trials):
        order = rng.permutation(n_right)
        ratios.append(greedy_ratio(graph, order, opt_size))
    return ratios


RESTART_DENSITIES = (0.15, 0.3, 0.45, 0.6)


def _pick_best_start(n, arrival_order, rng, trials_per_eval, restart_densities):
    """Screen the fully-rigid staircase graph plus one random dense graph
    per density in `restart_densities`, each with a quick Monte-Carlo
    evaluation, and return the one with the lowest score. Dense random
    graphs already create far more contention (shared neighborhoods) for
    RANKING than the staircase graph, whose specific arrival order gives the
    algorithm no choice at all (ratio is provably 1.0 -- see
    tests/test_graphs.py); screening several densities avoids anchoring the
    whole search on a bad, choice-free starting point.
    """
    candidates = [staircase_graph(n)]
    for density in restart_densities:
        candidates.append(random_dense_graph_with_perfect_matching(n, density, rng))

    best_graph, best_score = None, None
    for candidate in candidates:
        score, _ = mean_ranking_ratio(candidate, arrival_order, rng,
                                       trials_per_eval, opt_size=n)
        if best_score is None or score < best_score:
            best_graph, best_score = candidate, score
    return best_graph, best_score


def search_worst_case_graph(n, arrival_order, rng, n_iterations=150,
                             trials_per_eval=40, init_graph=None,
                             initial_temp=0.05, cooling=0.97,
                             initial_batch_size=None, batch_cooling=0.93,
                             restart_densities=RESTART_DENSITIES):
    """Simulated annealing over n-vs-n bipartite graphs with a guaranteed
    perfect matching, minimizing RANKING's Monte-Carlo mean competitive
    ratio under the fixed `arrival_order`.

    Initialization: if `init_graph` is not given, screens the staircase
    graph and one random dense graph per entry in `restart_densities`
    (`_pick_best_start`) and starts from whichever scores lowest.

    Move set: `perturb_batch_keep_perfect_matching`, flipping a *batch* of
    edges at once (rejected as a whole if the result loses the perfect
    matching). The batch size starts at `initial_batch_size` (default
    max(1, n // 8)) and decays geometrically by `batch_cooling` per
    iteration towards 1 -- large simultaneous flips early let the search
    build up the shared-neighborhood contention structure that makes
    RANKING fail (a single edge flip changes the ratio too little to escape
    a smooth, easy-instance neighborhood), while single-edge flips late let
    it fine-tune. Acceptance: always accept improvements; accept worsening
    moves with probability exp(-delta/temp), temp decaying geometrically
    (`cooling` per iteration) -- standard simulated annealing.

    Returns (best_graph, best_score, history); history[t] is the best score
    found up to and including iteration t (monotone non-increasing).
    """
    if init_graph is not None:
        graph = init_graph
        if max_matching_size(graph) != n:
            raise ValueError("initial graph must have a perfect matching")
        score, _ = mean_ranking_ratio(graph, arrival_order, rng, trials_per_eval, opt_size=n)
    else:
        graph, score = _pick_best_start(n, arrival_order, rng, trials_per_eval, restart_densities)

    best_graph, best_score = graph, score
    history = [best_score]
    temp = initial_temp
    batch_size = initial_batch_size if initial_batch_size is not None else max(1, n // 8)

    for _ in range(n_iterations):
        candidate = perturb_batch_keep_perfect_matching(graph, rng, max(1, round(batch_size)))
        cand_score, _ = mean_ranking_ratio(candidate, arrival_order, rng,
                                            trials_per_eval, opt_size=n)
        delta = cand_score - score
        accept = delta <= 0 or rng.random() < np.exp(-delta / max(temp, 1e-9))
        if accept:
            graph, score = candidate, cand_score
        if score < best_score:
            best_graph, best_score = graph, score
        history.append(best_score)
        temp *= cooling
        batch_size = max(1.0, batch_size * batch_cooling)

    return best_graph, best_score, history
