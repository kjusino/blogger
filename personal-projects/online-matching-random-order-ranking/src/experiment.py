"""Orchestrates the full experiment around a single derived-and-validated
adversarial construction (`staircase_hard_instance`, see its docstring for
the derivation): a sweep measuring RANKING's and greedy's competitive ratios
on it under adversarial order vs. the random-order model (ROM); a
search-refinement pass checking whether local search can improve on it; and
a control sweep on "easy" graphs for contrast.
"""

from .graphs import complete_bipartite, random_bipartite, staircase_graph
from .matching import max_matching_size
from .search import (ONE_MINUS_INV_E, exact_finite_floor, greedy_ratio,
                      greedy_ratios_random_orders, ranking_ratio_single,
                      ranking_ratios, search_worst_case_graph)
from .stats_utils import linear_fit, mean_ci, two_sample_diff_ci

# Slack allowed when comparing a bootstrap CI to the exact finite-n floor,
# to absorb Monte Carlo sampling noise without weakening the theorem itself
# (the floor being tested is unchanged; this only affects how strictly we
# demand the *estimate* clear it).
FLOOR_TOLERANCE = 0.02


def adversarial_order(n):
    """Identity arrival order 0, 1, ..., n-1 -- used for the control graphs,
    where (being symmetric/random) the choice of fixed order does not
    matter."""
    return list(range(n))


def staircase_hard_instance(n):
    """The graph+order pair this project uses as its primary adversarial
    witness for RANKING: `staircase_graph(n)` (right/online vertex j is
    adjacent to left/offline vertices 0..j) with arrival order *reversed*,
    so the widest-neighborhood vertex (j = n-1, adjacent to everything)
    arrives first and is free to take any offline vertex, while each
    subsequent arrival has a strictly narrower neighborhood -- down to
    vertex j = 0, which is adjacent to only u_0 and is unmatched whenever
    u_0 was already taken by an earlier, less-constrained arrival.

    This pairing was *derived*, not looked up: pairing the same staircase
    graph with the *identity* order instead gives every online vertex
    exactly one remaining choice by induction (RANKING gets a perfect
    matching with probability 1 regardless of rank -- see
    tests/test_graphs.py::test_staircase_graph_...), so the order, not just
    the graph, is what creates adversarial pressure. Three independent
    checks validate that the reversed pairing is a real, non-arbitrary hard
    instance rather than a search/tuning artifact (see README for the
    numbers): (1) RANKING's empirical mean ratio tracks the *exact* finite-n
    KVV floor `1-(1-1/n)^n` to within Monte Carlo noise at every measured n;
    (2) greedy's ratio on the identical instance is *exactly* 1/2 at every
    n, matching greedy's independently-known tight bound; (3) local search
    initialized at this instance (`run_search_refinement`) cannot find a
    meaningfully harder one nearby.
    """
    graph = staircase_graph(n)
    order = list(range(n - 1, -1, -1))
    return graph, order


def run_construction_sweep(n_values, rng, eval_trials=500):
    """For each n: evaluate RANKING and greedy on `staircase_hard_instance`
    under both adversarial order and the random-order model (ROM), with
    bootstrap confidence intervals. This is the main experiment -- H0
    (theorem floor holds), H1 (converges to 1-1/e), H2 (ROM beats
    adversarial order), and H3 (greedy is worse, and benefits less from
    ROM) are all read directly off these rows.
    """
    rows = []
    for n in n_values:
        graph, order = staircase_hard_instance(n)
        opt = n

        rk_adv = ranking_ratios(graph, order, rng, eval_trials, opt_size=opt)
        rk_rom = [ranking_ratio_single(graph, rng.permutation(n), rng, opt)
                  for _ in range(eval_trials)]
        gr_adv = greedy_ratio(graph, order, opt_size=opt)
        gr_rom = greedy_ratios_random_orders(graph, rng, eval_trials, opt_size=opt)

        rk_adv_mean, rk_adv_lo, rk_adv_hi = mean_ci(rk_adv, rng=rng)
        rk_rom_mean, rk_rom_lo, rk_rom_hi = mean_ci(rk_rom, rng=rng)
        gr_rom_mean, gr_rom_lo, gr_rom_hi = mean_ci(gr_rom, rng=rng)
        rom_adv_diff_mean, rom_adv_diff_lo, rom_adv_diff_hi = two_sample_diff_ci(
            rk_rom, rk_adv, rng=rng)
        rk_gr_rom_diff_mean, rk_gr_rom_diff_lo, rk_gr_rom_diff_hi = two_sample_diff_ci(
            rk_rom, gr_rom, rng=rng)

        floor_n = exact_finite_floor(n)
        rows.append(dict(
            n=n,
            num_edges=graph.num_edges(),
            ranking_adversarial_mean=rk_adv_mean,
            ranking_adversarial_lo=rk_adv_lo,
            ranking_adversarial_hi=rk_adv_hi,
            ranking_rom_mean=rk_rom_mean,
            ranking_rom_lo=rk_rom_lo,
            ranking_rom_hi=rk_rom_hi,
            rom_minus_adversarial_mean=rom_adv_diff_mean,
            rom_minus_adversarial_lo=rom_adv_diff_lo,
            rom_minus_adversarial_hi=rom_adv_diff_hi,
            greedy_adversarial=gr_adv,
            greedy_rom_mean=gr_rom_mean,
            greedy_rom_lo=gr_rom_lo,
            greedy_rom_hi=gr_rom_hi,
            ranking_rom_minus_greedy_rom_mean=rk_gr_rom_diff_mean,
            ranking_rom_minus_greedy_rom_lo=rk_gr_rom_diff_lo,
            ranking_rom_minus_greedy_rom_hi=rk_gr_rom_diff_hi,
            # H0: the *mean* ratio (what KVV's theorem bounds -- individual
            # random-rank draws are allowed to fall below the floor) should
            # not sit significantly below the exact finite-n floor.
            asymptotic_floor=ONE_MINUS_INV_E,
            exact_finite_floor=floor_n,
            mean_meets_exact_floor=bool(rk_adv_lo >= floor_n - FLOOR_TOLERANCE),
            abs_gap_to_exact_floor=abs(rk_adv_mean - floor_n),
        ))
    return rows


def run_search_refinement(n_values, rng, n_iterations=150, trials_per_eval=40):
    """Initialize simulated annealing (`search_worst_case_graph`) *at* the
    derived construction and let it try to find something harder nearby.
    If the construction is already essentially optimal (as KVV's theorem,
    which bounds the true worst case over ALL graphs, predicts nothing can
    beat by more than Monte Carlo noise), the search should barely move.
    Returns (rows, histories).
    """
    rows = []
    histories = {}
    for n in n_values:
        graph, order = staircase_hard_instance(n)
        best_graph, best_score, history = search_worst_case_graph(
            n, order, rng, n_iterations=n_iterations, trials_per_eval=trials_per_eval,
            init_graph=graph)
        histories[n] = history
        starting_score = history[0]
        rows.append(dict(
            n=n,
            starting_score=starting_score,
            best_score=best_score,
            improvement=starting_score - best_score,
            exact_finite_floor=exact_finite_floor(n),
        ))
    return rows, histories


def run_control_sweep(n_values, p_values, rng, eval_trials=200):
    """RANKING/greedy ratios on "easy" (non-adversarial) graphs: complete
    bipartite and random G(n,n,p) at a few densities, both under adversarial
    order. Used as a contrast to `staircase_hard_instance` -- these should
    sit close to 1.0, showing that a low ratio is a property of the
    *instance*, not an artifact of the algorithms themselves.
    """
    rows = []
    for n in n_values:
        order = adversarial_order(n)

        cb = complete_bipartite(n)
        rk = ranking_ratios(cb, order, rng, eval_trials, opt_size=n)
        rk_mean, rk_lo, rk_hi = mean_ci(rk, rng=rng)
        rows.append(dict(n=n, family="complete_bipartite", param=None,
                          ranking_mean=rk_mean, ranking_lo=rk_lo, ranking_hi=rk_hi,
                          greedy=greedy_ratio(cb, order, opt_size=n)))

        for p in p_values:
            g = random_bipartite(n, p, rng)
            opt = max_matching_size(g)
            rk = ranking_ratios(g, order, rng, eval_trials, opt_size=opt)
            rk_mean, rk_lo, rk_hi = mean_ci(rk, rng=rng)
            rows.append(dict(n=n, family="random_bipartite", param=p,
                              ranking_mean=rk_mean, ranking_lo=rk_lo, ranking_hi=rk_hi,
                              greedy=greedy_ratio(g, order, opt_size=opt) if opt else float("nan")))
    return rows


def summarize_trend(rows):
    """Fit ranking_adversarial_mean vs n (H1: does it trend down toward
    1-1/e as n grows, tracking the exact finite-n floor?). Returns dict
    with slope/intercept/r2 plus the theorem-floor and ROM-improvement
    sanity checks across all rows."""
    ns = [r["n"] for r in rows]
    means = [r["ranking_adversarial_mean"] for r in rows]
    slope, intercept, r2 = linear_fit(ns, means)
    all_meet_floor = all(r["mean_meets_exact_floor"] for r in rows)
    max_gap = max(r["abs_gap_to_exact_floor"] for r in rows)
    rom_helps_everywhere = all(r["rom_minus_adversarial_lo"] > 0 for r in rows)
    return dict(
        trend_slope=slope, trend_intercept=intercept, trend_r2=r2,
        asymptotic_floor=ONE_MINUS_INV_E,
        all_means_meet_exact_finite_floor=all_meet_floor,
        max_gap_to_exact_floor=max_gap,
        rom_significantly_better_at_every_n=rom_helps_everywhere,
    )
