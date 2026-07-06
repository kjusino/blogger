import random

from src.decoder import peel_decode, sample_erasures
from src.tanner import TannerGraph, build_regular_tanner_graph

# Small hand-crafted graph for exact, worked-by-hand peeling checks:
#   check 0 -- {var0, var1, var2}
#   check 1 -- {var2, var3}
_SMALL_GRAPH = TannerGraph(
    n=4,
    m=2,
    dv=0,
    dc=0,
    var_to_checks=[[0], [0], [0, 1], [1]],
    check_to_vars=[[0, 1, 2], [2, 3]],
)


def test_single_erasure_always_recovers():
    ok, remaining = peel_decode(_SMALL_GRAPH, {0})
    assert ok and not remaining


def test_two_erasures_on_same_check_stuck_without_help():
    # check0 has both var0 and var1 erased, and check1 shares no erased
    # neighbor with check0, so neither check ever drops to exactly 1
    # erased neighbor.
    ok, remaining = peel_decode(_SMALL_GRAPH, {0, 1})
    assert not ok
    assert remaining == {0, 1}


def test_shared_variable_resolves_both_checks():
    # var2 is erased alone on check1 (degree 1), so it peels first; that
    # then drops check0 to a single erased neighbor (var0), which peels
    # next -- a genuine two-step cascade.
    ok, remaining = peel_decode(_SMALL_GRAPH, {0, 2})
    assert ok and not remaining


def test_disjoint_single_erasures_both_recover_independently():
    ok, remaining = peel_decode(_SMALL_GRAPH, {0, 3})
    assert ok and not remaining


def test_triple_erasure_leaves_a_stuck_residual():
    ok, remaining = peel_decode(_SMALL_GRAPH, {0, 1, 2})
    assert not ok
    assert remaining == {0, 1}


def test_no_erasures_trivially_succeeds():
    ok, remaining = peel_decode(_SMALL_GRAPH, set())
    assert ok and not remaining


def test_all_erased_fails_on_small_graph():
    ok, remaining = peel_decode(_SMALL_GRAPH, {0, 1, 2, 3})
    assert not ok
    assert remaining  # nothing can ever be recovered with zero known bits


def test_sample_erasures_respects_probability_extremes():
    rng = random.Random(0)
    assert sample_erasures(50, 0.0, rng) == set()
    assert sample_erasures(50, 1.0, rng) == set(range(50))


def test_low_erasure_rate_almost_always_decodes_on_regular_graph():
    rng = random.Random(5)
    g = build_regular_tanner_graph(500, 3, 6, rng)
    successes = 0
    trials = 200
    for _ in range(trials):
        erased = sample_erasures(g.n, 0.1, rng)
        ok, _ = peel_decode(g, erased)
        successes += ok
    assert successes / trials > 0.95


def test_high_erasure_rate_almost_never_decodes_on_regular_graph():
    rng = random.Random(6)
    g = build_regular_tanner_graph(500, 3, 6, rng)
    successes = 0
    trials = 200
    for _ in range(trials):
        erased = sample_erasures(g.n, 0.7, rng)
        ok, _ = peel_decode(g, erased)
        successes += ok
    assert successes / trials < 0.05
