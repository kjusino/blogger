import random

import pytest

from src.tanner import build_regular_tanner_graph


def test_regularity_and_edge_count():
    rng = random.Random(0)
    g = build_regular_tanner_graph(60, 3, 6, rng)
    assert g.n == 60
    assert g.m == 30  # n*dv/dc = 60*3/6
    assert all(len(cs) == 3 for cs in g.var_to_checks)
    assert all(len(vs) == 6 for vs in g.check_to_vars)
    total_from_vars = sum(len(cs) for cs in g.var_to_checks)
    total_from_checks = sum(len(vs) for vs in g.check_to_vars)
    assert total_from_vars == total_from_checks == 60 * 3


def test_no_parallel_edges():
    rng = random.Random(1)
    g = build_regular_tanner_graph(400, 3, 6, rng)
    for v, checks in enumerate(g.var_to_checks):
        assert len(checks) == len(set(checks)), f"variable {v} has a parallel edge"
    for c, vs in enumerate(g.check_to_vars):
        assert len(vs) == len(set(vs)), f"check {c} has a parallel edge"


def test_var_check_adjacency_is_consistent():
    rng = random.Random(2)
    g = build_regular_tanner_graph(120, 3, 6, rng)
    for v, checks in enumerate(g.var_to_checks):
        for c in checks:
            assert v in g.check_to_vars[c]
    for c, vs in enumerate(g.check_to_vars):
        for v in vs:
            assert c in g.var_to_checks[v]


def test_rejects_non_integer_check_count():
    rng = random.Random(3)
    with pytest.raises(ValueError):
        # n*dv=10*3=30 is not divisible by dc=7
        build_regular_tanner_graph(10, 3, 7, rng)


def test_different_seeds_give_different_graphs():
    g1 = build_regular_tanner_graph(200, 3, 6, random.Random(10))
    g2 = build_regular_tanner_graph(200, 3, 6, random.Random(11))
    assert g1.var_to_checks != g2.var_to_checks
