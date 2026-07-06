"""Random regular Tanner-graph construction for (dv, dc)-regular LDPC codes.

Uses Gallager's configuration-model construction: each of the n variable
nodes owns dv "sockets" and each of the m = n*dv/dc check nodes owns dc
"sockets". A uniformly random perfect matching between the two socket
multisets defines the bipartite edge set. This is the standard way to
sample from the (dv, dc)-regular ensemble used in density-evolution theory.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class TannerGraph:
    """Bipartite variable/check graph for a (dv, dc)-regular LDPC code."""

    n: int
    m: int
    dv: int
    dc: int
    var_to_checks: list[list[int]]
    check_to_vars: list[list[int]]


def _try_build_once(
    n: int, m: int, dv: int, dc: int, rng: random.Random, max_stub_tries: int = 64
) -> list[set] | None:
    """One attempt at a swap-pop stub matching; returns None on failure.

    Processes variable stubs in random order. For each stub, draws random
    candidates from the remaining check-stub pool (removing each rejected
    candidate from the pool via swap-with-last so sampling stays O(1)) until
    one is found that is not already a neighbor of this variable. This
    keeps parallel-edge collisions local (a few rejected draws) instead of
    forcing a full-graph restart, which is what made the naive one-shot
    random matching fail almost always once dc is not tiny.
    """
    check_stub_pool = [c for c in range(m) for _ in range(dc)]
    var_to_checks: list[set] = [set() for _ in range(n)]
    check_to_vars: list[set] = [set() for _ in range(m)]

    var_stub_order = [v for v in range(n) for _ in range(dv)]
    rng.shuffle(var_stub_order)

    for v in var_stub_order:
        pool = check_stub_pool
        tries = min(max_stub_tries, len(pool))
        found = None
        for _ in range(tries):
            idx = rng.randrange(len(pool))
            c = pool[idx]
            if c not in var_to_checks[v]:
                pool[idx] = pool[-1]
                pool.pop()
                found = c
                break
        if found is None:
            return None
        var_to_checks[v].add(found)
        check_to_vars[found].add(v)

    return var_to_checks, check_to_vars  # type: ignore[return-value]


def build_regular_tanner_graph(
    n: int,
    dv: int,
    dc: int,
    rng: random.Random,
    max_resample_attempts: int = 200,
) -> TannerGraph:
    """Sample a random (dv, dc)-regular Tanner graph on n variable nodes.

    Requires n * dv to be divisible by dc (so the check-node count m =
    n*dv/dc is an integer, i.e. every check has exactly dc neighbors).
    Uses a randomized stub-matching construction (see `_try_build_once`)
    that avoids parallel edges (two edges between the same variable/check
    pair), which would create a length-2 cycle inconsistent with the
    locally-tree-like assumption density evolution relies on. This is
    standard practice for configuration-model LDPC construction.
    """
    if n <= 0 or dv <= 0 or dc <= 0:
        raise ValueError("n, dv, dc must be positive")
    total_edges = n * dv
    if total_edges % dc != 0:
        raise ValueError(
            f"n*dv={total_edges} must be divisible by dc={dc} "
            "(m = n*dv/dc must be an integer)"
        )
    m = total_edges // dc

    for _ in range(max_resample_attempts):
        result = _try_build_once(n, m, dv, dc, rng)
        if result is not None:
            var_to_checks, check_to_vars = result
            return TannerGraph(
                n=n,
                m=m,
                dv=dv,
                dc=dc,
                var_to_checks=[sorted(s) for s in var_to_checks],
                check_to_vars=[sorted(s) for s in check_to_vars],
            )

    raise RuntimeError(
        f"could not build a parallel-edge-free ({dv},{dc})-regular graph "
        f"on n={n} in {max_resample_attempts} attempts"
    )
