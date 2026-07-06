"""Peeling (message-passing) decoder for the binary erasure channel (BEC).

Over the BEC, whether iterative decoding recovers a transmitted codeword
depends only on the erasure pattern and the Tanner graph topology, never on
the actual bit values (Richardson & Urbanke, "Modern Coding Theory", ch. 3).
So the decoder below tracks only which variable nodes are still erased.

The algorithm is exactly belief propagation specialized to the BEC: a check
node with exactly one erased neighbor can resolve that neighbor (its value
is the XOR of the other, known, neighbors), so it is pushed onto a queue and
"peeled" off. This repeats until no check has exactly one erased neighbor.
"""

from __future__ import annotations

from collections import deque

from .tanner import TannerGraph


def peel_decode(graph: TannerGraph, erased: set[int]) -> tuple[bool, set[int]]:
    """Run the BEC peeling decoder.

    Args:
        graph: the Tanner graph.
        erased: set of variable-node indices erased by the channel.

    Returns:
        (success, remaining_erased) where success is True iff every erased
        bit was recovered, and remaining_erased is whatever is left over
        (empty on success).
    """
    erased = set(erased)
    check_erased_count = [0] * graph.m
    for c in range(graph.m):
        check_erased_count[c] = sum(1 for v in graph.check_to_vars[c] if v in erased)

    queue = deque(c for c in range(graph.m) if check_erased_count[c] == 1)
    queued = [check_erased_count[c] == 1 for c in range(graph.m)]

    while queue:
        c = queue.popleft()
        queued[c] = False
        if check_erased_count[c] != 1:
            continue
        erased_neighbor = next(v for v in graph.check_to_vars[c] if v in erased)
        erased.discard(erased_neighbor)
        for c2 in graph.var_to_checks[erased_neighbor]:
            check_erased_count[c2] -= 1
            if check_erased_count[c2] == 1 and not queued[c2]:
                queue.append(c2)
                queued[c2] = True

    return (len(erased) == 0, erased)


def sample_erasures(n: int, epsilon: float, rng) -> set[int]:
    """Sample an i.i.d. BEC erasure pattern: each bit erased w.p. epsilon."""
    return {v for v in range(n) if rng.random() < epsilon}
