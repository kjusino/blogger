"""Balanced-allocation ("balls into bins") simulators.

Three placement rules, increasing in how much randomness they get to see
before committing a ball to a bin:

- ``simulate_one_choice``: each ball lands in one uniform random bin.
  Classical result: max load - mean = Theta(log n / log log n) whp.
- ``simulate_classical_two_choice``: each ball samples two uniform random
  *distinct* bins (no graph constraint) and takes the less-loaded one.
  Azar-Broder-Karlin-Upfal (1999): max load - mean = log log n / log 2 + O(1).
- ``simulate_graphical_two_choice``: the "power of two choices" restricted
  to a graph G (Peres-Talwar-Wieder 2015): each ball samples a uniform
  random *edge* of G and takes the less-loaded endpoint. On good expanders
  this matches the classical two-choice bound; on poor expanders (paths,
  cycles) it degrades toward the one-choice bound.

All three place ``n_balls`` balls into ``n_bins`` bins and return the final
integer load vector.
"""

import numpy as np


def simulate_one_choice(n_bins: int, n_balls: int, rng: np.random.Generator) -> np.ndarray:
    loads = np.zeros(n_bins, dtype=np.int64)
    choices = rng.integers(0, n_bins, size=n_balls)
    np.add.at(loads, choices, 1)
    return loads


def simulate_classical_two_choice(n_bins: int, n_balls: int, rng: np.random.Generator) -> np.ndarray:
    if n_bins < 2:
        raise ValueError("classical two-choice requires at least 2 bins")
    loads = np.zeros(n_bins, dtype=np.int64)
    # sample two distinct bins per ball, up front, vectorized
    a = rng.integers(0, n_bins, size=n_balls)
    offset = rng.integers(1, n_bins, size=n_balls)
    b = (a + offset) % n_bins  # guaranteed != a
    for i in range(n_balls):
        u, v = a[i], b[i]
        if loads[u] <= loads[v]:
            loads[u] += 1
        else:
            loads[v] += 1
    return loads


def simulate_graphical_two_choice(
    edges: np.ndarray, n_bins: int, n_balls: int, rng: np.random.Generator
) -> np.ndarray:
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must be an (E, 2) array")
    n_edges = edges.shape[0]
    if n_edges == 0:
        raise ValueError("graph has no edges")
    loads = np.zeros(n_bins, dtype=np.int64)
    edge_choices = rng.integers(0, n_edges, size=n_balls)
    # pull the endpoint columns out once so the hot loop is pure python
    # ints/list indexing rather than repeated numpy fancy-indexing calls
    us = edges[edge_choices, 0].tolist()
    vs = edges[edge_choices, 1].tolist()
    loads_list = loads.tolist()
    for u, v in zip(us, vs):
        if loads_list[u] <= loads_list[v]:
            loads_list[u] += 1
        else:
            loads_list[v] += 1
    return np.array(loads_list, dtype=np.int64)


def max_load_gap(loads: np.ndarray, n_balls: int, n_bins: int) -> int:
    """max load minus the ideal average load (ceil(n_balls / n_bins))."""
    mean = -(-n_balls // n_bins)  # ceil division
    return int(loads.max() - mean)
