"""Directed multigraphs for single-commodity nonatomic routing games, plus a
random series-parallel network generator and three textbook instances used
as solver-correctness anchors (Pigou, Braess, three-link).

A series-parallel (SP) graph between two terminals is built recursively:
  - base case: a single edge between the two terminals.
  - series composition: chain two SP graphs end to end (sink of the first
    becomes source of the second).
  - parallel composition: overlay two SP graphs on the same terminals.
This is exactly the class of graphs Wardrop/Roughgarden-style routing
examples are usually drawn from, and it guarantees (by construction) a
connected DAG with every edge on some source-sink path -- no dangling
edges, no disconnected components, no cycles to worry about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.cost_functions import Polynomial, random_polynomial


@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    cost: Polynomial


@dataclass
class Network:
    n_nodes: int
    edges: list
    source: int
    sink: int
    demand: float = 1.0

    def __post_init__(self) -> None:
        if not (0 <= self.source < self.n_nodes and 0 <= self.sink < self.n_nodes):
            raise ValueError("source/sink out of range")
        if self.source == self.sink:
            raise ValueError("source and sink must differ")
        for e in self.edges:
            if not (0 <= e.u < self.n_nodes and 0 <= e.v < self.n_nodes):
                raise ValueError("edge endpoint out of range")

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def incidence_matrix(self) -> np.ndarray:
        """A[v, e] = +1 if edge e enters node v, -1 if it leaves node v."""
        A = np.zeros((self.n_nodes, self.n_edges))
        for j, e in enumerate(self.edges):
            A[e.v, j] += 1.0
            A[e.u, j] -= 1.0
        return A

    def conservation_rhs(self) -> np.ndarray:
        """b such that A @ f == b enforces flow conservation with `demand`
        units flowing from source to sink."""
        b = np.zeros(self.n_nodes)
        b[self.source] = -self.demand
        b[self.sink] = self.demand
        return b

    def reachable_and_coreachable(self) -> tuple:
        """Nodes reachable from source, and nodes that can reach sink -- used
        to sanity-check the random generator produces no dangling edges."""
        fwd = {self.source}
        changed = True
        while changed:
            changed = False
            for e in self.edges:
                if e.u in fwd and e.v not in fwd:
                    fwd.add(e.v)
                    changed = True
        bwd = {self.sink}
        changed = True
        while changed:
            changed = False
            for e in self.edges:
                if e.v in bwd and e.u not in bwd:
                    bwd.add(e.u)
                    changed = True
        return fwd, bwd


def _series(net_a: Network, net_b: Network) -> Network:
    """Chain net_b after net_a: net_a.sink is identified with net_b.source."""
    identify = {net_b.source: net_a.sink}
    mapping, n_nodes = _remap_with_offset(net_a.n_nodes, net_b, identify)
    edges = list(net_a.edges) + [Edge(mapping[e.u], mapping[e.v], e.cost) for e in net_b.edges]
    return Network(n_nodes, edges, source=net_a.source, sink=mapping[net_b.sink])


def _parallel(net_a: Network, net_b: Network) -> Network:
    """Overlay net_b on net_a: shared source and sink, other nodes fresh."""
    identify = {net_b.source: net_a.source, net_b.sink: net_a.sink}
    mapping, n_nodes = _remap_with_offset(net_a.n_nodes, net_b, identify)
    edges = list(net_a.edges) + [Edge(mapping[e.u], mapping[e.v], e.cost) for e in net_b.edges]
    return Network(n_nodes, edges, source=net_a.source, sink=net_a.sink)


def _remap_with_offset(offset: int, net_b: Network, identify: dict) -> tuple:
    """Map net_b's nodes: those in `identify` go to the given existing ids;
    every other node gets a fresh id starting at `offset` (the size of the
    graph net_b is being spliced into), so ids never collide."""
    mapping = {}
    next_id = offset
    for node in range(net_b.n_nodes):
        if node in identify:
            mapping[node] = identify[node]
        else:
            mapping[node] = next_id
            next_id += 1
    return mapping, next_id


def random_series_parallel(rng: np.random.Generator, degree: int, max_edges: int = 16,
                            split_prob: float = 0.75, coeff_scale: float = 1.0) -> Network:
    """A random series-parallel network with edge latencies of degree exactly
    `degree` (nonnegative coefficients), recursively split into series or
    parallel components until a per-call edge budget is exhausted."""

    def build(budget: int) -> Network:
        if budget <= 1 or rng.random() > split_prob:
            cost = random_polynomial(degree, rng, scale=coeff_scale)
            return Network(2, [Edge(0, 1, cost)], source=0, sink=1)
        left_budget = rng.integers(1, budget)
        right_budget = budget - left_budget
        left = build(left_budget)
        right = build(right_budget)
        return _series(left, right) if rng.random() < 0.5 else _parallel(left, right)

    net = build(max_edges)
    fwd, bwd = net.reachable_and_coreachable()
    assert net.sink in fwd and net.source in bwd, "generator produced a disconnected network"
    return net


# --- Textbook instances used to validate the equilibrium/optimum solver ---

def pigou_network() -> Network:
    """The canonical example achieving the p=1 bound of 4/3 exactly: edge 1
    l(x)=x, edge 2 l(x)=1, unit demand."""
    edges = [Edge(0, 1, Polynomial((0.0, 1.0))), Edge(0, 1, Polynomial((1.0,)))]
    return Network(2, edges, source=0, sink=1, demand=1.0)


def two_link_pigou_style(degree: int, b: float) -> Network:
    """Edge 1: l(x) = x^degree. Edge 2: l(x) = b (constant). Unit demand.
    Used to reproduce theory.two_link_* via the generic NLP solver."""
    top = tuple([0.0] * degree + [1.0])
    edges = [Edge(0, 1, Polynomial(top)), Edge(0, 1, Polynomial((b,)))]
    return Network(2, edges, source=0, sink=1, demand=1.0)


def braess_network(with_shortcut: bool) -> Network:
    """Braess's paradox: nodes s=0, A=1, B=2, t=3.
    s->A: l(x)=x   A->t: l(x)=1
    s->B: l(x)=1   B->t: l(x)=x
    optional shortcut A->B: l(x)=0
    """
    edges = [
        Edge(0, 1, Polynomial((0.0, 1.0))),   # s -> A, l(x) = x
        Edge(1, 3, Polynomial((1.0,))),       # A -> t, l(x) = 1
        Edge(0, 2, Polynomial((1.0,))),       # s -> B, l(x) = 1
        Edge(2, 3, Polynomial((0.0, 1.0))),   # B -> t, l(x) = x
    ]
    if with_shortcut:
        edges.append(Edge(1, 2, Polynomial((0.0,))))  # A -> B, l(x) = 0
    return Network(4, edges, source=0, sink=3, demand=1.0)


def three_parallel_links() -> Network:
    """Three parallel edges with distinct affine costs, for a solver
    correctness check beyond the two-link case."""
    edges = [
        Edge(0, 1, Polynomial((0.0, 1.0))),   # l(x) = x
        Edge(0, 1, Polynomial((0.5, 0.5))),   # l(x) = 0.5 + 0.5x
        Edge(0, 1, Polynomial((1.0, 0.2))),   # l(x) = 1 + 0.2x
    ]
    return Network(2, edges, source=0, sink=1, demand=1.0)
