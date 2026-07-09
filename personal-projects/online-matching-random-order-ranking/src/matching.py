"""Bipartite graphs, exact maximum matching (Hopcroft-Karp), and the two online
matching algorithms this project compares: deterministic GREEDY and the
randomized RANKING algorithm of Karp, Vazirani & Vazirani (STOC 1990).

Vertex sets are always `left` (offline, known in advance) and `right`
(online, revealed one at a time in "arrival order"). An online algorithm may
only use, when vertex `j` (right) arrives, the edges incident to `j` and the
matching decisions already made -- it may never look ahead or revise a past
match. `greedy_online_matching` and `ranking_online_matching` both enforce
this by construction: they scan `arrival_order` once, left to right.
"""

from collections import deque


class BipartiteGraph:
    """An undirected bipartite graph on `n_left` (offline) + `n_right`
    (online) vertices, stored as a fixed edge set. Adjacency lists are built
    lazily and cached; `with_edge_toggled` returns a *new* graph (used by the
    local-search adversarial-instance finder in `search.py`) rather than
    mutating in place, so a graph object is always safe to reuse across many
    algorithm runs.
    """

    __slots__ = ("n_left", "n_right", "edges", "_adj_left", "_adj_right")

    def __init__(self, n_left, n_right, edges):
        self.n_left = n_left
        self.n_right = n_right
        self.edges = frozenset(edges)
        for i, j in self.edges:
            if not (0 <= i < n_left) or not (0 <= j < n_right):
                raise ValueError(f"edge ({i},{j}) out of range for graph "
                                  f"with n_left={n_left}, n_right={n_right}")
        self._adj_left = None
        self._adj_right = None

    @property
    def adj_left(self):
        if self._adj_left is None:
            adj = [[] for _ in range(self.n_left)]
            for i, j in self.edges:
                adj[i].append(j)
            for row in adj:
                row.sort()
            self._adj_left = adj
        return self._adj_left

    @property
    def adj_right(self):
        if self._adj_right is None:
            adj = [[] for _ in range(self.n_right)]
            for i, j in self.edges:
                adj[j].append(i)
            for row in adj:
                row.sort()
            self._adj_right = adj
        return self._adj_right

    def with_edge_toggled(self, i, j):
        """Return a new graph with edge (i, j) added if absent, removed if
        present. Does not mutate self."""
        edges = set(self.edges)
        if (i, j) in edges:
            edges.discard((i, j))
        else:
            edges.add((i, j))
        return BipartiteGraph(self.n_left, self.n_right, edges)

    def num_edges(self):
        return len(self.edges)


def max_bipartite_matching(graph):
    """Exact maximum-cardinality bipartite matching via Hopcroft-Karp,
    O(E * sqrt(V)). Returns (match_left, match_right, size) where
    match_left[i] is the matched right-vertex (or -1) and vice versa.
    """
    n_left, n_right = graph.n_left, graph.n_right
    adj = graph.adj_left
    INF = float("inf")
    match_left = [-1] * n_left
    match_right = [-1] * n_right
    dist = [0] * n_left

    def bfs():
        queue = deque()
        for u in range(n_left):
            if match_left[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found_free = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                w = match_right[v]
                if w == -1:
                    found_free = True
                elif dist[w] == INF:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        return found_free

    def dfs(u):
        for v in adj[u]:
            w = match_right[v]
            if w == -1 or (dist[w] == dist[u] + 1 and dfs(w)):
                match_left[u] = v
                match_right[v] = u
                return True
        dist[u] = INF
        return False

    size = 0
    while bfs():
        for u in range(n_left):
            if match_left[u] == -1 and dfs(u):
                size += 1
    return match_left, match_right, size


def max_matching_size(graph):
    return max_bipartite_matching(graph)[2]


def greedy_online_matching(graph, arrival_order=None):
    """Deterministic greedy: each online vertex takes the lowest-indexed
    unmatched offline neighbor available at arrival time, or stays unmatched.
    """
    if arrival_order is None:
        arrival_order = range(graph.n_right)
    adj_right = graph.adj_right
    matched_left = [False] * graph.n_left
    match_right = [-1] * graph.n_right
    for j in arrival_order:
        for i in adj_right[j]:
            if not matched_left[i]:
                matched_left[i] = True
                match_right[j] = i
                break
    return match_right


def ranking_online_matching(graph, rank, arrival_order=None):
    """RANKING (Karp-Vazirani-Vazirani 1990): `rank` is a permutation of
    range(n_left) fixed *before* any online vertex arrives (rank[i] is
    vertex i's priority; lower = matched first). Each online vertex is
    matched to its unmatched neighbor of lowest rank, or stays unmatched.
    Setting rank = identity recovers greedy with a fixed tie-break order;
    the algorithm's power comes from drawing rank uniformly at random.
    """
    if arrival_order is None:
        arrival_order = range(graph.n_right)
    adj_right = graph.adj_right
    matched_left = [False] * graph.n_left
    match_right = [-1] * graph.n_right
    for j in arrival_order:
        best_i, best_rank = -1, None
        for i in adj_right[j]:
            if not matched_left[i] and (best_rank is None or rank[i] < best_rank):
                best_i, best_rank = i, rank[i]
        if best_i != -1:
            matched_left[best_i] = True
            match_right[j] = best_i
    return match_right


def matching_size(match_right):
    return sum(1 for x in match_right if x != -1)


def is_valid_matching(graph, match_right):
    """A matching is valid iff every matched pair is a real edge and no
    offline vertex is reused."""
    if len(match_right) != graph.n_right:
        return False
    used_left = set()
    adj_right = graph.adj_right
    for j, i in enumerate(match_right):
        if i == -1:
            continue
        if i in used_left:
            return False
        if i not in adj_right[j]:
            return False
        used_left.add(i)
    return True
