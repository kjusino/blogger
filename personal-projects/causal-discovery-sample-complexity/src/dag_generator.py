"""Random sparse linear-Gaussian structural equation models (SEMs).

Generates a random DAG on p nodes with a bound on the maximum node degree
(parents + children), assigns linear coefficients to each edge, and draws
samples from the resulting linear-Gaussian SEM:

    X_j = sum_{k in parents(j)} beta_{k,j} * X_k + eps_j,   eps_j ~ N(0, 1)

The topological order is fixed to 0, 1, ..., p-1 so that node j can only
have parents among {0, ..., j-1}. This is a standard construction for
benchmarking causal discovery algorithms.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearSEM:
    p: int
    max_degree: int
    parents: list  # parents[j] = sorted list of parent indices of node j
    weights: dict  # weights[(i, j)] = edge coefficient beta_{i,j}

    def skeleton(self) -> np.ndarray:
        """Symmetric p x p boolean adjacency matrix of the undirected skeleton."""
        adj = np.zeros((self.p, self.p), dtype=bool)
        for j, ps in enumerate(self.parents):
            for i in ps:
                adj[i, j] = True
                adj[j, i] = True
        return adj

    def realized_max_degree(self) -> int:
        adj = self.skeleton()
        if self.p == 0:
            return 0
        return int(adj.sum(axis=0).max())

    def analytic_covariance(self) -> np.ndarray:
        """Exact population covariance matrix of the linear-Gaussian SEM,
        computed from the structural equations rather than estimated from
        samples: Sigma = (I - B)^-1 (I - B)^-T for unit noise variances,
        where B[i, j] = beta_{i,j} is the weighted adjacency matrix.

        Because parents(j) subset {0, ..., j-1}, B is strictly upper
        triangular and (I - B) is always invertible.
        """
        B = np.zeros((self.p, self.p))
        for j, ps in enumerate(self.parents):
            for i in ps:
                B[i, j] = self.weights[(i, j)]
        # X = B^T X + eps  =>  X = (I - B^T)^-1 eps = (I - B)^-T eps
        # so Cov(X) = (I - B)^-T (I - B)^-1 = inv.T @ inv, NOT inv @ inv.T.
        inv = np.linalg.inv(np.eye(self.p) - B)
        return inv.T @ inv

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n i.i.d. samples, returns an (n, p) array."""
        X = np.zeros((n, self.p))
        for j in range(self.p):
            mean = np.zeros(n)
            for i in self.parents[j]:
                mean += self.weights[(i, j)] * X[:, i]
            X[:, j] = mean + rng.normal(0.0, 1.0, size=n)
        # Standardize columns so every variable has unit marginal variance.
        # This keeps the Fisher z partial-correlation test well-scaled across
        # nodes regardless of how much variance has accumulated downstream.
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        return X


def _sample_edge_weight(rng: np.random.Generator) -> float:
    """Draw a coefficient with magnitude bounded away from zero.

    Faithfulness violations (near-cancelling paths) are a separate
    phenomenon from finite-sample estimation error; keeping |beta| away
    from 0 avoids accidentally studying the former when the experiment is
    designed to study the latter.
    """
    magnitude = rng.uniform(0.4, 1.2)
    sign = rng.choice([-1.0, 1.0])
    return sign * magnitude


def generate_random_dag(p: int, max_degree: int, rng: np.random.Generator) -> LinearSEM:
    """Generate a random DAG with (skeleton) degree capped at max_degree.

    Construction: process nodes in topological order 1..p-1. For node j,
    candidate parents are all earlier nodes whose current skeleton degree
    is still below max_degree. From those candidates, sample up to
    max_degree parents uniformly at random (fewer if not enough eligible
    candidates remain). This yields graphs whose realized max degree is
    at most max_degree, and typically close to it for p >> max_degree.
    """
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")
    parents = [[] for _ in range(p)]
    weights = {}
    degree = np.zeros(p, dtype=int)

    for j in range(1, p):
        eligible = [i for i in range(j) if degree[i] < max_degree]
        capacity = max_degree - degree[j]
        if capacity <= 0 or not eligible:
            continue
        k = rng.integers(0, min(capacity, len(eligible)) + 1)
        if k == 0:
            continue
        chosen = rng.choice(eligible, size=k, replace=False)
        for i in sorted(int(c) for c in chosen):
            parents[j].append(i)
            weights[(i, j)] = _sample_edge_weight(rng)
            degree[i] += 1
            degree[j] += 1

    return LinearSEM(p=p, max_degree=max_degree, parents=parents, weights=weights)


def generate_faithful_dag(
    p: int,
    max_degree: int,
    rng: np.random.Generator,
    max_weight_attempts: int = 40,
    max_structure_attempts: int = 10,
    eps: float = 1e-8,
    min_margin: float = 0.12,
) -> LinearSEM:
    """Generate a random DAG whose implied population distribution is
    faithful to its graph AND whose weakest true-edge signal exceeds
    min_margin, verified by an oracle run of the PC skeleton search on the
    exact (sample-free) covariance matrix.

    Two failure modes of a naively-sampled random linear-Gaussian SEM
    would otherwise contaminate a study of *sample complexity*:

    1. Unfaithfulness: near-cancelling paths make a genuine edge's implied
       partial correlation vanish for some conditioning set the algorithm
       tests. No amount of data fixes this -- it's a population-level
       property, not an estimation error.
    2. Weak signal: even a faithful SEM can have a true edge whose implied
       partial correlation is tiny (but nonzero), demanding astronomically
       large n to detect. The "n = Theta(d^2 log p)" sample-complexity
       theory implicitly assumes a fixed lower bound on this signal
       strength (the "strong faithfulness" / beta-min condition); without
       enforcing one explicitly, graph-to-graph variation in signal
       strength would swamp the d and p scaling this experiment measures.

    This rejection-sampling procedure keeps the DAG's structure fixed and
    redraws edge weights until the oracle recovers the exact true skeleton
    with every true edge's margin (see pc_algorithm.min_true_edge_margin)
    at least min_margin; if that fails repeatedly it redraws the structure
    itself.
    """
    # Imported lazily to avoid a module-level circular import with
    # pc_algorithm (which only depends on partial_correlation).
    from .pc_algorithm import estimate_skeleton_oracle, min_true_edge_margin

    for _ in range(max_structure_attempts):
        sem = generate_random_dag(p, max_degree, rng)
        true_skeleton = sem.skeleton()
        if not sem.weights:
            return sem  # edgeless graph: trivially faithful, no margin to check
        max_cond_set = min(p - 2, max_degree + 3)
        for _ in range(max_weight_attempts):
            cov = sem.analytic_covariance()
            oracle = estimate_skeleton_oracle(cov, eps=eps, max_cond_set=max_cond_set)
            if np.array_equal(oracle.skeleton, true_skeleton):
                margin = min_true_edge_margin(true_skeleton, oracle)
                if margin >= min_margin:
                    return sem
            # Redraw weights only, keep the same parent structure.
            for key in sem.weights:
                sem.weights[key] = _sample_edge_weight(rng)
    raise RuntimeError(
        f"Could not generate a faithful DAG with margin >= {min_margin} for "
        f"p={p}, max_degree={max_degree} after {max_structure_attempts} "
        "structure attempts."
    )
