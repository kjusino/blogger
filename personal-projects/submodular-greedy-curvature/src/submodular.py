"""Weighted-coverage monotone submodular functions with controllable curvature.

f(S) = sum of weights of "features" covered by the union of elements in S,
where each element deterministically covers a fixed random subset of a
shared feature universe. This is the textbook (weighted) set-cover / max-
coverage objective, a canonical monotone submodular function.
"""

import numpy as np


class WeightedCoverageFunction:
    """A monotone submodular set function over ground set {0, ..., n-1}."""

    def __init__(self, n, m, p, weight_low, weight_high, rng):
        if not (0 < p <= 1):
            raise ValueError("p must be in (0, 1]")
        if n < 1 or m < 1:
            raise ValueError("n and m must be positive")

        self.n = n
        self.m = m
        self.p = p

        covers = rng.random((n, m)) < p
        # Guarantee every element covers at least one feature, otherwise its
        # own marginal value (and hence the curvature denominator) is zero.
        empty_rows = np.where(~covers.any(axis=1))[0]
        for i in empty_rows:
            covers[i, rng.integers(0, m)] = True
        self.covers = covers  # shape (n, m), bool

        self.feature_weights = rng.uniform(weight_low, weight_high, size=m)

    def value(self, S):
        """f(S): total weight of features covered by the union of S."""
        if not S:
            return 0.0
        mask = np.zeros(self.m, dtype=bool)
        for i in S:
            mask |= self.covers[i]
        return float(self.feature_weights[mask].sum())

    def marginal_gain(self, j, S):
        """f(j | S) = f(S union {j}) - f(S)."""
        if j in S:
            return 0.0
        if not S:
            already = np.zeros(self.m, dtype=bool)
        else:
            already = np.zeros(self.m, dtype=bool)
            for i in S:
                already |= self.covers[i]
        new_features = self.covers[j] & ~already
        return float(self.feature_weights[new_features].sum())

    def curvature(self):
        """Total curvature c = 1 - min_j [ f(j | V\\{j}) / f(j | {}) ].

        Per Conforti & Cornuejols (1984). Elements with zero standalone
        value (f(j|{}) == 0) cannot occur here by construction (every
        element covers >= 1 feature with strictly positive weight), so the
        ratio is always well-defined.
        """
        full = set(range(self.n))
        ratios = []
        for j in range(self.n):
            f_alone = self.marginal_gain(j, set())
            f_last = self.marginal_gain(j, full - {j})
            ratios.append(f_last / f_alone)
        return 1.0 - min(ratios)

    @classmethod
    def from_arrays(cls, covers, feature_weights):
        """Build an instance directly from a (n, m) coverage matrix and an
        (m,) weight vector, bypassing the random constructor."""
        obj = cls.__new__(cls)
        obj.n, obj.m = covers.shape
        obj.p = None
        obj.covers = covers
        obj.feature_weights = feature_weights
        return obj


def make_grouped_redundancy_instance(n, n_groups, group_membership_prob,
                                      redundancy_mult, weight_low, weight_high, rng):
    """Build a coverage instance with a controllable *redundancy intensity*.

    Each element i has an exclusive private feature worth w_i ~
    Uniform(weight_low, weight_high) -- its guaranteed private value.
    Additionally, `n_groups` "overlap groups" are created; group g is
    assigned a random subset of elements (each included independently with
    probability `group_membership_prob`, with at least 2 members forced so
    every group creates genuine redundancy) who all cover group g's single
    feature, worth redundancy_mult * Uniform(weight_low, weight_high).

    Unlike a single block shared by *every* element (which only adds a
    constant offset and never changes which subset is optimal), partial,
    non-uniform group membership means different elements compete for
    overlapping value in different, non-trivial combinations -- so which k
    elements are jointly optimal genuinely depends on the group structure,
    not just each element's standalone value. `redundancy_mult` dials the
    resulting total curvature from exactly 0 (mult=0, purely modular, greedy
    is provably exact) up towards 1 (large mult, overlap value dominates).
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if redundancy_mult < 0:
        raise ValueError("redundancy_mult must be non-negative")

    private_weights = rng.uniform(weight_low, weight_high, size=n)

    m = n + n_groups
    covers = np.zeros((n, m), dtype=bool)
    for i in range(n):
        covers[i, i] = True

    feature_weights = np.zeros(m)
    feature_weights[:n] = private_weights

    if n_groups > 0 and redundancy_mult > 0:
        memberships = rng.random((n_groups, n)) < group_membership_prob
        for g in range(n_groups):
            if memberships[g].sum() < 2:
                idx = rng.choice(n, size=2, replace=False)
                memberships[g, :] = False
                memberships[g, idx] = True
        group_weights = rng.uniform(weight_low, weight_high, size=n_groups) * redundancy_mult
        for g in range(n_groups):
            covers[memberships[g], n + g] = True
        feature_weights[n:] = group_weights

    return WeightedCoverageFunction.from_arrays(covers, feature_weights)
