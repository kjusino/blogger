"""Test subspaces used across the experiments. All subspaces are represented by an
n x d matrix Q with orthonormal columns (Q^T Q = I_d), so that the eps-subspace-
embedding condition reduces to a statement about singular values of S^T Q -- see
theory.subspace_distortion.
"""

import numpy as np


def incoherent_basis(n, d, rng):
    """Random d-dim subspace of R^n via QR of a Gaussian matrix. Leverage scores
    (row norms^2 of Q) concentrate near d/n for every row -- low, ~uniform coherence.
    """
    G = rng.standard_normal((n, d))
    Q, _ = np.linalg.qr(G)
    return Q


def coherent_basis(n, d, rng):
    """Maximally coherent d-dim subspace: span of the first d standard basis vectors.
    The first d rows each have leverage score 1 (the theoretical maximum); the
    remaining n - d rows have leverage 0. Average leverage is still d/n, but it is
    concentrated entirely on a d-row 'spike' instead of spread uniformly -- the
    regime where naive uniform row sampling is known to fail.
    """
    Q = np.zeros((n, d))
    Q[:d, :] = np.eye(d)
    return Q


def leverage_scores(Q):
    return np.sum(Q ** 2, axis=1)


def random_least_squares_system(Q, rng, noise=0.1):
    """Build a full-rank design matrix A = Q @ R sharing Q's column space (R is a
    random orthogonal d x d matrix, so A stays well-conditioned) and a right-hand
    side b = A @ x_true + noise, for the downstream sketch-and-solve experiment.
    """
    n, d = Q.shape
    Ru, _, Rvt = np.linalg.svd(rng.standard_normal((d, d)))
    R = Ru @ Rvt
    A = Q @ R
    x_true = rng.standard_normal(d)
    b = A @ x_true + noise * rng.standard_normal(n)
    return A, b, x_true
