"""Spectral / representation-theoretic alignment of a trained embedding table.

The idea (generalizing the well-known "grokked networks do Fourier analysis"
finding from cyclic groups to arbitrary finite groups): the matrix-coefficient
functions of a group's irreducible representations form a complete orthonormal
basis for the space of real-valued functions on the group (Peter-Weyl for
finite groups). If a trained embedding table W (one row per group element)
concentrates its variance in a small number of these "harmonic blocks" rather
than spreading uniformly across all of them, that's exactly the structured,
non-lookup-table circuit grokking work reports for cyclic groups (concentration
in a handful of Fourier frequencies) -- just measured in a basis that also
makes sense for non-abelian groups (rotations/reflections for dihedral,
quaternionic matrix entries for Q8).

`basis_blocks` builds that basis directly from `group.irreps()` and is
agnostic to which group it is: it works from matrix-coefficient candidate
vectors, projects out whatever a previous irrep already covered (so
irreducible-but-real-span-duplicating irreps, e.g. complex-conjugate pairs
for an abelian group, don't get double-counted), and keeps only genuinely new
directions found by SVD. `alignment_score` then measures how concentrated a
trained embedding's variance is across those blocks, and calibrates that
number against a null distribution obtained by permuting which group element
each embedding row belongs to -- a shuffle that keeps every embedding vector
and the table's overall statistics identical while destroying any group
structure.
"""
import numpy as np


def basis_blocks(group, tol=1e-6):
    """Complete orthonormal basis of R^|G| grouped into per-irrep blocks.

    Returns a list of (label, basis) where `basis` is an (|G| x k) matrix
    with orthonormal columns. Blocks are mutually orthogonal by construction
    and their basis vectors, concatenated, span all of R^|G|.
    """
    order = group.order
    covered = np.zeros((order, 0))
    blocks = []
    for label, dim, rho in group.irreps():
        candidates = []
        for i in range(dim):
            for j in range(dim):
                vec = np.array([rho(g)[i, j] for g in group.elements])
                candidates.append(vec.real.copy())
                if np.max(np.abs(vec.imag)) > tol:
                    candidates.append(vec.imag.copy())
        M = np.stack(candidates, axis=1)
        if covered.shape[1] > 0:
            M = M - covered @ (covered.T @ M)
        u, s, _ = np.linalg.svd(M, full_matrices=False)
        rank = int(np.sum(s > tol))
        if rank == 0:
            continue
        new_basis = u[:, :rank]
        blocks.append((label, new_basis))
        covered = np.concatenate([covered, new_basis], axis=1)
    return blocks


def assert_basis_is_complete(group, blocks, atol=1e-6):
    """Sanity invariant: the blocks' basis vectors form an orthonormal basis
    of all of R^|G| (total dim == |G|, Gram matrix == identity)."""
    order = group.order
    full = np.concatenate([b for _, b in blocks], axis=1)
    if full.shape[1] != order:
        return False
    gram = full.T @ full
    return np.allclose(gram, np.eye(order), atol=atol)


def block_energies(embeddings, blocks):
    """Total variance of `embeddings` (|G| x d) explained by each block,
    summed across all d embedding dimensions."""
    energies = []
    for _, basis in blocks:
        projected = basis.T @ embeddings  # (block_dim, d)
        energies.append(float(np.sum(projected ** 2)))
    return np.array(energies)


def concentration_score(energies):
    """Normalized participation-ratio concentration in [0, 1].

    1.0 = all variance in a single block (maximally structured/sparse).
    0.0 = variance spread perfectly uniformly across every block (as
    unstructured noise would look, in expectation).
    """
    total = energies.sum()
    num_blocks = len(energies)
    if total <= 0 or num_blocks <= 1:
        return 0.0
    p = energies / total
    participation_ratio = 1.0 / np.sum(p ** 2)
    return float((num_blocks - participation_ratio) / (num_blocks - 1))


def alignment_score(embeddings, blocks, n_shuffles=200, rng=None):
    """Concentration score of `embeddings` against a row-shuffle null.

    The null permutes which group element each embedding row is attached to,
    which preserves the exact multiset of embedding vectors (and hence the
    raw concentration score's sensitivity to embedding norm/scale) while
    destroying any alignment with the group's harmonic structure.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    true_score = concentration_score(block_energies(embeddings, blocks))
    order = embeddings.shape[0]
    null_scores = np.empty(n_shuffles)
    for i in range(n_shuffles):
        perm = rng.permutation(order)
        null_scores[i] = concentration_score(block_energies(embeddings[perm], blocks))
    null_mean = float(null_scores.mean())
    null_std = float(null_scores.std())
    z = (true_score - null_mean) / null_std if null_std > 1e-12 else float("inf")
    percentile = float(np.mean(null_scores < true_score))
    return {
        "score": true_score,
        "null_mean": null_mean,
        "null_std": null_std,
        "z": z,
        "percentile": percentile,
    }
