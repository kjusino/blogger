import numpy as np
import pytest

from grok.groups import CyclicGroup, DihedralGroup, QuaternionGroup
from grok.spectral import (
    basis_blocks, assert_basis_is_complete, block_energies,
    concentration_score, alignment_score,
)

GROUPS = [
    ("C11", CyclicGroup(11)),
    ("D9", DihedralGroup(9)),
    ("Q8", QuaternionGroup()),
]


@pytest.mark.parametrize("name,group", GROUPS)
def test_basis_blocks_are_complete_orthonormal_basis(name, group):
    blocks = basis_blocks(group)
    assert assert_basis_is_complete(group, blocks)
    total_dim = sum(b.shape[1] for _, b in blocks)
    assert total_dim == group.order


@pytest.mark.parametrize("name,group", GROUPS)
def test_block_energies_sum_to_total_variance(name, group):
    rng = np.random.default_rng(0)
    blocks = basis_blocks(group)
    embeddings = rng.normal(size=(group.order, 6))
    energies = block_energies(embeddings, blocks)
    assert np.isclose(energies.sum(), np.sum(embeddings ** 2), atol=1e-6)


def test_concentration_score_extremes():
    # all variance in one block -> maximal concentration
    assert concentration_score(np.array([10.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)
    # perfectly uniform spread -> zero concentration
    assert concentration_score(np.array([5.0, 5.0, 5.0, 5.0])) == pytest.approx(0.0)
    # single block: degenerate, defined as 0
    assert concentration_score(np.array([7.0])) == 0.0


def test_concentration_score_monotonic_in_skew():
    # more mass on one block should never decrease the concentration score
    flat = concentration_score(np.array([1.0, 1.0, 1.0, 1.0]))
    skewed = concentration_score(np.array([4.0, 1.0, 1.0, 1.0]))
    very_skewed = concentration_score(np.array([10.0, 1.0, 1.0, 1.0]))
    assert flat < skewed < very_skewed


def test_alignment_score_detects_structured_embedding():
    rng = np.random.default_rng(3)
    group = CyclicGroup(23)
    blocks = basis_blocks(group)
    _, basis = blocks[5]  # some non-trivial frequency block
    structured = basis @ rng.normal(size=(basis.shape[1], 4))
    result = alignment_score(structured, blocks, n_shuffles=150, rng=rng)
    assert result["score"] == pytest.approx(1.0, abs=1e-6)
    assert result["z"] > 5.0


def test_alignment_score_does_not_flag_random_noise():
    rng = np.random.default_rng(4)
    group = CyclicGroup(23)
    blocks = basis_blocks(group)
    noise = rng.normal(size=(group.order, 8))
    result = alignment_score(noise, blocks, n_shuffles=150, rng=rng)
    assert abs(result["z"]) < 3.0  # not "significant" structure


@pytest.mark.parametrize("name,group", GROUPS)
def test_alignment_score_reproducible_with_same_rng_seed(name, group):
    rng_state = np.random.default_rng(0)
    blocks = basis_blocks(group)
    embeddings = np.random.default_rng(1).normal(size=(group.order, 4))
    r1 = alignment_score(embeddings, blocks, n_shuffles=50, rng=np.random.default_rng(7))
    r2 = alignment_score(embeddings, blocks, n_shuffles=50, rng=np.random.default_rng(7))
    assert r1 == r2
