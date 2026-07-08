import random

import numpy as np
import pytest

from grok.groups import CyclicGroup, DihedralGroup, QuaternionGroup, group_task_pairs

GROUPS = [
    ("C7", CyclicGroup(7)),
    ("C13", CyclicGroup(13)),
    ("D9", DihedralGroup(9)),
    ("D15", DihedralGroup(15)),
    ("Q8", QuaternionGroup()),
]


@pytest.mark.parametrize("name,group", GROUPS)
def test_identity_and_inverse(name, group):
    for a in range(group.order):
        assert group.mul(a, group.identity) == a
        assert group.mul(group.identity, a) == a
        assert group.mul(a, group.inv(a)) == group.identity
        assert group.mul(group.inv(a), a) == group.identity


@pytest.mark.parametrize("name,group", GROUPS)
def test_associativity(name, group):
    rng = random.Random(42)
    order = group.order
    triples = [(rng.randrange(order), rng.randrange(order), rng.randrange(order))
               for _ in range(500)]
    for a, b, c in triples:
        assert group.mul(group.mul(a, b), c) == group.mul(a, group.mul(b, c))


@pytest.mark.parametrize("name,group", GROUPS)
def test_closure(name, group):
    order = group.order
    for a in range(order):
        for b in range(order):
            result = group.mul(a, b)
            assert 0 <= result < order


def test_cyclic_is_abelian():
    g = CyclicGroup(11)
    for a in range(11):
        for b in range(11):
            assert g.mul(a, b) == g.mul(b, a)


def test_dihedral_is_not_abelian():
    g = DihedralGroup(9)
    found_noncommuting = any(
        g.mul(a, b) != g.mul(b, a) for a in range(g.order) for b in range(g.order))
    assert found_noncommuting


def test_quaternion_is_not_abelian():
    g = QuaternionGroup()
    found_noncommuting = any(
        g.mul(a, b) != g.mul(b, a) for a in range(g.order) for b in range(g.order))
    assert found_noncommuting


def test_quaternion_minus_one_is_central_order_two():
    g = QuaternionGroup()
    minus_one = g._labels.index("-1")
    assert g.mul(minus_one, minus_one) == g.identity
    for a in range(g.order):
        assert g.mul(minus_one, a) == g.mul(a, minus_one)


@pytest.mark.parametrize("name,group", GROUPS)
def test_irreps_are_homomorphisms(name, group):
    for label, dim, rho in group.irreps():
        for a in range(group.order):
            for b in range(group.order):
                lhs = rho(group.mul(a, b))
                rhs = rho(a) @ rho(b)
                assert np.allclose(lhs, rhs, atol=1e-8), f"{name}/{label} fails at ({a},{b})"


@pytest.mark.parametrize("name,group", GROUPS)
def test_irreps_are_unitary(name, group):
    """Every representation used here should be unitary (orthogonal, since
    real or built from rotation/reflection/Pauli-like blocks) -- this is
    assumed by the Peter-Weyl orthogonality argument in spectral.py."""
    for label, dim, rho in group.irreps():
        for g in group.elements:
            m = rho(g)
            assert np.allclose(m @ m.conj().T, np.eye(dim), atol=1e-8), \
                f"{name}/{label} not unitary at g={g}"


@pytest.mark.parametrize("name,group", GROUPS)
def test_identity_maps_to_identity_matrix(name, group):
    for label, dim, rho in group.irreps():
        assert np.allclose(rho(group.identity), np.eye(dim), atol=1e-8)


@pytest.mark.parametrize("name,group", GROUPS)
def test_group_task_pairs_shape_and_labels(name, group):
    a_idx, b_idx, labels = group_task_pairs(group)
    n = group.order * group.order
    assert a_idx.shape == (n,)
    assert b_idx.shape == (n,)
    assert labels.shape == (n,)
    assert labels.min() >= 0 and labels.max() < group.order
    # spot-check a handful of entries against group.mul directly
    rng = np.random.default_rng(0)
    for idx in rng.integers(0, n, size=20):
        assert labels[idx] == group.mul(int(a_idx[idx]), int(b_idx[idx]))
