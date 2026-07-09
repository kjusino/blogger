import numpy as np
import pytest

from toriccode.stabilizer import ToricLattice


@pytest.mark.parametrize("L", [3, 4, 5, 6])
def test_stabilizer_rank_is_L_squared_minus_1(L):
    lat = ToricLattice(L)
    assert lat.stabilizer_rank == L * L - 1


@pytest.mark.parametrize("L", [3, 4, 5, 6])
def test_no_error_has_zero_syndrome(L):
    lat = ToricLattice(L)
    h = np.zeros((L, L), dtype=np.uint8)
    v = np.zeros((L, L), dtype=np.uint8)
    assert not lat.syndrome(h, v).any()
    assert lat.is_trivial(h, v)


@pytest.mark.parametrize("L", [3, 5, 7])
def test_single_qubit_error_flips_exactly_two_plaquettes(L):
    lat = ToricLattice(L)
    for r in range(L):
        for c in range(L):
            h = np.zeros((L, L), dtype=np.uint8)
            h[r, c] = 1
            v = np.zeros((L, L), dtype=np.uint8)
            s = lat.syndrome(h, v)
            assert s.sum() == 2
            # H[r,c] borders plaquettes (r-1,c) and (r,c)
            assert s[r, c] == 1
            assert s[(r - 1) % L, c] == 1

            h = np.zeros((L, L), dtype=np.uint8)
            v = np.zeros((L, L), dtype=np.uint8)
            v[r, c] = 1
            s = lat.syndrome(h, v)
            assert s.sum() == 2
            # V[r,c] borders plaquettes (r,c-1) and (r,c)
            assert s[r, c] == 1
            assert s[r, (c - 1) % L] == 1


@pytest.mark.parametrize("L", [3, 4, 5])
def test_star_operators_are_trivial_and_undetectable(L, rng=np.random.default_rng(0)):
    """Every single vertex star operator (and random XORs of several of
    them) must have zero syndrome and be classified as trivial."""
    lat = ToricLattice(L)

    def star(r, c):
        h = np.zeros((L, L), dtype=np.uint8)
        v = np.zeros((L, L), dtype=np.uint8)
        h[r % L, c % L] ^= 1
        h[r % L, (c - 1) % L] ^= 1
        v[r % L, c % L] ^= 1
        v[(r - 1) % L, c % L] ^= 1
        return h, v

    for _ in range(20):
        h = np.zeros((L, L), dtype=np.uint8)
        v = np.zeros((L, L), dtype=np.uint8)
        n_terms = rng.integers(1, 6)
        for _ in range(n_terms):
            r, c = rng.integers(0, L, size=2)
            sh, sv = star(int(r), int(c))
            h ^= sh
            v ^= sv
        assert not lat.syndrome(h, v).any()
        assert lat.is_trivial(h, v)


@pytest.mark.parametrize("L", [3, 4, 5, 6, 7])
def test_noncontractible_loop_is_undetectable_but_nontrivial(L):
    """A straight dual-lattice loop wrapping once around the torus (a full
    row of V edges) has zero syndrome but is NOT a stabilizer element -- it
    is exactly a logical X operator, regardless of the parity of L."""
    lat = ToricLattice(L)
    h = np.zeros((L, L), dtype=np.uint8)
    v = np.zeros((L, L), dtype=np.uint8)
    v[0, :] = 1  # full row of V edges: a horizontal-winding dual loop
    assert not lat.syndrome(h, v).any()
    assert not lat.is_trivial(h, v)

    h2 = np.zeros((L, L), dtype=np.uint8)
    v2 = np.zeros((L, L), dtype=np.uint8)
    h2[:, 0] = 1  # full column of H edges: a vertical-winding dual loop
    assert not lat.syndrome(h2, v2).any()
    assert not lat.is_trivial(h2, v2)

    # And the two loops together should also be nontrivial (independent
    # logical operators, their sum is a third nontrivial class).
    h3 = h ^ h2
    v3 = v ^ v2
    assert not lat.syndrome(h3, v3).any()
    assert not lat.is_trivial(h3, v3)


@pytest.mark.parametrize("L", [4, 5])
def test_contractible_loop_around_single_plaquette_is_trivial(L):
    """The boundary of a single plaquette is itself (trivially) a product of
    star operators is NOT generally true; but tracing the edges of a small
    contractible rectangle built from an even number of adjacent single-qubit
    toggles that XOR to a vertex star must reduce to trivial. This test
    instead checks a concrete even-length open path is trivial iff it closes
    up into a contractible loop made of two stars."""
    lat = ToricLattice(L)
    h = np.zeros((L, L), dtype=np.uint8)
    v = np.zeros((L, L), dtype=np.uint8)
    # Two adjacent vertex stars sharing an edge: their XOR is a small
    # contractible loop (a "plaquette-like" 6-edge loop), must be trivial.
    for (r, c) in [(1, 1), (1, 2)]:
        h[r, c] ^= 1
        h[r, (c - 1) % L] ^= 1
        v[r, c] ^= 1
        v[(r - 1) % L, c] ^= 1
    assert not lat.syndrome(h, v).any()
    assert lat.is_trivial(h, v)


@pytest.mark.parametrize("L", [5, 8])
def test_dual_path_toggles_reproduce_correct_syndrome(L, rng=np.random.default_rng(1)):
    """Applying the toggles of a dual-lattice path between two random
    plaquettes must create a chain whose syndrome is defects exactly at
    those two plaquettes (or zero syndrome, if the two points coincide)."""
    lat = ToricLattice(L)
    for _ in range(30):
        r1, c1, r2, c2 = rng.integers(0, L, size=4).tolist()
        h = np.zeros((L, L), dtype=np.uint8)
        v = np.zeros((L, L), dtype=np.uint8)
        toggles = lat.dual_path_toggles((r1, c1), (r2, c2))
        lat.apply_toggles(h, v, toggles)
        s = lat.syndrome(h, v)
        expected = np.zeros((L, L), dtype=np.uint8)
        if (r1, c1) != (r2, c2):
            expected[r1, c1] ^= 1
            expected[r2, c2] ^= 1
        assert np.array_equal(s, expected)


def test_torus_distance_matches_brute_force():
    L = 9
    for r1 in range(L):
        for c1 in range(L):
            for r2 in range(L):
                for c2 in range(L):
                    d = ToricLattice.torus_distance((r1, c1), (r2, c2), L)
                    dr = min(abs(r1 - r2), L - abs(r1 - r2))
                    dc = min(abs(c1 - c2), L - abs(c1 - c2))
                    assert d == dr + dc
