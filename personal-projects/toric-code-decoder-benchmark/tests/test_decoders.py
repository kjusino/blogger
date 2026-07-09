import numpy as np
import pytest

from toriccode.decoders import mwpm, union_find
from toriccode.stabilizer import ToricLattice

DECODE_FNS = {"mwpm": mwpm.decode, "union_find": union_find.decode}


@pytest.mark.parametrize("decoder", ["mwpm", "union_find"])
@pytest.mark.parametrize("L", [4, 5, 6])
def test_decoder_produces_matching_syndrome(decoder, L, rng=np.random.default_rng(2)):
    """A decoder's correction must always cancel the observed syndrome,
    regardless of whether it ends up choosing the "wrong" homology class."""
    lat = ToricLattice(L)
    decode_fn = DECODE_FNS[decoder]
    for _ in range(25):
        h_err = (rng.random((L, L)) < 0.1).astype(np.uint8)
        v_err = (rng.random((L, L)) < 0.1).astype(np.uint8)
        syndrome = lat.syndrome(h_err, v_err)
        h_corr, v_corr = decode_fn(lat, syndrome)
        residual = lat.syndrome(h_err ^ h_corr, v_err ^ v_corr)
        assert not residual.any()


@pytest.mark.parametrize("decoder", ["mwpm", "union_find"])
def test_zero_syndrome_gives_zero_correction(decoder):
    L = 5
    lat = ToricLattice(L)
    decode_fn = DECODE_FNS[decoder]
    syndrome = np.zeros((L, L), dtype=np.uint8)
    h_corr, v_corr = decode_fn(lat, syndrome)
    assert not h_corr.any()
    assert not v_corr.any()


@pytest.mark.parametrize("decoder", ["mwpm", "union_find"])
def test_single_isolated_error_pair_is_corrected_exactly(decoder):
    """Two well-separated single-qubit errors, far below the code distance's
    error-correcting radius, must be decoded back to the identity class."""
    L = 9
    lat = ToricLattice(L)
    decode_fn = DECODE_FNS[decoder]
    h_err = np.zeros((L, L), dtype=np.uint8)
    v_err = np.zeros((L, L), dtype=np.uint8)
    h_err[2, 2] = 1
    v_err[6, 6] = 1
    syndrome = lat.syndrome(h_err, v_err)
    h_corr, v_corr = decode_fn(lat, syndrome)
    assert not lat.syndrome(h_err ^ h_corr, v_err ^ v_corr).any()
    assert lat.is_trivial(h_err ^ h_corr, v_err ^ v_corr)


def test_mwpm_and_union_find_agree_on_low_noise_instances(rng=np.random.default_rng(3)):
    """Well below threshold, both decoders should almost always recover the
    same (trivial) logical class."""
    L = 7
    lat = ToricLattice(L)
    p = 0.02
    agreements = 0
    n = 60
    for _ in range(n):
        h_err = (rng.random((L, L)) < p).astype(np.uint8)
        v_err = (rng.random((L, L)) < p).astype(np.uint8)
        syndrome = lat.syndrome(h_err, v_err)

        h1, v1 = mwpm.decode(lat, syndrome)
        h2, v2 = union_find.decode(lat, syndrome)

        trivial1 = lat.is_trivial(h_err ^ h1, v_err ^ v1)
        trivial2 = lat.is_trivial(h_err ^ h2, v_err ^ v2)
        agreements += int(trivial1 == trivial2)

    assert agreements / n > 0.9


def test_cluster_defects_respects_growing_radius():
    from toriccode.decoders.union_find import cluster_defects

    L = 10
    # Two well-separated pairs of defects, close within each pair.
    defects = [(0, 0), (0, 1), (5, 5), (5, 6)]
    clusters = cluster_defects(defects, L)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2, 2]


def test_cluster_defects_all_even_parity():
    from collections import Counter

    from toriccode.decoders.union_find import cluster_defects

    L = 8
    rng = np.random.default_rng(4)
    for _ in range(20):
        m = int(rng.integers(1, 6)) * 2
        defects = [tuple(rng.integers(0, L, size=2).tolist()) for _ in range(m)]
        clusters = cluster_defects(defects, L)
        for cluster in clusters:
            assert len(cluster) % 2 == 0
