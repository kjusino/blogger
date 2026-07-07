import numpy as np
import pytest

from src.lattice import (
    init_lattice,
    neighbor_sum,
    total_energy,
    total_magnetization,
    energy_per_spin,
    magnetization_per_spin,
    site_flip_delta_energy,
)


def test_init_lattice_shapes_and_values():
    rng = np.random.default_rng(0)
    spins = init_lattice(8, rng, mode="random")
    assert spins.shape == (8, 8)
    assert set(np.unique(spins)).issubset({-1, 1})

    ordered = init_lattice(8, rng, mode="ordered")
    assert np.all(ordered == 1)


def test_init_lattice_bad_mode():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        init_lattice(4, rng, mode="bogus")


def test_ground_state_energy_per_spin_is_minus_two():
    rng = np.random.default_rng(0)
    spins = init_lattice(10, rng, mode="ordered")
    assert energy_per_spin(spins) == pytest.approx(-2.0)


def test_ground_state_magnetization_is_one():
    rng = np.random.default_rng(0)
    spins = init_lattice(10, rng, mode="ordered")
    assert magnetization_per_spin(spins) == pytest.approx(1.0)


def test_antiferromagnetic_checkerboard_energy_is_plus_two():
    L = 8
    rows, cols = np.indices((L, L))
    spins = np.where((rows + cols) % 2 == 0, 1, -1).astype(np.int8)
    # Every bond connects opposite-parity sites -> every bond is anti-aligned.
    assert energy_per_spin(spins) == pytest.approx(2.0)
    assert magnetization_per_spin(spins) == pytest.approx(0.0)


def test_total_energy_matches_naive_double_loop():
    rng = np.random.default_rng(1)
    L = 6
    spins = init_lattice(L, rng, mode="random")
    naive = 0.0
    for i in range(L):
        for j in range(L):
            s = spins[i, j]
            right = spins[i, (j + 1) % L]
            down = spins[(i + 1) % L, j]
            naive += -1.0 * s * right + -1.0 * s * down
    assert total_energy(spins) == pytest.approx(naive)


def test_neighbor_sum_range():
    rng = np.random.default_rng(2)
    spins = init_lattice(8, rng, mode="random")
    nb = neighbor_sum(spins)
    assert nb.shape == spins.shape
    assert np.all(np.abs(nb) <= 4)
    assert np.all((nb % 2) == 0)  # sum of 4 +-1 values is always even


def test_site_flip_delta_energy_matches_before_after_difference():
    rng = np.random.default_rng(3)
    L = 6
    spins = init_lattice(L, rng, mode="random")
    for (i, j) in [(0, 0), (2, 3), (5, 5)]:
        e_before = total_energy(spins)
        predicted_delta = site_flip_delta_energy(spins, i, j)
        spins[i, j] *= -1
        e_after = total_energy(spins)
        assert (e_after - e_before) == pytest.approx(predicted_delta)
        spins[i, j] *= -1  # restore
