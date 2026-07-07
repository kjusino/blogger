import numpy as np
import pytest

from src.lattice import init_lattice, total_energy, total_magnetization
from src.metropolis import checkerboard_sweep, run_metropolis, _checkerboard_mask


def test_checkerboard_mask_is_proper_2_coloring_on_periodic_lattice():
    L = 8
    mask = _checkerboard_mask(L)
    # Every neighbor of a "True"-colored site must be "False"-colored, and
    # vice versa -- this is what makes simultaneous same-color updates valid.
    rolled_up = np.roll(mask, 1, axis=0)
    rolled_down = np.roll(mask, -1, axis=0)
    rolled_left = np.roll(mask, 1, axis=1)
    rolled_right = np.roll(mask, -1, axis=1)
    for rolled in (rolled_up, rolled_down, rolled_left, rolled_right):
        assert np.all(rolled[mask] == False)
        assert np.all(rolled[~mask] == True)


def test_checkerboard_sweep_rejects_odd_lattice_size():
    rng = np.random.default_rng(0)
    spins = init_lattice(7, rng, mode="random")
    with pytest.raises(ValueError):
        checkerboard_sweep(spins, T=2.0, rng=rng)


def test_zero_temperature_limit_only_lowers_energy():
    # At T -> 0 (approximated by a very small T), Metropolis should behave
    # like greedy energy descent: no uphill moves get accepted.
    rng = np.random.default_rng(0)
    spins = init_lattice(10, rng, mode="random")
    e_prev = total_energy(spins)
    for _ in range(50):
        checkerboard_sweep(spins, T=1e-6, rng=rng)
        e_now = total_energy(spins)
        assert e_now <= e_prev + 1e-9
        e_prev = e_now


def test_ground_state_is_a_fixed_point_at_low_temperature():
    rng = np.random.default_rng(0)
    spins = init_lattice(8, rng, mode="ordered")
    for _ in range(20):
        checkerboard_sweep(spins, T=0.1, rng=rng)
    assert total_magnetization(spins) == pytest.approx(64.0)


def test_high_temperature_magnetization_averages_near_zero():
    result = run_metropolis(L=16, T=10.0, n_equil=50, n_sample=200, sample_interval=1, seed=42)
    assert abs(np.mean(result["magnetization"])) < 0.15


def test_low_temperature_magnetization_stays_near_saturation():
    result = run_metropolis(
        L=16, T=0.5, n_equil=200, n_sample=200, sample_interval=1, seed=42, start="ordered"
    )
    assert np.mean(np.abs(result["magnetization"])) > 0.9


def test_energy_and_magnetization_within_physical_bounds():
    result = run_metropolis(L=10, T=2.5, n_equil=100, n_sample=100, sample_interval=1, seed=1)
    assert np.all(result["energy"] >= -2.0 - 1e-9)
    assert np.all(result["energy"] <= 2.0 + 1e-9)
    assert np.all(np.abs(result["magnetization"]) <= 1.0 + 1e-9)


def test_checkerboard_vectorized_matches_naive_single_spin_metropolis_statistically():
    """The vectorized checkerboard update should reach the same equilibrium
    energy distribution as a textbook sequential single-spin Metropolis sweep,
    since both satisfy detailed balance for the same Gibbs distribution.
    """
    L = 8
    T = 2.5
    N = L * L

    def naive_sweep(spins, T, rng):
        for i in range(L):
            for j in range(L):
                s = spins[i, j]
                nb = (
                    spins[(i + 1) % L, j]
                    + spins[(i - 1) % L, j]
                    + spins[i, (j + 1) % L]
                    + spins[i, (j - 1) % L]
                )
                delta = 2.0 * s * nb
                if delta <= 0 or rng.random() < np.exp(-delta / T):
                    spins[i, j] *= -1
        return spins

    rng_a = np.random.default_rng(7)
    spins_a = init_lattice(L, rng_a, mode="random")
    energies_a = []
    for _ in range(400):
        checkerboard_sweep(spins_a, T, rng_a)
        energies_a.append(total_energy(spins_a) / N)

    rng_b = np.random.default_rng(7)
    spins_b = init_lattice(L, rng_b, mode="random")
    energies_b = []
    for _ in range(400):
        naive_sweep(spins_b, T, rng_b)
        energies_b.append(total_energy(spins_b) / N)

    # Same seed and same physics but different (valid) update schedules will
    # not track identically step by step; compare converged means instead.
    mean_a = np.mean(energies_a[200:])
    mean_b = np.mean(energies_b[200:])
    assert mean_a == pytest.approx(mean_b, abs=0.15)
