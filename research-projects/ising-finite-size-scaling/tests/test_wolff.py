import numpy as np
import pytest

from src.lattice import init_lattice, total_magnetization
from src.wolff import wolff_step, run_wolff


def test_wolff_step_flips_a_contiguous_aligned_cluster():
    rng = np.random.default_rng(0)
    spins = init_lattice(10, rng, mode="ordered")
    size = wolff_step(spins, T=2.0, rng=rng)
    assert 1 <= size <= 100
    # Exactly `size` spins should now be -1 (flipped from the all-+1 start).
    assert np.sum(spins == -1) == size


def test_wolff_cluster_size_grows_with_temperature_near_zero():
    # Near T -> 0, p_add -> 1, so the whole (uniform, ordered-start) lattice
    # should be swept into one giant cluster almost every time.
    rng = np.random.default_rng(1)
    spins = init_lattice(10, rng, mode="ordered")
    size = wolff_step(spins, T=0.05, rng=rng)
    assert size == 100


def test_wolff_high_temperature_magnetization_averages_near_zero():
    result = run_wolff(L=16, T=10.0, n_equil=20, n_sample=100, sample_interval=1, seed=3)
    assert abs(np.mean(result["magnetization"])) < 0.2


def test_wolff_low_temperature_magnetization_stays_near_saturation():
    result = run_wolff(
        L=16, T=0.7, n_equil=20, n_sample=100, sample_interval=1, seed=3, start="ordered"
    )
    assert np.mean(np.abs(result["magnetization"])) > 0.9


def test_wolff_energy_and_magnetization_within_physical_bounds():
    result = run_wolff(L=12, T=2.3, n_equil=30, n_sample=100, sample_interval=1, seed=5)
    assert np.all(result["energy"] >= -2.0 - 1e-9)
    assert np.all(result["energy"] <= 2.0 + 1e-9)
    assert np.all(np.abs(result["magnetization"]) <= 1.0 + 1e-9)
    assert np.all(result["cluster_size"] >= 1)
    assert np.all(result["cluster_size"] <= 12 * 12)


def test_wolff_cluster_size_diverges_near_criticality_relative_to_high_T():
    from src import theory

    rng_hot = np.random.default_rng(9)
    spins_hot = init_lattice(24, rng_hot, mode="random")
    hot_sizes = [wolff_step(spins_hot, T=10.0, rng=rng_hot) for _ in range(50)]

    rng_crit = np.random.default_rng(9)
    spins_crit = init_lattice(24, rng_crit, mode="random")
    for _ in range(200):
        wolff_step(spins_crit, T=theory.T_C, rng=rng_crit)
    crit_sizes = [wolff_step(spins_crit, T=theory.T_C, rng=rng_crit) for _ in range(50)]

    assert np.mean(crit_sizes) > 3 * np.mean(hot_sizes)
