"""Square-lattice 2D Ising model with periodic boundary conditions."""
import numpy as np


def init_lattice(L, rng, mode="random"):
    """Return an L x L array of +-1 spins.

    mode="random": each spin independently +1/-1 (infinite-temperature start).
    mode="ordered": all spins +1 (ground-state start).
    """
    if mode == "random":
        return rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    elif mode == "ordered":
        return np.ones((L, L), dtype=np.int8)
    raise ValueError(f"unknown mode: {mode}")


def neighbor_sum(spins):
    """Sum of the 4 periodic nearest neighbors for every site, vectorized."""
    return (
        np.roll(spins, 1, axis=0)
        + np.roll(spins, -1, axis=0)
        + np.roll(spins, 1, axis=1)
        + np.roll(spins, -1, axis=1)
    )


def total_energy(spins, J=1.0):
    """Total Hamiltonian H = -J * sum_{<i,j>} s_i s_j, each bond counted once."""
    right = spins * np.roll(spins, -1, axis=1)
    down = spins * np.roll(spins, -1, axis=0)
    return -J * float(np.sum(right) + np.sum(down))


def total_magnetization(spins):
    return float(np.sum(spins))


def energy_per_spin(spins, J=1.0):
    L = spins.shape[0]
    return total_energy(spins, J) / (L * L)


def magnetization_per_spin(spins):
    L = spins.shape[0]
    return total_magnetization(spins) / (L * L)


def site_flip_delta_energy(spins, i, j, J=1.0):
    """Naive (non-vectorized, single-site) energy cost of flipping site (i, j).

    Used only as an independent reference for testing the vectorized
    checkerboard update path in metropolis.py.
    """
    L = spins.shape[0]
    s = spins[i, j]
    nb = (
        spins[(i + 1) % L, j]
        + spins[(i - 1) % L, j]
        + spins[i, (j + 1) % L]
        + spins[i, (j - 1) % L]
    )
    return 2.0 * J * s * nb
