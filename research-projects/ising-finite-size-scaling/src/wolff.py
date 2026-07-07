"""Wolff single-cluster Monte Carlo algorithm for the 2D Ising model.

Unlike Metropolis single-spin-flip, Wolff builds and flips a whole cluster of
aligned spins per step, with bond-inclusion probability p_add = 1 - exp(-2J/T).
This is rejection-free (the whole cluster flip is always accepted) and greatly
reduces critical slowing down near T_c, at the cost of a variable amount of
work per step (the cluster size itself is a random variable that diverges at
T_c). It is implemented here as a plain BFS since clusters do not vectorize
the way independent single-spin checkerboard updates do.
"""
import numpy as np

from .lattice import total_energy, total_magnetization


def wolff_step(spins, T, rng, J=1.0):
    """Grow and flip one Wolff cluster in place. Returns the cluster size."""
    L = spins.shape[0]
    p_add = 1.0 - np.exp(-2.0 * J / T)

    i0, j0 = rng.integers(0, L), rng.integers(0, L)
    seed_spin = spins[i0, j0]
    in_cluster = np.zeros((L, L), dtype=bool)
    in_cluster[i0, j0] = True
    stack = [(i0, j0)]
    size = 1

    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = (i + di) % L, (j + dj) % L
            if not in_cluster[ni, nj] and spins[ni, nj] == seed_spin:
                if rng.random() < p_add:
                    in_cluster[ni, nj] = True
                    stack.append((ni, nj))
                    size += 1

    spins[in_cluster] *= -1
    return size


def run_wolff(L, T, n_equil, n_sample, sample_interval, seed, J=1.0, start="random"):
    """Run Wolff MC and return per-spin observable time series plus cluster sizes."""
    rng = np.random.default_rng(seed)
    from .lattice import init_lattice

    spins = init_lattice(L, rng, mode=start)
    N = L * L

    for _ in range(n_equil):
        wolff_step(spins, T, rng, J)

    energies = np.empty(n_sample)
    mags = np.empty(n_sample)
    cluster_sizes = np.empty(n_sample)
    for k in range(n_sample):
        last_size = 1
        for _ in range(sample_interval):
            last_size = wolff_step(spins, T, rng, J)
        cluster_sizes[k] = last_size
        energies[k] = total_energy(spins, J) / N
        mags[k] = total_magnetization(spins) / N

    return {
        "energy": energies,
        "magnetization": mags,
        "cluster_size": cluster_sizes,
        "L": L,
        "T": T,
        "seed": seed,
    }
