"""Vectorized checkerboard (red-black) Metropolis Monte Carlo for the 2D Ising model.

On a square lattice with periodic boundary conditions and an even side length L,
sites split into two color classes by parity of (i + j); every site's 4 neighbors
lie in the opposite class. That makes each color class an independent-update set:
flipping any subset of same-color spins simultaneously cannot change any other
same-color spin's local field, so a synchronous vectorized update within a color
class satisfies detailed balance exactly like a sequential single-spin Metropolis
sweep would. One "sweep" below updates both color classes once (every site
touched once), matching the usual definition used in the Ising MC literature.
"""
import numpy as np

from .lattice import neighbor_sum, total_energy, total_magnetization

_ROWS_CACHE = {}


def _checkerboard_mask(L):
    if L not in _ROWS_CACHE:
        rows, cols = np.indices((L, L))
        _ROWS_CACHE[L] = ((rows + cols) % 2 == 0)
    return _ROWS_CACHE[L]


def checkerboard_sweep(spins, T, rng, J=1.0):
    """Perform one full sweep (both color classes) in place. Returns spins."""
    if spins.shape[0] % 2 != 0:
        raise ValueError("checkerboard updates require an even lattice side length")
    even_mask = _checkerboard_mask(spins.shape[0])
    for mask in (even_mask, ~even_mask):
        nb = neighbor_sum(spins)
        delta_e = 2.0 * J * spins * nb
        accept_prob = np.exp(np.minimum(-delta_e / T, 0.0))
        rnd = rng.random(spins.shape)
        flip = mask & ((delta_e <= 0) | (rnd < accept_prob))
        spins[flip] *= -1
    return spins


def run_metropolis(L, T, n_equil, n_sample, sample_interval, seed, J=1.0, start="random"):
    """Run Metropolis MC and return per-spin observable time series.

    Returns a dict with 'energy' and 'magnetization' arrays (length n_sample),
    each the per-spin (intensive) value sampled every `sample_interval` sweeps
    after `n_equil` equilibration sweeps.
    """
    rng = np.random.default_rng(seed)
    from .lattice import init_lattice

    spins = init_lattice(L, rng, mode=start)
    N = L * L

    for _ in range(n_equil):
        checkerboard_sweep(spins, T, rng, J)

    energies = np.empty(n_sample)
    mags = np.empty(n_sample)
    for k in range(n_sample):
        for _ in range(sample_interval):
            checkerboard_sweep(spins, T, rng, J)
        energies[k] = total_energy(spins, J) / N
        mags[k] = total_magnetization(spins) / N

    return {"energy": energies, "magnetization": mags, "L": L, "T": T, "seed": seed}
