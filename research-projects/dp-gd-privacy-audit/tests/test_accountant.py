import math

import numpy as np
import pytest

from src.accountant import classical_gaussian_epsilon, epsilon_from_rdp, rdp_alpha


def test_composition_additivity_exact():
    for alpha, sigma, T in [(2.0, 1.0, 1), (5.0, 0.7, 4), (10.0, 2.0, 8), (3.5, 1.5, 3)]:
        assert rdp_alpha(alpha, sigma, T) == pytest.approx(T * rdp_alpha(alpha, sigma, 1))


def test_epsilon_strictly_decreases_as_sigma_increases():
    delta = 1e-5
    T = 4
    sigmas = [0.3, 0.5, 1.0, 2.0, 4.0]
    epsilons = [epsilon_from_rdp(s, T, delta)[0] for s in sigmas]
    for i in range(len(epsilons) - 1):
        assert epsilons[i] > epsilons[i + 1]


def test_epsilon_strictly_increases_as_T_increases():
    delta = 1e-5
    sigma = 1.0
    Ts = [1, 2, 4, 8, 16]
    epsilons = [epsilon_from_rdp(sigma, T, delta)[0] for T in Ts]
    for i in range(len(epsilons) - 1):
        assert epsilons[i] < epsilons[i + 1]


def test_epsilon_strictly_decreases_as_delta_increases():
    sigma = 1.0
    T = 4
    deltas = [1e-8, 1e-6, 1e-4, 1e-2]
    epsilons = [epsilon_from_rdp(sigma, T, d)[0] for d in deltas]
    for i in range(len(epsilons) - 1):
        assert epsilons[i] > epsilons[i + 1]


def test_t1_rdp_vs_classical_within_factor_of_2():
    delta = 1e-5
    for sigma in [0.5, 1.0, 2.0, 4.0]:
        rdp_eps, _ = epsilon_from_rdp(sigma, 1, delta)
        classical_eps = classical_gaussian_epsilon(sigma, delta)
        ratio = rdp_eps / classical_eps
        assert 0.5 <= ratio <= 2.0


def test_epsilon_finite_positive_for_reasonable_inputs():
    for sigma in [0.5, 1.0, 2.0]:
        for T in [1, 2, 8]:
            eps, alpha_star = epsilon_from_rdp(sigma, T, 1e-5)
            assert math.isfinite(eps)
            assert eps > 0
            assert alpha_star > 1


def test_epsilon_degenerate_inputs_return_inf_not_crash():
    eps, alpha_star = epsilon_from_rdp(1.0, 4, delta=0.0)
    assert eps == math.inf

    eps, alpha_star = epsilon_from_rdp(1.0, 4, delta=-0.1)
    assert eps == math.inf

    eps, alpha_star = epsilon_from_rdp(0.0, 4, delta=1e-5)
    assert eps == math.inf

    eps, alpha_star = epsilon_from_rdp(1.0, 0, delta=1e-5)
    assert eps == math.inf


def test_classical_gaussian_epsilon_degenerate():
    assert classical_gaussian_epsilon(0.0, 1e-5) == math.inf
    assert classical_gaussian_epsilon(1.0, 0.0) == math.inf
    assert classical_gaussian_epsilon(1.0, 1.5) == math.inf
