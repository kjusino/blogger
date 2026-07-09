import numpy as np
import pytest

from src.functions import MajorityFunction, ParityFunction
from src.noise_sensitivity import (
    majority_sheppard_limit,
    monte_carlo_noise_sensitivity,
    parity_noise_sensitivity_exact,
)


def test_noise_sensitivity_zero_delta_is_zero():
    f = ParityFunction(n=11)
    rng = np.random.default_rng(0)
    result = monte_carlo_noise_sensitivity(f, delta=0.0, n_samples=2000, rng=rng)
    assert result.estimate == pytest.approx(0.0, abs=1e-9)


def test_noise_sensitivity_rejects_bad_delta():
    f = ParityFunction(n=5)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        monte_carlo_noise_sensitivity(f, delta=1.5, n_samples=100, rng=rng)
    with pytest.raises(ValueError):
        monte_carlo_noise_sensitivity(f, delta=-0.1, n_samples=100, rng=rng)


def test_parity_noise_sensitivity_matches_closed_form():
    n = 9
    f = ParityFunction(n=n)
    rng = np.random.default_rng(3)
    for delta in (0.05, 0.1, 0.25, 0.5):
        result = monte_carlo_noise_sensitivity(f, delta=delta, n_samples=30000, rng=rng)
        expected = parity_noise_sensitivity_exact(n, delta)
        assert result.estimate == pytest.approx(expected, abs=0.02)


def test_noise_sensitivity_monotonic_in_delta_for_majority():
    f = MajorityFunction(n=21)
    rng = np.random.default_rng(4)
    deltas = [0.01, 0.1, 0.3, 0.5]
    estimates = [
        monte_carlo_noise_sensitivity(f, delta=d, n_samples=20000, rng=rng).estimate
        for d in deltas
    ]
    assert all(estimates[i] <= estimates[i + 1] + 0.02 for i in range(len(estimates) - 1))


def test_majority_noise_sensitivity_converges_to_sheppard_limit_as_n_grows():
    delta = 0.2
    limit = majority_sheppard_limit(delta)
    rng = np.random.default_rng(7)
    errors = []
    for n in (11, 401, 4001):
        f = MajorityFunction(n=n)
        result = monte_carlo_noise_sensitivity(f, delta=delta, n_samples=20000, rng=rng)
        errors.append(abs(result.estimate - limit))
    # error should shrink (not strictly monotonically, but the last should be
    # much smaller than the first) as n grows toward the CLT limit.
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.03


def test_sheppard_limit_matches_known_special_cases():
    # delta=1 (fully independent, balanced function): arccos(0)/pi = 0.5
    assert majority_sheppard_limit(1.0) == pytest.approx(0.5)
    # delta=0 (no noise): arccos(1)/pi = 0
    assert majority_sheppard_limit(0.0) == pytest.approx(0.0)


def test_noise_sensitivity_at_delta_one_is_fully_independent_value():
    # delta=1 rerandomizes every coordinate, so y is fully independent of x:
    # NS_1(f) = P[f(x)!=f(y)] for independent x,y = 2p(1-p), p=Pr[f=1].
    f = ParityFunction(n=6)  # balanced, p=0.5 -> NS_1 = 0.5
    rng = np.random.default_rng(5)
    result = monte_carlo_noise_sensitivity(f, delta=1.0, n_samples=30000, rng=rng)
    assert result.estimate == pytest.approx(0.5, abs=0.02)
