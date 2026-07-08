"""End-to-end checks that the theory and simulator modules agree when the
simulator's assumptions match the WARS model's, and diverge in a controlled,
directional way when they don't.
"""

import numpy as np

from pbs.comparison import monte_carlo_noise_floor, sup_distance
from pbs.simulator import simulate_staleness_curve
from pbs.wars_model import wars_staleness_curve


def test_validation_configs_match_theory_within_noise_floor():
    deltas = np.linspace(0, 4, 12)
    n_trials = 150_000
    noise_floor = monte_carlo_noise_floor(n_trials)

    configs = [
        dict(n_replicas=5, w=1, r=1, family="exponential", rate=1.0),
        dict(n_replicas=5, w=2, r=2, family="exponential", rate=1.0),
        dict(n_replicas=7, w=2, r=2, family="lognormal", rate=1.0, sigma=1.0),
    ]

    for seed, config in enumerate(configs):
        theory = wars_staleness_curve(
            deltas=deltas, n_trials=n_trials, rng=np.random.default_rng(100 + seed), **config,
        )
        empirical = simulate_staleness_curve(
            deltas=deltas, n_trials=n_trials, rng=np.random.default_rng(200 + seed),
            sigma_het=0.0, selection="random", **config,
        ).curve
        distance = sup_distance(theory, empirical)
        # Allow a small safety margin above the pure noise floor for the
        # extra randomness in the (independently re-drawn) read quorum.
        assert distance < noise_floor * 2.5, (config, distance, noise_floor)


def test_heterogeneous_fixed_config_diverges_beyond_noise_floor():
    deltas = np.linspace(0, 4, 12)
    n_trials = 150_000
    noise_floor = monte_carlo_noise_floor(n_trials)

    config = dict(n_replicas=5, w=2, r=2, family="exponential", rate=1.0)
    theory = wars_staleness_curve(
        deltas=deltas, n_trials=n_trials, rng=np.random.default_rng(300, ), **config,
    )
    empirical = simulate_staleness_curve(
        deltas=deltas, n_trials=n_trials, rng=np.random.default_rng(400),
        sigma_het=1.5, selection="fixed", **config,
    ).curve
    distance = sup_distance(theory, empirical)
    assert distance > noise_floor * 5
    # Directional claim: sticky routing to persistently-slow replicas makes
    # the classical model *optimistic* (it understates real staleness).
    assert np.all(empirical >= theory - 1e-9)


def test_full_experiment_pipeline_runs_end_to_end():
    """Smoke test mirroring experiments/run_experiments.py at reduced scale."""
    deltas = np.linspace(0, 3, 6)
    rng_theory = np.random.default_rng(1)
    rng_empirical = np.random.default_rng(2)

    theory = wars_staleness_curve(
        n_replicas=5, w=2, r=2, family="exponential", rate=1.0,
        deltas=deltas, n_trials=5_000, rng=rng_theory,
    )
    result = simulate_staleness_curve(
        n_replicas=5, w=2, r=2, family="exponential", rate=1.0,
        deltas=deltas, n_trials=5_000, rng=rng_empirical,
        sigma_het=0.5, selection="fixed",
    )
    assert theory.shape == deltas.shape
    assert result.curve.shape == deltas.shape
    assert np.all(np.isfinite(theory))
    assert np.all(np.isfinite(result.curve))
