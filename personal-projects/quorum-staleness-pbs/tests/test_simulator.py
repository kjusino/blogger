import numpy as np

from pbs.simulator import simulate_staleness_curve


def test_strict_quorum_zero_staleness_simulator_random_selection():
    rng = np.random.default_rng(20)
    deltas = np.array([0.0, 0.5, 5.0])
    result = simulate_staleness_curve(
        n_replicas=5, w=3, r=3, family="exponential", rate=1.0,
        deltas=deltas, n_trials=20_000, rng=rng,
        sigma_het=0.8, selection="random",
    )
    assert np.array_equal(result.curve, np.zeros_like(result.curve))


def test_strict_quorum_zero_staleness_simulator_fixed_selection():
    # The pigeonhole argument (|written set| >= W and |read set| = R with
    # W + R > N) holds regardless of *which* R replicas are read, so it must
    # also hold for the fixed-slowest-subset selection policy.
    rng = np.random.default_rng(21)
    deltas = np.array([0.0, 0.5, 5.0])
    result = simulate_staleness_curve(
        n_replicas=5, w=3, r=3, family="lognormal", rate=1.0,
        deltas=deltas, n_trials=20_000, rng=rng,
        sigma_het=1.2, selection="fixed",
    )
    assert np.array_equal(result.curve, np.zeros_like(result.curve))


def test_homogeneous_random_selection_matches_wars_model_assumptions():
    # sigma_het=0 + selection="random" is definitionally identical to the
    # WARS model's own assumptions -- this is a structural sanity check
    # that the two independent code paths agree, not just a numeric
    # closeness check (that is covered in test_integration.py).
    rng = np.random.default_rng(22)
    deltas = np.linspace(0, 3, 5)
    result = simulate_staleness_curve(
        n_replicas=5, w=2, r=2, family="exponential", rate=1.0,
        deltas=deltas, n_trials=10_000, rng=rng,
        sigma_het=0.0, selection="random",
    )
    assert np.array_equal(result.multipliers, np.ones(5))
    assert result.fixed_idx is None


def test_fixed_selection_returns_slowest_indices():
    rng = np.random.default_rng(23)
    deltas = np.array([1.0])
    result = simulate_staleness_curve(
        n_replicas=6, w=2, r=2, family="exponential", rate=1.0,
        deltas=deltas, n_trials=5_000, rng=rng,
        sigma_het=0.6, selection="fixed",
    )
    assert result.fixed_idx is not None
    assert len(result.fixed_idx) == 2
    # the chosen indices must indeed be the two largest multipliers
    top_two = np.argsort(-result.multipliers)[:2]
    assert set(result.fixed_idx.tolist()) == set(top_two.tolist())


def test_heterogeneous_fixed_slow_selection_increases_staleness_vs_random():
    rng = np.random.default_rng(24)
    deltas = np.linspace(0, 2, 6)
    common_kwargs = dict(
        n_replicas=6, w=2, r=2, family="exponential", rate=1.0,
        deltas=deltas, n_trials=100_000, sigma_het=1.5,
    )
    fixed_result = simulate_staleness_curve(**common_kwargs, selection="fixed", rng=np.random.default_rng(1))
    random_result = simulate_staleness_curve(**common_kwargs, selection="random", rng=np.random.default_rng(1))
    # Reading from the persistently-slowest replicas should be at least as
    # stale, pointwise, as reading from a uniformly random subset -- and
    # strictly worse on average.
    assert fixed_result.curve.mean() > random_result.curve.mean()


def test_invalid_selection_rejected():
    rng = np.random.default_rng(25)
    try:
        simulate_staleness_curve(
            n_replicas=4, w=1, r=1, family="exponential", rate=1.0,
            deltas=np.array([0.0]), n_trials=100, rng=rng, selection="bogus",
        )
        assert False, "expected ValueError"
    except ValueError:
        pass
