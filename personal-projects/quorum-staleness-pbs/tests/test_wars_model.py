import numpy as np

from pbs.wars_model import wars_staleness_curve


def test_staleness_curve_is_monotonically_nonincreasing_in_delta():
    rng = np.random.default_rng(10)
    deltas = np.linspace(0, 5, 15)
    curve = wars_staleness_curve(
        n_replicas=5, w=2, r=2, family="exponential", rate=1.0,
        deltas=deltas, n_trials=50_000, rng=rng,
    )
    diffs = np.diff(curve)
    assert np.all(diffs <= 1e-9)


def test_staleness_curve_bounds_in_unit_interval():
    rng = np.random.default_rng(11)
    deltas = np.linspace(0, 5, 10)
    curve = wars_staleness_curve(
        n_replicas=6, w=3, r=2, family="lognormal", rate=1.0,
        deltas=deltas, n_trials=30_000, rng=rng, sigma=1.0,
    )
    assert np.all(curve >= 0.0) and np.all(curve <= 1.0)


def test_staleness_at_delta_zero_is_positive_for_partial_quorum():
    # W + R <= N (partial quorum): at the instant of commit there should be
    # a nontrivial chance the read quorum has not yet seen the write.
    rng = np.random.default_rng(12)
    curve = wars_staleness_curve(
        n_replicas=5, w=2, r=2, family="exponential", rate=1.0,
        deltas=np.array([0.0]), n_trials=100_000, rng=rng,
    )
    assert curve[0] > 0.05


def test_staleness_decays_to_near_zero_for_large_delta():
    rng = np.random.default_rng(13)
    curve = wars_staleness_curve(
        n_replicas=5, w=2, r=2, family="exponential", rate=1.0,
        deltas=np.array([50.0]), n_trials=50_000, rng=rng,
    )
    assert curve[0] < 0.01


def test_strict_quorum_zero_staleness_wars_model():
    # W + R > N guarantees pigeonhole overlap between write- and read-sets,
    # so staleness probability must be exactly zero for every delta >= 0,
    # regardless of the latency distribution.
    rng = np.random.default_rng(14)
    deltas = np.array([0.0, 0.1, 1.0, 10.0])
    curve = wars_staleness_curve(
        n_replicas=5, w=3, r=3, family="lognormal", rate=2.0,
        deltas=deltas, n_trials=20_000, rng=rng, sigma=1.5,
    )
    assert np.array_equal(curve, np.zeros_like(curve))
