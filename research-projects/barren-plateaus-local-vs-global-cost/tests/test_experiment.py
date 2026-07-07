import numpy as np
import pytest

from barren_plateaus import experiment


def test_estimate_gradient_variance_shape_and_reproducibility():
    r1 = experiment.estimate_gradient_variance("global", n=3, depth=2, num_samples=30, seed=42)
    r2 = experiment.estimate_gradient_variance("global", n=3, depth=2, num_samples=30, seed=42)
    assert r1.variance == pytest.approx(r2.variance)
    assert r1.samples.shape == (30,)
    assert r1.variance >= 0.0
    assert r1.variance_stderr >= 0.0


def test_different_seeds_give_different_samples():
    r1 = experiment.estimate_gradient_variance("local", n=3, depth=2, num_samples=20, seed=1)
    r2 = experiment.estimate_gradient_variance("local", n=3, depth=2, num_samples=20, seed=2)
    assert not np.allclose(r1.samples, r2.samples)


def test_unknown_cost_type_raises():
    with pytest.raises(ValueError):
        experiment.estimate_gradient_variance("nonexistent", n=2, depth=1, num_samples=5, seed=0)


def test_run_sweep_produces_full_grid():
    results = experiment.run_sweep(
        cost_types=["global", "local"], depths=[1, 2], n_values=[2, 3], num_samples=10, base_seed=0
    )
    assert len(results) == 2 * 2 * 2
    seen = {(r.cost_type, r.depth, r.n) for r in results}
    assert seen == {
        ("global", 1, 2), ("global", 1, 3), ("global", 2, 2), ("global", 2, 3),
        ("local", 1, 2), ("local", 1, 3), ("local", 2, 2), ("local", 2, 3),
    }


def test_global_cost_gradient_mean_near_zero_by_symmetry():
    # E[dC/dtheta] = 0 exactly by the theta -> -theta / theta -> theta+pi symmetry
    # of a uniformly-random-angle ensemble; check it's small relative to the std dev.
    r = experiment.estimate_gradient_variance("global", n=4, depth=2, num_samples=400, seed=7)
    std = np.sqrt(r.variance)
    assert abs(r.mean) < 0.3 * std / np.sqrt(400) * 10  # generous bound, just guards against a sign/indexing bug


@pytest.mark.integration
def test_global_cost_variance_smaller_than_local_at_moderate_width():
    # The core scientific claim, checked end-to-end at a small but nontrivial
    # size: at fixed shallow depth, the global cost's gradient variance is
    # already substantially smaller (steeper concentration) than the local
    # cost's, at 8 qubits.
    n, depth = 8, 2
    global_r = experiment.estimate_gradient_variance("global", n=n, depth=depth, num_samples=150, seed=100)
    local_r = experiment.estimate_gradient_variance("local", n=n, depth=depth, num_samples=150, seed=200)
    assert global_r.variance < local_r.variance


@pytest.mark.integration
def test_variance_decreases_with_width_for_global_cost():
    depth = 2
    small = experiment.estimate_gradient_variance("global", n=4, depth=depth, num_samples=150, seed=10)
    large = experiment.estimate_gradient_variance("global", n=10, depth=depth, num_samples=150, seed=11)
    assert large.variance < small.variance
