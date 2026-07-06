import numpy as np
import pytest

from src.distributions import paired_perturbation, uniform
from src.theory import paired_collision_probability
from src.testers import (
    TESTERS,
    collision_probability,
    collision_statistic,
    collision_tester,
    naive_learner_tester,
)


def test_collision_statistic_manual_example():
    # counts: value 0 -> 2, value 1 -> 1, value 2 -> 3
    samples = np.array([0, 0, 1, 2, 2, 2])
    # collisions = C(2,2) + C(1,2) + C(3,2) = 1 + 0 + 3 = 4
    assert collision_statistic(samples, n=3) == 4


def test_collision_statistic_no_collisions_when_all_distinct():
    samples = np.array([0, 1, 2, 3])
    assert collision_statistic(samples, n=4) == 0


def test_collision_probability_of_uniform_is_one_over_n():
    p = uniform(37)
    assert collision_probability(p) == pytest.approx(1.0 / 37)


def test_collision_probability_matches_paired_theory_formula():
    rng = np.random.default_rng(0)
    n, epsilon = 40, 0.2
    p = paired_perturbation(n, epsilon, rng)
    assert collision_probability(p) == pytest.approx(
        paired_collision_probability(n, epsilon), abs=1e-12
    )


def test_collision_tester_rejects_far_distribution_more_than_uniform():
    """With enough samples, the collision tester's rejection rate against a
    clearly-far distribution should be much higher than its false-positive
    rate against uniform, at the same (n, epsilon, m)."""
    rng = np.random.default_rng(123)
    n, epsilon, m, trials = 200, 0.3, 400, 60

    far_rejections = 0
    uniform_rejections = 0
    for _ in range(trials):
        p = paired_perturbation(n, epsilon, rng)
        far_samples = rng.choice(n, size=m, p=p)
        uniform_samples = rng.integers(0, n, size=m)
        far_rejections += collision_tester(far_samples, n, epsilon)
        uniform_rejections += collision_tester(uniform_samples, n, epsilon)

    far_power = far_rejections / trials
    uniform_fpr = uniform_rejections / trials
    assert far_power > uniform_fpr + 0.3


def test_collision_tester_requires_at_least_two_samples():
    with pytest.raises(ValueError):
        collision_tester(np.array([0]), n=10, epsilon=0.1)


def test_naive_learner_tester_rejects_far_distribution_more_than_uniform():
    rng = np.random.default_rng(7)
    n, epsilon, m, trials = 50, 0.3, 4000, 60

    far_rejections = 0
    uniform_rejections = 0
    for _ in range(trials):
        p = paired_perturbation(n, epsilon, rng)
        far_samples = rng.choice(n, size=m, p=p)
        uniform_samples = rng.integers(0, n, size=m)
        far_rejections += naive_learner_tester(far_samples, n, epsilon)
        uniform_rejections += naive_learner_tester(uniform_samples, n, epsilon)

    far_power = far_rejections / trials
    uniform_fpr = uniform_rejections / trials
    assert far_power > uniform_fpr + 0.3


def test_testers_registry_contains_both():
    assert set(TESTERS) == {"collision", "naive_learner"}
    assert TESTERS["collision"] is collision_tester
    assert TESTERS["naive_learner"] is naive_learner_tester
