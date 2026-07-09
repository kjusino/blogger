import numpy as np
import pytest

from src.regret import exp3_bound, fit_log_law, fit_power_law, run_bandit, ucb1_bound


class _AlwaysArm:
    """A dummy algorithm that always plays a fixed arm, for exact regret checks."""

    def __init__(self, arm):
        self.arm = arm

    def select_arm(self):
        return self.arm

    def update(self, arm, reward):
        pass


def test_run_bandit_zero_regret_when_always_playing_the_best_arm():
    # Arm 0 always pays 1, arm 1 always pays 0 -> arm 0 is best in hindsight.
    table = np.zeros((100, 2))
    table[:, 0] = 1.0
    result = run_bandit(_AlwaysArm(0), table)
    assert result["final_regret"] == pytest.approx(0.0)
    assert np.all(result["regret_trace"] == 0.0)


def test_run_bandit_full_regret_when_always_playing_the_worst_arm():
    table = np.zeros((100, 2))
    table[:, 0] = 1.0
    result = run_bandit(_AlwaysArm(1), table)
    assert result["final_regret"] == pytest.approx(100.0)
    assert result["regret_trace"][-1] == pytest.approx(100.0)
    assert result["regret_trace"][49] == pytest.approx(50.0)


def test_run_bandit_records_arms_and_rewards():
    table = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    result = run_bandit(_AlwaysArm(0), table)
    assert result["arms_played"].tolist() == [0, 0, 0]
    assert result["rewards_received"].tolist() == [1.0, 0.0, 1.0]


def test_exp3_bound_increases_with_horizon_and_arms():
    assert exp3_bound(4, 1000) < exp3_bound(4, 10000)
    assert exp3_bound(4, 1000) < exp3_bound(16, 1000)


def test_exp3_bound_matches_hand_computation():
    # 2*sqrt(e-1)*sqrt(K*T*ln K) for K=4, T=100
    expected = 2 * np.sqrt(np.e - 1) * np.sqrt(4 * 100 * np.log(4))
    assert exp3_bound(4, 100) == pytest.approx(expected)


def test_ucb1_bound_zero_with_no_gaps():
    assert ucb1_bound([0.0, 0.0], 1000) == 0.0


def test_ucb1_bound_positive_and_increases_with_horizon():
    low = ucb1_bound([0.1, 0.2], 1000)
    high = ucb1_bound([0.1, 0.2], 100000)
    assert low > 0
    assert high > low


def test_fit_power_law_recovers_known_exponent():
    x = np.array([10.0, 100.0, 1000.0, 10000.0])
    y = 3.0 * x**0.5
    exponent, r_squared = fit_power_law(x, y)
    assert exponent == pytest.approx(0.5, abs=1e-9)
    assert r_squared == pytest.approx(1.0, abs=1e-9)


def test_fit_power_law_handles_noisy_data():
    rng = np.random.default_rng(0)
    x = np.linspace(100, 10000, 30)
    y = 2.0 * x**0.5 * (1 + rng.normal(0, 0.02, size=30))
    exponent, r_squared = fit_power_law(x, y)
    assert exponent == pytest.approx(0.5, abs=0.05)
    assert r_squared > 0.95


def test_fit_log_law_recovers_known_coefficients():
    x = np.array([10.0, 100.0, 1000.0, 10000.0])
    y = 5.0 * np.log(x) + 2.0
    a, b, r_squared = fit_log_law(x, y)
    assert a == pytest.approx(5.0, abs=1e-9)
    assert b == pytest.approx(2.0, abs=1e-9)
    assert r_squared == pytest.approx(1.0, abs=1e-9)
