import pytest

from src.density_evolution import (
    de_converges_to_zero,
    de_step,
    de_trajectory,
    find_threshold,
)


def test_de_step_zero_erasure_stays_zero():
    assert de_step(0.0, epsilon=0.0, dv=3, dc=6) == 0.0


def test_de_step_matches_hand_computation():
    # x=0.2, eps=0.4, dv=3, dc=6:
    # x' = 0.4 * (1 - (1-0.2)^5)^2 = 0.4 * (1 - 0.8^5)^2
    expected = 0.4 * (1 - 0.8**5) ** 2
    assert de_step(0.2, 0.4, 3, 6) == pytest.approx(expected)


def test_converges_below_known_threshold():
    # literature threshold for (3,6) is ~0.4294; comfortably below it
    assert de_converges_to_zero(0.35, dv=3, dc=6)


def test_diverges_above_known_threshold():
    assert not de_converges_to_zero(0.55, dv=3, dc=6)


def test_threshold_matches_literature_value_for_3_6():
    # Richardson & Urbanke, "Modern Coding Theory": (3,6) BEC threshold ~= 0.4294
    threshold = find_threshold(3, 6, tol=1e-7)
    assert threshold == pytest.approx(0.4294, abs=2e-4)


def test_threshold_is_monotonic_in_check_degree():
    # Fixing dv, a larger dc means a sparser/weaker code -> lower threshold.
    t_dc6 = find_threshold(3, 6)
    t_dc8 = find_threshold(3, 8)
    assert t_dc8 < t_dc6


def test_trajectory_length_and_monotonicity_below_threshold():
    traj = de_trajectory(0.3, dv=3, dc=6, n_iters=40)
    assert len(traj) == 41
    assert traj[0] == 0.3
    # non-increasing sequence converging toward 0
    assert all(traj[i + 1] <= traj[i] + 1e-12 for i in range(len(traj) - 1))
    assert traj[-1] < 1e-6


def test_trajectory_stalls_above_threshold():
    traj = de_trajectory(0.6, dv=3, dc=6, n_iters=200)
    assert traj[-1] > 0.05  # settles at a nonzero fixed point, doesn't vanish


def test_rate_of_regular_ensemble():
    # rate = 1 - dv/dc; used only in the summary but worth pinning down
    assert 1 - 3 / 6 == pytest.approx(0.5)
