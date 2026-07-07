import numpy as np
import pytest

from src.algorithm import tau, tau_vec, opt, cost, competitive_ratio


class TestTau:
    def test_lambda_zero_is_classical_regardless_of_prediction(self):
        # lambda = 0: tau = b for ANY prediction y (full fallback to the
        # classical worst-case-optimal strategy).
        b = 10
        for y in [0, 1, 5, 9, 10, 11, 100]:
            assert tau(y, 0.0, b) == b

    def test_lambda_one_follows_prediction_exactly(self):
        # lambda = 1: buy immediately if y >= b, else defer to y+1.
        b = 10
        assert tau(11, 1.0, b) == 1  # y >= b -> buy immediately
        assert tau(10, 1.0, b) == 1  # y == b -> buy immediately (y >= b branch)
        assert tau(0, 1.0, b) == 1  # y + 1 = 1
        assert tau(5, 1.0, b) == 6  # y + 1 = 6
        assert tau(9, 1.0, b) == 10  # y + 1 = 10 = b, still in range

    def test_hand_computed_midpoint_lambda(self):
        # b=10, lambda=0.5. tau_star = round(1 + 0.5*9) = round(5.5) = 6
        # (Python's round() is round-half-to-even: 5.5 -> 6).
        b = 10
        lam = 0.5
        assert tau(10, lam, b) == 6  # y >= b branch
        assert tau(0, lam, b) == 6  # y = 0 must equal tau_star exactly
        assert tau(5, lam, b) == 8  # round(6 + 0.5*4) = round(8.0) = 8
        assert tau(9, lam, b) == 10  # round(10 + 0.5*0) = 10

    def test_y_equals_b_boundary_uses_high_branch(self):
        b = 20
        lam = 0.3
        # y == b should hit the y >= b branch, i.e. equal tau_star.
        assert tau(b, lam, b) == tau(b + 1, lam, b)

    def test_clip_to_valid_range(self):
        # tau is always in [1, b].
        for b in [1, 2, 10, 1000]:
            for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
                for y in [0, 1, b // 2, b, b + 1, 5 * b]:
                    t = tau(y, lam, b)
                    assert 1 <= t <= b

    def test_b_equals_one_edge_case(self):
        # With b=1, buying is always day 1 no matter what.
        b = 1
        for lam in [0.0, 0.5, 1.0]:
            for y in [0, 1, 5]:
                assert tau(y, lam, b) == 1

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError):
            tau(5, -0.1, 10)
        with pytest.raises(ValueError):
            tau(5, 1.1, 10)

    def test_invalid_b_raises(self):
        with pytest.raises(ValueError):
            tau(5, 0.5, 0)

    def test_negative_prediction_raises(self):
        with pytest.raises(ValueError):
            tau(-1, 0.5, 10)

    def test_tau_vec_matches_scalar_tau(self):
        b = 37
        lam = 0.42
        ys = np.arange(0, 100)
        vec_result = tau_vec(ys, lam, b)
        scalar_result = np.array([tau(int(y), lam, b) for y in ys])
        np.testing.assert_array_equal(vec_result, scalar_result)


class TestOptCostRatio:
    def test_opt_is_min_x_b(self):
        assert opt(3, 10) == 3
        assert opt(15, 10) == 10
        assert opt(10, 10) == 10

    def test_cost_below_threshold(self):
        # x < tau: cost = x (trip ended before the buy day).
        assert cost(4, 6, 10) == 4

    def test_cost_at_or_above_threshold(self):
        # x >= tau: cost = (tau - 1) + b.
        assert cost(6, 6, 10) == 5 + 10
        assert cost(100, 6, 10) == 5 + 10

    def test_classical_worst_case_ratio_is_two_at_boundary(self):
        # Classical strategy (lambda=0, tau=b): worst case is x = b, giving
        # cost = (b-1) + b = 2b-1, OPT = b, ratio = (2b-1)/b -> 2 as b->inf.
        b = 1000
        t = tau(0, 0.0, b)
        assert t == b
        ratio = competitive_ratio(b, t, b)
        assert ratio == pytest.approx((2 * b - 1) / b)
        assert ratio < 2.0
        assert ratio == pytest.approx(2.0, abs=1e-2)

    def test_perfect_prediction_lambda_one_is_optimal(self):
        # lambda=1, y=x exactly: should always match OPT (ratio = 1).
        b = 50
        for x in [1, 10, 49, 50, 51, 200]:
            t = tau(x, 1.0, b)
            ratio = competitive_ratio(x, t, b)
            assert ratio == pytest.approx(1.0)

    def test_x_zero_edge_case_ratio_convention(self):
        # x=0: both cost and OPT are 0; by convention competitive_ratio
        # returns 1.0 rather than NaN.
        b = 10
        t = tau(0, 0.5, b)
        assert cost(0, t, b) == 0
        assert opt(0, b) == 0
        assert competitive_ratio(0, t, b) == 1.0

    def test_competitive_ratio_vectorized(self):
        b = 20
        t = tau(5, 0.5, b)
        xs = np.array([0, 1, 5, 19, 20, 40])
        ratios = competitive_ratio(xs, t, b)
        assert ratios.shape == xs.shape
        for x, r in zip(xs, ratios):
            assert r == pytest.approx(competitive_ratio(int(x), t, b))
