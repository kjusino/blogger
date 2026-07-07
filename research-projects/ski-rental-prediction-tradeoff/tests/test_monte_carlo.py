import numpy as np
import pytest

from src.monte_carlo import expected_ratio, argmin_lambda, lambda_star_approx, fit_c
from src.predictor import sample_x_lognormal, sample_x_mixture, noisy_prediction


class TestPredictorSamplers:
    def test_lognormal_sampler_shape_and_positivity(self):
        rng = np.random.default_rng(0)
        x = sample_x_lognormal(1000, b=100, rng=rng)
        assert x.shape == (1000,)
        assert np.all(x >= 1)

    def test_mixture_sampler_shape_and_positivity(self):
        rng = np.random.default_rng(0)
        x = sample_x_mixture(1000, b=100, rng=rng)
        assert x.shape == (1000,)
        assert np.all(x >= 1)

    def test_noisy_prediction_sigma_zero_is_exact(self):
        rng = np.random.default_rng(0)
        x = np.array([1, 10, 50, 100, 500])
        y = noisy_prediction(x, sigma=0.0, rng=rng)
        np.testing.assert_array_equal(y, x)

    def test_noisy_prediction_nonnegative(self):
        rng = np.random.default_rng(0)
        x = np.array([1, 2, 3, 100])
        y = noisy_prediction(x, sigma=2.0, rng=rng)
        assert np.all(y >= 0)


class TestExpectedRatio:
    def test_sigma_zero_lambda_one_recovers_near_optimal(self):
        # Perfect predictor (sigma=0) + full trust (lambda=1) should give
        # an expected ratio very close to 1 (matches OPT whenever
        # tau(y=x,...) doesn't get clamped, which is generically the case).
        rng = np.random.default_rng(42)
        r = expected_ratio(1.0, b=100, sigma=0.0, n_samples=5000, rng=rng, x_sampler=sample_x_lognormal)
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_sigma_zero_lambda_zero_recovers_classical_ratio(self):
        # Perfect predictor but lambda=0 (predictor ignored): should behave
        # like the classical strategy, expected ratio somewhere in (1, 2).
        rng = np.random.default_rng(42)
        r = expected_ratio(0.0, b=100, sigma=0.0, n_samples=5000, rng=rng, x_sampler=sample_x_lognormal)
        assert 1.0 <= r <= 2.0

    def test_expected_ratio_is_deterministic_given_seed(self):
        # Same seed -> same sequence of draws -> identical estimate. This
        # is what makes the MC study reproducible for the README numbers.
        r1 = expected_ratio(0.5, b=100, sigma=0.5, n_samples=2000, rng=np.random.default_rng(7), x_sampler=sample_x_lognormal)
        r2 = expected_ratio(0.5, b=100, sigma=0.5, n_samples=2000, rng=np.random.default_rng(7), x_sampler=sample_x_lognormal)
        assert r1 == r2

    def test_expected_ratio_bounded_reasonably(self):
        rng = np.random.default_rng(1)
        for sigma in [0.0, 0.5, 2.0]:
            for lam in [0.0, 0.5, 1.0]:
                r = expected_ratio(lam, b=50, sigma=sigma, n_samples=2000, rng=rng, x_sampler=sample_x_lognormal)
                assert r >= 1.0
                assert np.isfinite(r)


class TestArgminLambda:
    def test_argmin_at_sigma_zero_is_near_one(self):
        # With a perfect predictor, full trust should be (near-)optimal.
        rng = np.random.default_rng(42)
        lambda_grid = np.linspace(0.0, 1.0, 11)
        best_lam, ratios = argmin_lambda(100, 0.0, lambda_grid, 3000, rng, sample_x_lognormal)
        assert best_lam == pytest.approx(1.0)
        assert ratios[-1] == min(ratios)

    def test_ratios_array_has_expected_shape(self):
        rng = np.random.default_rng(0)
        lambda_grid = np.linspace(0.0, 1.0, 5)
        _, ratios = argmin_lambda(50, 0.5, lambda_grid, 500, rng, sample_x_lognormal)
        assert ratios.shape == (5,)


class TestLambdaStarHeuristic:
    def test_heuristic_at_sigma_zero_is_one(self):
        assert lambda_star_approx(0.0, c=1.234) == pytest.approx(1.0)

    def test_heuristic_decreasing_in_sigma(self):
        c = 0.5
        vals = [lambda_star_approx(s, c) for s in [0.0, 0.5, 1.0, 2.0]]
        assert all(v1 >= v2 for v1, v2 in zip(vals, vals[1:]))

    def test_fit_c_recovers_exact_anchor(self):
        c_true = 0.7
        sigma_anchor = 1.5
        lam_anchor = lambda_star_approx(sigma_anchor, c_true)
        c_fit = fit_c(sigma_anchor, lam_anchor)
        assert c_fit == pytest.approx(c_true)

    def test_fit_c_raises_on_zero_anchor(self):
        with pytest.raises(ValueError):
            fit_c(0.0, 1.0)
