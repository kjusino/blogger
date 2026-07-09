import numpy as np

from src.experiment import minority_measure_upper_bound


class _ConstantModel:
    def __init__(self, preds):
        self._preds = preds

    def predict(self, x):
        return self._preds


def test_minority_upper_bound_is_conservative_when_zero_observed():
    n = 500
    model = _ConstantModel(np.zeros(n, dtype=int))
    ub = minority_measure_upper_bound(model, np.zeros((n, 2)))
    assert ub > 0.0  # never collapse to exactly zero on a finite sample
    assert ub < 0.02  # but should be small for n=500 all-agreeing observations


def test_minority_upper_bound_near_half_for_balanced_predictions():
    n = 2000
    rng = np.random.default_rng(0)
    preds = rng.integers(0, 2, size=n)
    model = _ConstantModel(preds)
    ub = minority_measure_upper_bound(model, np.zeros((n, 2)))
    assert 0.44 < ub <= 0.5


def test_minority_upper_bound_shrinks_with_more_samples_at_fixed_rate():
    rng = np.random.default_rng(1)
    small_n, large_n = 100, 5000
    preds_small = (rng.random(small_n) < 0.05).astype(int)
    preds_large = (rng.random(large_n) < 0.05).astype(int)
    ub_small = minority_measure_upper_bound(_ConstantModel(preds_small), np.zeros((small_n, 2)))
    ub_large = minority_measure_upper_bound(_ConstantModel(preds_large), np.zeros((large_n, 2)))
    assert ub_large < ub_small
