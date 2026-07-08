import numpy as np
import pytest

from src.fitting import fit_growth_models


def test_recovers_loglog_synthetic_data():
    ns = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192], dtype=float)
    true_a, true_b = 1.44, 0.5  # 1/ln(2) ~ ABKU constant
    gaps = true_a * np.log(np.log(ns)) + true_b
    fit = fit_growth_models(ns, gaps)
    assert fit.loglog_r2 > 0.999
    assert fit.best_model == "loglog"
    assert fit.loglog_params[0] == pytest.approx(true_a, rel=1e-2)


def test_recovers_log_synthetic_data():
    ns = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192], dtype=float)
    true_a, true_b = 0.9, -1.0
    gaps = true_a * np.log(ns) + true_b
    fit = fit_growth_models(ns, gaps)
    assert fit.log_r2 > 0.999
    assert fit.best_model == "log"
    assert fit.log_params[0] == pytest.approx(true_a, rel=1e-2)


def test_noisy_loglog_data_still_prefers_loglog_model():
    rng = np.random.default_rng(0)
    ns = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384], dtype=float)
    gaps = 1.44 * np.log(np.log(ns)) + 0.5 + rng.normal(0, 0.05, size=ns.shape)
    fit = fit_growth_models(ns, gaps)
    assert fit.best_model == "loglog"


def test_requires_at_least_three_points():
    with pytest.raises(ValueError):
        fit_growth_models([10, 20], [1, 2])


def test_rejects_n_less_or_equal_one():
    with pytest.raises(ValueError):
        fit_growth_models([1, 10, 100], [0, 1, 2])
