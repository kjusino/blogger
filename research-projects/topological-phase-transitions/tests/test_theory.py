import numpy as np

from tda_phase_transitions import theory


def test_er_threshold_is_one_over_n():
    assert theory.er_giant_component_threshold(100) == 1.0 / 100


def test_rgg_threshold_scales_as_sqrt_log_n_over_n():
    thr = theory.rgg_giant_component_threshold(1000)
    expected = np.sqrt(theory.RGG_CRITICAL_MEAN_DEGREE / (np.pi * 1000))
    assert np.isclose(thr, expected)
    # threshold should shrink as n grows
    assert theory.rgg_giant_component_threshold(4000) < theory.rgg_giant_component_threshold(1000)


def test_chung_lu_threshold_reduces_to_1_over_w_for_constant_weights():
    weights = np.full(50, 3.0)
    thr = theory.chung_lu_giant_component_threshold(weights)
    assert np.isclose(thr, 1.0 / 3.0)


def test_chung_lu_threshold_is_lower_for_heavy_tailed_weights():
    rng = np.random.default_rng(0)
    uniform_weights = np.full(1000, 2.0)
    heavy_weights = rng.pareto(2.5, size=1000) + 1.0
    heavy_weights = heavy_weights * (uniform_weights.mean() / heavy_weights.mean())
    thr_uniform = theory.chung_lu_giant_component_threshold(uniform_weights)
    thr_heavy = theory.chung_lu_giant_component_threshold(heavy_weights)
    # Same mean degree budget, but heavy-tailed weights percolate at a lower theta
    assert thr_heavy < thr_uniform
