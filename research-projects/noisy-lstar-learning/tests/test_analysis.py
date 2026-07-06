import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (
    success_rate_table,
    mean_field_table,
    repetition_factor_table,
    union_bound_success_prediction,
    sorted_noise_rates,
)


def fake_results():
    return [
        {"noise_rate": 0.0, "strategy": "none", "success": True, "raw_queries": 10,
         "distinct_queries": 10, "hypothesis_num_states": 5},
        {"noise_rate": 0.0, "strategy": "none", "success": True, "raw_queries": 12,
         "distinct_queries": 12, "hypothesis_num_states": 5},
        {"noise_rate": 0.3, "strategy": "none", "success": False, "raw_queries": 11,
         "distinct_queries": 11, "hypothesis_num_states": None},
        {"noise_rate": 0.3, "strategy": "adaptive", "success": True, "raw_queries": 550,
         "distinct_queries": 11, "hypothesis_num_states": 5},
        {"noise_rate": 0.3, "strategy": "adaptive", "success": True, "raw_queries": 500,
         "distinct_queries": 10, "hypothesis_num_states": 5},
    ]


def test_success_rate_table():
    table = success_rate_table(fake_results())
    assert table[(0.0, "none")] == 1.0
    assert table[(0.3, "none")] == 0.0
    assert table[(0.3, "adaptive")] == 1.0


def test_mean_field_table_skips_none():
    table = mean_field_table(fake_results(), "hypothesis_num_states")
    assert table[(0.0, "none")] == 5.0
    # (0.3, "none") has a single row with hypothesis_num_states=None -> mean of empty list -> nan
    import math
    assert math.isnan(table[(0.3, "none")])


def test_repetition_factor_table():
    table = repetition_factor_table(fake_results())
    assert abs(table[(0.0, "none")] - 1.0) < 1e-9  # 10/10 and 12/12
    expected_adaptive = ((550 / 11) + (500 / 10)) / 2
    assert abs(table[(0.3, "adaptive")] - expected_adaptive) < 1e-9


def test_union_bound_prediction_is_lower_bound_shape():
    results = fake_results()
    pred = union_bound_success_prediction(results, strategy="adaptive", delta_q=1e-4)
    # mean distinct_queries for adaptive at noise 0.3 is (11+10)/2 = 10.5
    expected = 1.0 - 10.5 * 1e-4
    assert abs(pred[(0.3,)] - expected) < 1e-9
    assert 0.0 <= pred[(0.3,)] <= 1.0


def test_sorted_noise_rates():
    results = fake_results()
    assert sorted_noise_rates(results) == [0.0, 0.3]
