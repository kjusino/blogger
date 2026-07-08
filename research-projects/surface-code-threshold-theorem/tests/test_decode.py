import pytest

from src.circuits import build_surface_code_circuit
from src.decode import sample_and_decode, wilson_interval


def test_wilson_interval_zero_successes_is_nonnegative_and_bounded():
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0
    assert 0 < hi < 0.1


def test_wilson_interval_all_successes():
    lo, hi = wilson_interval(100, 100)
    assert hi == pytest.approx(1.0)
    assert 0.9 < lo < 1.0


def test_wilson_interval_matches_hand_computed_value():
    # successes=50, n=100 -> Wilson center should be exactly 0.5 (phat=0.5
    # is a fixed point of the recentering), matching the textbook formula.
    lo, hi = wilson_interval(50, 100, z=1.96)
    assert abs((lo + hi) / 2 - 0.5) < 1e-9
    assert lo < 0.5 < hi


def test_wilson_interval_empty_sample():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_zero_noise_circuit_has_zero_logical_errors():
    circuit = build_surface_code_circuit(distance=3, rounds=3, p=0.0)
    result = sample_and_decode(circuit, shots=500, seed=1, distance=3, p=0.0)
    assert result.logical_errors == 0
    assert result.logical_error_rate == 0.0


def test_decode_result_is_deterministic_given_seed():
    circuit = build_surface_code_circuit(distance=3, rounds=3, p=0.01)
    r1 = sample_and_decode(circuit, shots=1000, seed=42, distance=3, p=0.01)
    r2 = sample_and_decode(circuit, shots=1000, seed=42, distance=3, p=0.01)
    assert r1.logical_errors == r2.logical_errors


def test_higher_noise_gives_higher_logical_error_rate():
    low_p_circuit = build_surface_code_circuit(distance=3, rounds=3, p=0.001)
    high_p_circuit = build_surface_code_circuit(distance=3, rounds=3, p=0.05)
    low = sample_and_decode(low_p_circuit, shots=5000, seed=7, distance=3, p=0.001)
    high = sample_and_decode(high_p_circuit, shots=5000, seed=7, distance=3, p=0.05)
    assert high.logical_error_rate > low.logical_error_rate


def test_confidence_interval_contains_observed_rate():
    circuit = build_surface_code_circuit(distance=3, rounds=3, p=0.01)
    result = sample_and_decode(circuit, shots=5000, seed=3, distance=3, p=0.01)
    lo, hi = result.confidence_interval
    assert lo <= result.logical_error_rate <= hi
