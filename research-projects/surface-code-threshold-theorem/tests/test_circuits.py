import pytest
import stim

from src.circuits import build_surface_code_circuit


def test_rejects_even_distance():
    with pytest.raises(ValueError):
        build_surface_code_circuit(distance=4, rounds=4, p=0.01)


def test_rejects_too_small_distance():
    with pytest.raises(ValueError):
        build_surface_code_circuit(distance=1, rounds=1, p=0.01)


def test_rejects_zero_rounds():
    with pytest.raises(ValueError):
        build_surface_code_circuit(distance=3, rounds=0, p=0.01)


def test_rejects_invalid_probability():
    with pytest.raises(ValueError):
        build_surface_code_circuit(distance=3, rounds=3, p=1.5)
    with pytest.raises(ValueError):
        build_surface_code_circuit(distance=3, rounds=3, p=-0.1)


def test_returns_valid_circuit_with_one_logical_observable():
    circuit = build_surface_code_circuit(distance=3, rounds=3, p=0.01)
    assert isinstance(circuit, stim.Circuit)
    assert circuit.num_observables == 1
    assert circuit.num_detectors > 0


def test_larger_distance_has_more_detectors():
    small = build_surface_code_circuit(distance=3, rounds=3, p=0.01)
    large = build_surface_code_circuit(distance=5, rounds=5, p=0.01)
    assert large.num_detectors > small.num_detectors


def test_zero_noise_circuit_never_flips_detectors_or_observable():
    circuit = build_surface_code_circuit(distance=3, rounds=3, p=0.0)
    sampler = circuit.compile_detector_sampler(seed=1)
    detectors, observables = sampler.sample(shots=200, separate_observables=True)
    assert not detectors.any()
    assert not observables.any()
