import numpy as np
import pytest

from barren_plateaus import statevector as sv


def test_zero_state_normalized():
    state = sv.zero_state(3)
    assert sv.norm(state) == pytest.approx(1.0)
    assert sv.prob_all_zero(state) == pytest.approx(1.0)


def test_ry_pi_flips_to_one():
    # RY(pi)|0> = |1> (up to global phase convention used here: exact |1>)
    state = sv.zero_state(1)
    state = sv.apply_single_qubit_gate(state, sv.ry(np.pi), 0)
    assert abs(state[1]) == pytest.approx(1.0)
    assert abs(state[0]) == pytest.approx(0.0, abs=1e-12)


def test_ry_pi_half_is_uniform_superposition():
    state = sv.zero_state(1)
    state = sv.apply_single_qubit_gate(state, sv.ry(np.pi / 2), 0)
    probs = np.abs(state) ** 2
    assert probs[0] == pytest.approx(0.5)
    assert probs[1] == pytest.approx(0.5)


def test_rz_preserves_computational_basis_probabilities():
    state = sv.zero_state(2)
    state = sv.apply_single_qubit_gate(state, sv.ry(0.7), 0)
    state = sv.apply_single_qubit_gate(state, sv.ry(1.3), 1)
    probs_before = np.abs(state) ** 2
    state = sv.apply_single_qubit_gate(state, sv.rz(0.4), 0)
    probs_after = np.abs(state) ** 2
    np.testing.assert_allclose(probs_before, probs_after, atol=1e-12)


def test_gates_are_unitary():
    for theta in [0.0, 0.3, 1.7, np.pi, 4.2]:
        for gate in (sv.ry(theta), sv.rz(theta)):
            should_be_identity = gate.conj().T @ gate
            np.testing.assert_allclose(should_be_identity, np.eye(2), atol=1e-10)


def test_single_qubit_gate_preserves_norm():
    rng = np.random.default_rng(0)
    state = sv.zero_state(4)
    for q in range(4):
        state = sv.apply_single_qubit_gate(state, sv.ry(rng.uniform(0, 2 * np.pi)), q)
        state = sv.apply_single_qubit_gate(state, sv.rz(rng.uniform(0, 2 * np.pi)), q)
    assert sv.norm(state) == pytest.approx(1.0)


def test_cnot_truth_table():
    # |10> -(CNOT 0->1)-> |11>
    state = sv.zero_state(2)
    state = sv.apply_single_qubit_gate(state, sv.ry(np.pi), 0)  # qubit 0 -> |1>
    assert abs(state[1, 0]) == pytest.approx(1.0)
    state = sv.apply_cnot(state, control=0, target=1)
    assert abs(state[1, 1]) == pytest.approx(1.0)
    assert abs(state[1, 0]) == pytest.approx(0.0, abs=1e-12)


def test_cnot_identity_when_control_zero():
    state = sv.zero_state(2)
    state = sv.apply_single_qubit_gate(state, sv.ry(0.6), 1)
    before = state.copy()
    after = sv.apply_cnot(state, control=0, target=1)
    np.testing.assert_allclose(before, after, atol=1e-12)


def test_cnot_preserves_norm_on_random_state():
    rng = np.random.default_rng(1)
    state = sv.zero_state(5)
    for q in range(5):
        state = sv.apply_single_qubit_gate(state, sv.ry(rng.uniform(0, 2 * np.pi)), q)
    state = sv.apply_cnot(state, control=1, target=3)
    assert sv.norm(state) == pytest.approx(1.0)


def test_bell_state_via_cnot():
    state = sv.zero_state(2)
    state = sv.apply_single_qubit_gate(state, sv.ry(np.pi / 2), 0)
    state = sv.apply_cnot(state, control=0, target=1)
    probs = np.abs(state) ** 2
    assert probs[0, 0] == pytest.approx(0.5, abs=1e-10)
    assert probs[1, 1] == pytest.approx(0.5, abs=1e-10)
    assert probs[0, 1] == pytest.approx(0.0, abs=1e-10)
    assert probs[1, 0] == pytest.approx(0.0, abs=1e-10)


def test_expectation_z_on_basis_states():
    state0 = sv.zero_state(1)
    assert sv.expectation_z(state0, 0) == pytest.approx(1.0)
    state1 = sv.apply_single_qubit_gate(state0, sv.ry(np.pi), 0)
    assert sv.expectation_z(state1, 0) == pytest.approx(-1.0)
