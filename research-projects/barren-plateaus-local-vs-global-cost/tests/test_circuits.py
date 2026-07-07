import numpy as np
import pytest

from barren_plateaus import circuits, statevector as sv


def test_run_circuit_preserves_norm():
    rng = np.random.default_rng(2)
    for n in [1, 2, 3, 5]:
        for depth in [1, 3]:
            params = circuits.random_params(n, depth, rng)
            state = circuits.run_circuit(n, depth, params)
            assert sv.norm(state) == pytest.approx(1.0, abs=1e-8)


def test_run_circuit_rejects_wrong_param_shape():
    with pytest.raises(ValueError):
        circuits.run_circuit(3, 2, np.zeros((2, 4, 2)))  # wrong qubit count


def test_zero_depth_zero_params_is_zero_state():
    n = 3
    params = np.zeros((1, n, 2))
    state = circuits.run_circuit(n, depth=1, params=params)
    assert sv.prob_all_zero(state) == pytest.approx(1.0)


def test_cost_global_is_zero_at_identity():
    n = 4
    params = np.zeros((2, n, 2))
    state = circuits.run_circuit(n, depth=2, params=params)
    assert circuits.cost_global(state) == pytest.approx(0.0, abs=1e-10)


def test_cost_local_is_zero_at_identity():
    n = 4
    params = np.zeros((2, n, 2))
    state = circuits.run_circuit(n, depth=2, params=params)
    assert circuits.cost_local(state) == pytest.approx(0.0, abs=1e-10)


def test_cost_local_at_pi_flip():
    # n=1 so there is no entangling gate to muddy the picture: a bare
    # RY(pi) on the single qubit flips |0> -> |1>.
    n = 1
    params = np.zeros((1, n, 2))
    params[0, 0, 0] = np.pi
    state = circuits.run_circuit(n, depth=1, params=params)
    assert circuits.cost_local(state) == pytest.approx(1.0, abs=1e-10)
    assert circuits.cost_global(state) == pytest.approx(1.0, abs=1e-10)


def test_cost_local_averages_over_all_qubits():
    # Flip only the *last* qubit to |1> (depth=1, no other rotation); with a
    # linear entangler CNOT(0,1),...,CNOT(n-2,n-1), the last qubit is never a
    # control, so this flip does not cascade anywhere, leaving qubits 0..n-2
    # at |0> and only the last at |1> -> average cost 1/n.
    n = 4
    params = np.zeros((1, n, 2))
    params[0, n - 1, 0] = np.pi
    state = circuits.run_circuit(n, depth=1, params=params)
    assert circuits.cost_local(state) == pytest.approx(1.0 / n, abs=1e-10)


def test_cost_local_cascades_through_linear_chain():
    # Flipping qubit 0 to |1> instead *does* cascade: CNOT(0,1) flips qubit 1
    # (control=1), which then flips qubit 2 via CNOT(1,2), and so on, so
    # every qubit ends at |1> and the average cost is 1.
    n = 4
    params = np.zeros((1, n, 2))
    params[0, 0, 0] = np.pi
    state = circuits.run_circuit(n, depth=1, params=params)
    assert circuits.cost_local(state) == pytest.approx(1.0, abs=1e-10)


def test_linear_entangler_has_no_ring_degeneracy():
    # Regression test: a *ring* entangler (wrapping CNOT(n-1,0) back to qubit
    # 0) makes <Z_0> after one layer provably independent of qubit 0's own
    # rotation (an exact CNOT-conjugation cancellation). The linear chain
    # must not reproduce that: perturbing a qubit's own first-layer rotation
    # should move the local cost.
    rng = np.random.default_rng(11)
    n, depth = 4, 1
    params = circuits.random_params(n, depth, rng)
    base = circuits.cost_local(circuits.run_circuit(n, depth, params))
    perturbed = params.copy()
    perturbed[0, n // 2, 0] += np.pi / 2
    moved = circuits.cost_local(circuits.run_circuit(n, depth, perturbed))
    assert abs(moved - base) > 1e-6


def test_costs_bounded_in_unit_interval():
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = rng.integers(1, 5)
        depth = rng.integers(1, 4)
        params = circuits.random_params(n, depth, rng)
        state = circuits.run_circuit(n, depth, params)
        assert -1e-9 <= circuits.cost_global(state) <= 1 + 1e-9
        assert -1e-9 <= circuits.cost_local(state) <= 1 + 1e-9


def test_random_params_shape_and_range():
    rng = np.random.default_rng(4)
    params = circuits.random_params(5, 3, rng)
    assert params.shape == (3, 5, 2)
    assert np.all(params >= 0.0) and np.all(params < 2 * np.pi)
