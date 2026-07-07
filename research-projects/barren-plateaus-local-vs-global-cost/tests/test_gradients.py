import numpy as np
import pytest

from barren_plateaus import circuits, gradients


def test_parameter_shift_matches_finite_difference():
    rng = np.random.default_rng(5)
    for cost_type, cost_fn in circuits.COST_FUNCTIONS.items():
        for _ in range(5):
            n = rng.integers(2, 5)
            depth = rng.integers(1, 3)
            params = circuits.random_params(n, depth, rng)
            layer = int(rng.integers(0, depth))
            qubit = int(rng.integers(0, n))
            gate_idx = int(rng.integers(0, 2))

            analytic = gradients.parameter_shift_gradient(
                n, depth, params, cost_fn, layer=layer, qubit=qubit, gate_idx=gate_idx
            )
            numeric = gradients.finite_difference_gradient(
                n, depth, params, cost_fn, layer=layer, qubit=qubit, gate_idx=gate_idx
            )
            assert analytic == pytest.approx(numeric, abs=1e-4), (cost_type, n, depth, layer, qubit, gate_idx)


def test_gradient_of_single_qubit_local_cost_analytic():
    # n=1, depth=1, cost = (1 - <Z>) / 2 with only an RY gate on the parameter of interest.
    # <Z(theta)> = cos(theta) so C(theta) = (1 - cos(theta))/2, dC/dtheta = sin(theta)/2.
    n, depth = 1, 1
    for theta in [0.1, 0.9, 2.4, 4.0]:
        params = np.zeros((depth, n, 2))
        params[0, 0, 0] = theta
        grad = gradients.parameter_shift_gradient(n, depth, params, circuits.cost_local, layer=0, qubit=0, gate_idx=0)
        assert grad == pytest.approx(np.sin(theta) / 2, abs=1e-8)


def test_gradient_zero_at_symmetric_point_for_local_cost():
    # At theta=0 (RY angle), the local-cost gradient wrt that same angle is 0
    # since C(theta) = (1-cos theta)/2 is even around theta=0 -> derivative 0.
    n, depth = 1, 1
    params = np.zeros((depth, n, 2))
    grad = gradients.parameter_shift_gradient(n, depth, params, circuits.cost_local, layer=0, qubit=0, gate_idx=0)
    assert grad == pytest.approx(0.0, abs=1e-10)
