"""Hardware-efficient random ansatz and the two canonical cost functions
(global vs. local) used in the barren-plateau literature
(McClean et al. 2018; Cerezo et al. 2021, "Cost function dependent barren
plateaus in shallow parametrized quantum circuits").
"""

import numpy as np

from . import statevector as sv

# Each layer applies RY then RZ to every qubit (2 parameters/qubit/layer),
# followed by a ring of CNOTs. This is the standard "hardware-efficient
# ansatz" used to probe trainability landscapes.
PARAMS_PER_QUBIT_PER_LAYER = 2


def random_params(n: int, depth: int, rng: np.random.Generator) -> np.ndarray:
    """Parameters of shape (depth, n, 2), uniform on [0, 2*pi).

    Uniform-random angles per layer are the standard cheap substitute for an
    exact Haar-random 2-design in this literature: they are not literally
    Haar-random, but a handful of such layers rapidly approximates one for
    the purpose of measuring gradient concentration.
    """
    return rng.uniform(0.0, 2.0 * np.pi, size=(depth, n, PARAMS_PER_QUBIT_PER_LAYER))


def run_circuit(n: int, depth: int, params: np.ndarray) -> np.ndarray:
    if params.shape != (depth, n, PARAMS_PER_QUBIT_PER_LAYER):
        raise ValueError(f"expected params shape {(depth, n, PARAMS_PER_QUBIT_PER_LAYER)}, got {params.shape}")
    state = sv.zero_state(n)
    for layer in range(depth):
        for q in range(n):
            state = sv.apply_single_qubit_gate(state, sv.ry(params[layer, q, 0]), q)
            state = sv.apply_single_qubit_gate(state, sv.rz(params[layer, q, 1]), q)
        # Linear nearest-neighbor entangler (no wraparound). A *ring* entangler
        # (wrapping CNOT(n-1, 0) back to qubit 0) creates an exact analytic
        # cancellation at depth=1: CNOT's operator identity
        # CNOT_{c->t}^dagger Z_t CNOT_{c->t} = Z_t Z_c means the closed loop
        # of conjugations returns a Z_0^2 = I factor, making <Z_0> provably
        # *independent* of qubit 0's own rotation angle. The linear chain
        # avoids that degeneracy.
        for q in range(n - 1):
            state = sv.apply_cnot(state, q, q + 1)
    return state


def cost_global(state: np.ndarray) -> float:
    """C_G = 1 - |<0...0|psi>|^2, the fidelity-to-|0> cost.

    Its observable, the projector onto |0...0>, acts on all n qubits at
    once -- the textbook "global" cost function.
    """
    return 1.0 - sv.prob_all_zero(state)


def cost_local(state: np.ndarray) -> float:
    """C_L = 1 - (1/n) * sum_i <Z_i>_+, averaged single-qubit-|1> probability.

    This is the standard "local cost" of the barren-plateau literature
    (Cerezo et al. 2021): a sum of observables that each act on only one
    qubit (identity elsewhere), as opposed to the global cost's single
    observable acting on all n qubits at once.
    """
    n = state.ndim
    z_sum = sum(sv.expectation_z(state, q) for q in range(n))
    return (1.0 - z_sum / n) / 2.0


COST_FUNCTIONS = {"global": cost_global, "local": cost_local}
