"""Minimal statevector simulator.

A state of `n` qubits is represented as a complex ndarray of shape
`(2,) * n` (rank-n tensor, axis `q` is qubit `q`). This avoids ever
materializing the full `2**n x 2**n` unitary: gates are applied as local
tensor contractions, which is the standard trick for classically
simulating circuits up to a few tens of qubits.
"""

import numpy as np


def zero_state(n: int) -> np.ndarray:
    state = np.zeros((2,) * n, dtype=complex)
    state[(0,) * n] = 1.0
    return state


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


def apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    """Apply a 2x2 unitary to `qubit`, returning a new state tensor of the same shape."""
    out = np.tensordot(gate, state, axes=([1], [qubit]))
    return np.moveaxis(out, 0, qubit)


def apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    """Apply CNOT(control -> target)."""
    n = state.ndim
    other_axes = [a for a in range(n) if a not in (control, target)]
    order = [control, target] + other_axes
    moved = np.transpose(state, order)
    rest_shape = moved.shape[2:]
    mat = moved.reshape(2, 2, -1)
    out = mat.copy()
    out[1, 0, :] = mat[1, 1, :]
    out[1, 1, :] = mat[1, 0, :]
    out = out.reshape((2, 2) + rest_shape)
    inv_order = np.argsort(order)
    return np.transpose(out, inv_order)


def expectation_z(state: np.ndarray, qubit: int) -> float:
    idx0 = [slice(None)] * state.ndim
    idx0[qubit] = 0
    idx1 = [slice(None)] * state.ndim
    idx1[qubit] = 1
    p0 = float(np.sum(np.abs(state[tuple(idx0)]) ** 2))
    p1 = float(np.sum(np.abs(state[tuple(idx1)]) ** 2))
    return p0 - p1


def prob_all_zero(state: np.ndarray) -> float:
    amp = state[(0,) * state.ndim]
    return float(np.abs(amp) ** 2)


def norm(state: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.abs(state) ** 2)))
