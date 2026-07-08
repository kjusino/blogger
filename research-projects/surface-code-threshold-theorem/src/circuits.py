"""Stim circuit construction for the rotated surface code under uniform
circuit-level depolarizing noise (the noise model used in the original
surface-code threshold estimates, e.g. Fowler et al. 2012)."""

import stim


def build_surface_code_circuit(distance: int, rounds: int, p: float) -> stim.Circuit:
    """Build a rotated-surface-code Z-memory experiment.

    A single noise strength `p` is applied uniformly to every noise
    channel stim exposes for this task: two-qubit-gate depolarization,
    reset error, measurement error, and idling data-qubit depolarization
    between rounds. This is the standard "uniform circuit noise" model
    used to define the surface code's fault-tolerance threshold.
    """
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be an odd integer >= 3")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be a probability in [0, 1]")

    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )
