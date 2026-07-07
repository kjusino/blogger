"""Sweep gradient variance over (cost type, depth, number of qubits)."""

from dataclasses import dataclass, field

import numpy as np

from . import circuits, gradients


@dataclass
class VarianceEstimate:
    cost_type: str
    depth: int
    n: int
    num_samples: int
    mean: float
    variance: float
    variance_stderr: float
    samples: np.ndarray = field(repr=False)


def estimate_gradient_variance(
    cost_type: str, n: int, depth: int, num_samples: int, seed: int
) -> VarianceEstimate:
    if cost_type not in circuits.COST_FUNCTIONS:
        raise ValueError(f"unknown cost_type {cost_type!r}")
    cost_fn = circuits.COST_FUNCTIONS[cost_type]
    rng = np.random.default_rng(seed)

    # Probe the gradient at a middle qubit of the first layer. An edge qubit
    # (e.g. qubit 0 with a linear entangler) is a poor probe: it is only ever
    # a CNOT *control* in its own layer, so it is not entangled with the rest
    # of the circuit until a later layer wraps back around, understating how
    # fast the gradient actually concentrates.
    probe_qubit = n // 2
    grads = np.empty(num_samples, dtype=float)
    for i in range(num_samples):
        params = circuits.random_params(n, depth, rng)
        grads[i] = gradients.parameter_shift_gradient(n, depth, params, cost_fn, qubit=probe_qubit)

    mean = float(grads.mean())
    var = float(grads.var(ddof=1))
    # Var[sample variance] ~= 2*sigma^4 / (M-1) under an approx-normal assumption;
    # used only to draw error bars, not for any hypothesis test.
    var_stderr = float(np.sqrt(2.0 / (num_samples - 1)) * var)

    return VarianceEstimate(
        cost_type=cost_type,
        depth=depth,
        n=n,
        num_samples=num_samples,
        mean=mean,
        variance=var,
        variance_stderr=var_stderr,
        samples=grads,
    )


def run_sweep(
    cost_types,
    depths,
    n_values,
    num_samples: int,
    base_seed: int,
):
    """Run estimate_gradient_variance over the full grid; returns a flat list."""
    results = []
    seed = base_seed
    for cost_type in cost_types:
        for depth in depths:
            for n in n_values:
                results.append(
                    estimate_gradient_variance(cost_type, n, depth, num_samples, seed)
                )
                seed += 1
    return results
