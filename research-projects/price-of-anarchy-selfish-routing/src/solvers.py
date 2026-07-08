"""Generic Wardrop-equilibrium and social-optimum solvers for single-commodity
nonatomic routing games on arbitrary DAGs, via edge-flow convex programs (no
path enumeration, so this scales to graphs with many s-t paths).

Wardrop equilibrium = the flow minimizing the convex "potential"
    Phi(f) = sum_e integral_0^{f_e} l_e(x) dx
subject to flow conservation and f >= 0 (Beckmann, McGuire, Winsten 1956).
At the minimizer, all used s-t paths have equal, minimal latency -- the
defining property of a Wardrop equilibrium -- which is exactly the
first-order (KKT) condition of this convex program.

Social optimum = the flow minimizing total cost
    C(f) = sum_e f_e * l_e(f_e)
subject to the same constraints. Both objectives are convex because every
term is (a sum of) x^k with k >= 1 and a nonnegative coefficient.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize

from src.network import Network


def _reduced_conservation(network: Network) -> tuple:
    """Drop one (redundant) conservation row -- the rows of the incidence
    matrix always sum to zero, so keeping all of them makes the equality
    constraints rank-deficient."""
    A = network.incidence_matrix()
    b = network.conservation_rhs()
    return A[:-1], b[:-1]


def _feasible_start(network: Network) -> np.ndarray:
    A_reduced, b_reduced = _reduced_conservation(network)
    n = network.n_edges
    result = linprog(c=np.zeros(n), A_eq=A_reduced, b_eq=b_reduced,
                      bounds=[(0, None)] * n, method="highs")
    if not result.success:
        raise RuntimeError(f"no feasible flow found: {result.message}")
    return result.x


def _solve_convex_program(network: Network, value_fn, grad_fn) -> np.ndarray:
    A_reduced, b_reduced = _reduced_conservation(network)
    n = network.n_edges
    x0 = _feasible_start(network)

    constraints = [{
        "type": "eq",
        "fun": lambda f: A_reduced @ f - b_reduced,
        "jac": lambda f: A_reduced,
    }]
    bounds = [(0.0, None)] * n

    result = minimize(value_fn, x0, jac=grad_fn, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-14})
    if not result.success:
        raise RuntimeError(f"convex program did not converge: {result.message}")
    return np.clip(result.x, 0.0, None)


def solve_equilibrium(network: Network) -> np.ndarray:
    """The Wardrop-equilibrium edge-flow vector."""

    def value(f):
        return sum(e.cost.integral(f[i]) for i, e in enumerate(network.edges))

    def grad(f):
        return np.array([e.cost(f[i]) for i, e in enumerate(network.edges)])

    return _solve_convex_program(network, value, grad)


def solve_optimum(network: Network) -> np.ndarray:
    """The social-optimum edge-flow vector."""

    def value(f):
        return sum(e.cost.flow_cost(f[i]) for i, e in enumerate(network.edges))

    def grad(f):
        return np.array([e.cost.flow_cost_grad(f[i]) for i, e in enumerate(network.edges)])

    return _solve_convex_program(network, value, grad)


def total_cost(network: Network, f: np.ndarray) -> float:
    """C(f) = sum_e f_e * l_e(f_e), the quantity users collectively pay."""
    return float(sum(e.cost.flow_cost(f[i]) for i, e in enumerate(network.edges)))


def price_of_anarchy(network: Network) -> dict:
    """Solve both programs and return PoA plus the flows/costs behind it."""
    f_eq = solve_equilibrium(network)
    f_opt = solve_optimum(network)
    c_eq = total_cost(network, f_eq)
    c_opt = total_cost(network, f_opt)
    poa = 1.0 if c_opt <= 1e-12 else c_eq / c_opt
    return {
        "poa": poa,
        "equilibrium_flow": f_eq,
        "optimum_flow": f_opt,
        "equilibrium_cost": c_eq,
        "optimum_cost": c_opt,
    }
