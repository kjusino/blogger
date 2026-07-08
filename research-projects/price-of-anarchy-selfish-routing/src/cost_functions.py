"""Polynomial edge-latency functions with nonnegative coefficients.

A latency function is l(x) = sum_{k=0}^{d} coeffs[k] * x^k, coeffs[k] >= 0.
This is exactly the cost-function class for which Roughgarden's topology
independent price-of-anarchy bound (see theory.py) is stated: polynomials
of degree at most d with nonnegative coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Polynomial:
    """l(x) = sum_k coeffs[k] * x^k, coeffs[k] >= 0. degree = len(coeffs) - 1."""

    coeffs: tuple

    def __post_init__(self) -> None:
        if len(self.coeffs) == 0:
            raise ValueError("coeffs must be non-empty")
        if any(c < 0 for c in self.coeffs):
            raise ValueError("all coefficients must be nonnegative")

    @property
    def degree(self) -> int:
        return len(self.coeffs) - 1

    def __call__(self, x: float) -> float:
        """l(x)."""
        x = np.asarray(x, dtype=float)
        return sum(c * x**k for k, c in enumerate(self.coeffs))

    def grad(self, x: float) -> float:
        """l'(x), needed for gradients of x * l(x)."""
        x = np.asarray(x, dtype=float)
        return sum(k * c * x ** (k - 1) for k, c in enumerate(self.coeffs) if k >= 1)

    def integral(self, x: float) -> float:
        """int_0^x l(t) dt = sum_k coeffs[k] * x^(k+1) / (k+1)."""
        x = np.asarray(x, dtype=float)
        return sum(c * x ** (k + 1) / (k + 1) for k, c in enumerate(self.coeffs))

    def flow_cost(self, x: float) -> float:
        """x * l(x), the total latency contributed by flow x on this edge."""
        return x * self(x)

    def flow_cost_grad(self, x: float) -> float:
        """d/dx [x * l(x)] = l(x) + x * l'(x)."""
        return self(x) + x * self.grad(x)


def affine(a: float, b: float) -> Polynomial:
    """l(x) = a*x + b."""
    return Polynomial((b, a))


def constant(b: float) -> Polynomial:
    """l(x) = b."""
    return Polynomial((b,))


def monomial(a: float, degree: int) -> Polynomial:
    """l(x) = a * x^degree."""
    return Polynomial(tuple([0.0] * degree + [a]))


def random_polynomial(degree: int, rng: np.random.Generator, scale: float = 1.0,
                       min_top_coeff: float = 0.05) -> Polynomial:
    """A random nonneg-coefficient polynomial of exactly `degree`, coefficients
    drawn independently, each optionally zeroed out except the top one (kept
    strictly positive so the edge genuinely belongs to the degree-`degree`
    class rather than a lower one)."""
    coeffs = [rng.uniform(0, scale) if rng.random() < 0.7 else 0.0 for _ in range(degree)]
    coeffs.append(max(rng.uniform(0, scale), min_top_coeff))
    return Polynomial(tuple(coeffs))
