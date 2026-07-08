import numpy as np
import pytest

from src.cost_functions import Polynomial, affine, constant, monomial, random_polynomial


def test_rejects_negative_coefficient():
    with pytest.raises(ValueError):
        Polynomial((1.0, -0.5))


def test_rejects_empty_coefficients():
    with pytest.raises(ValueError):
        Polynomial(())


def test_degree():
    assert Polynomial((1.0,)).degree == 0
    assert Polynomial((1.0, 2.0, 3.0)).degree == 2


def test_evaluation():
    p = affine(a=2.0, b=1.0)  # l(x) = 2x + 1
    assert p(0.0) == pytest.approx(1.0)
    assert p(3.0) == pytest.approx(7.0)
    assert constant(5.0)(100.0) == pytest.approx(5.0)
    assert monomial(2.0, 3)(2.0) == pytest.approx(2.0 * 8.0)


def test_integral_matches_numeric_quadrature():
    p = Polynomial((0.3, 0.7, 1.1, 0.2))  # 0.3 + 0.7x + 1.1x^2 + 0.2x^3
    for x in [0.0, 0.5, 1.0, 2.3]:
        xs = np.linspace(0, x, 200_001)
        numeric = np.trapezoid(p(xs), xs) if x > 0 else 0.0
        assert p.integral(x) == pytest.approx(numeric, abs=1e-4)


def test_integral_of_zero_is_zero():
    p = Polynomial((0.0, 0.0, 0.0))
    assert p.integral(5.0) == pytest.approx(0.0)


def test_grad_matches_finite_difference():
    p = Polynomial((0.4, 1.3, 0.9, 0.6))
    for x in [0.1, 1.0, 2.5]:
        h = 1e-6
        fd = (p(x + h) - p(x - h)) / (2 * h)
        assert p.grad(x) == pytest.approx(fd, rel=1e-4)


def test_flow_cost_and_its_gradient():
    p = Polynomial((0.2, 0.0, 1.5))  # 0.2 + 1.5 x^2
    x = 1.7
    assert p.flow_cost(x) == pytest.approx(x * p(x))
    h = 1e-6
    fd = (p.flow_cost(x + h) - p.flow_cost(x - h)) / (2 * h)
    assert p.flow_cost_grad(x) == pytest.approx(fd, rel=1e-4)


def test_random_polynomial_has_correct_degree_and_nonneg_coeffs():
    rng = np.random.default_rng(0)
    for degree in [1, 2, 3, 4]:
        for _ in range(20):
            p = random_polynomial(degree, rng, scale=2.0)
            assert p.degree == degree
            assert p.coeffs[-1] > 0.0
            assert all(c >= 0 for c in p.coeffs)
