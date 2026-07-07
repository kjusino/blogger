import numpy as np

from src.data import make_dataset
from src.model import loss, per_example_gradients, sigmoid


def test_sigmoid_basic_values():
    assert np.isclose(sigmoid(np.array([0.0]))[0], 0.5)
    assert sigmoid(np.array([100.0]))[0] > 0.999
    assert sigmoid(np.array([-100.0]))[0] < 0.001


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(123)
    X, y = make_dataset(n=20, d=4, rng=rng)
    theta = rng.normal(size=4)

    grads = per_example_gradients(X, y, theta)
    # Mean per-example gradient should match the finite-difference gradient
    # of the mean loss.
    analytic_mean_grad = grads.mean(axis=0)

    eps = 1e-6
    numeric_grad = np.zeros(4)
    for j in range(4):
        theta_plus = theta.copy()
        theta_plus[j] += eps
        theta_minus = theta.copy()
        theta_minus[j] -= eps
        numeric_grad[j] = (loss(X, y, theta_plus) - loss(X, y, theta_minus)) / (2 * eps)

    np.testing.assert_allclose(analytic_mean_grad, numeric_grad, atol=1e-4)


def test_per_example_gradient_shape():
    rng = np.random.default_rng(0)
    X, y = make_dataset(n=10, d=3, rng=rng)
    theta = np.zeros(3)
    grads = per_example_gradients(X, y, theta)
    assert grads.shape == (10, 3)
