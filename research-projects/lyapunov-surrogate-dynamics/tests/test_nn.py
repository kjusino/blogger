import numpy as np

from src.nn import MLP


def test_backprop_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    net = MLP([2, 6, 2], activation="tanh", seed=2)
    X = rng.normal(size=(5, 2))
    Y = rng.normal(size=(5, 2))

    grad_W, grad_b = net._backward(X, Y)
    num_grad_W, num_grad_b = net.numeric_gradient(X, Y, eps=1e-5)

    for gw, ngw in zip(grad_W, num_grad_W):
        assert np.max(np.abs(gw - ngw)) < 1e-6
    for gb, ngb in zip(grad_b, num_grad_b):
        assert np.max(np.abs(gb - ngb)) < 1e-6


def test_backprop_gradient_matches_finite_difference_relu():
    # Also check with the relu activation (piecewise-linear, so gradients
    # can be checked away from the kink by using inputs unlikely to land
    # exactly on zero pre-activations).
    rng = np.random.default_rng(1)
    net = MLP([3, 5, 2], activation="relu", seed=3)
    X = rng.normal(size=(6, 3)) * 2.0
    Y = rng.normal(size=(6, 2))

    grad_W, grad_b = net._backward(X, Y)
    num_grad_W, num_grad_b = net.numeric_gradient(X, Y, eps=1e-5)

    for gw, ngw in zip(grad_W, num_grad_W):
        assert np.max(np.abs(gw - ngw)) < 1e-5
    for gb, ngb in zip(grad_b, num_grad_b):
        assert np.max(np.abs(gb - ngb)) < 1e-5


def test_analytic_jacobian_matches_finite_difference():
    net = MLP([3, 8, 5, 3], activation="tanh", seed=1)
    x = np.array([0.5, -0.3, 1.2])
    J_analytic = net.jacobian(x)
    J_fd = net.jacobian_fd(x, eps=1e-6)
    assert np.max(np.abs(J_analytic - J_fd)) < 1e-6


def test_adam_training_fits_tiny_synthetic_regression():
    # A trivial synthetic target the network should be able to fit nearly
    # exactly: y = 2*x0 - x1 (a linear function, well within an MLP's
    # capacity), used purely to confirm the Adam optimizer loop actually
    # drives the loss down over training.
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, size=(200, 2))
    Y = np.stack([2 * X[:, 0] - X[:, 1], X[:, 0] + X[:, 1]], axis=1)

    net = MLP([2, 16, 2], activation="tanh", seed=0)
    initial_loss = net.loss(X, Y)
    history = net.train(X, Y, epochs=300, batch_size=32, lr=5e-3, seed=0)
    final_loss = net.loss(X, Y)

    assert final_loss < initial_loss
    assert final_loss < 1e-3
    assert history[-1] == final_loss
