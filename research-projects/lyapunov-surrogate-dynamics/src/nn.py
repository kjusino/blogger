"""A small feed-forward MLP implemented from scratch in plain NumPy:
manual forward pass, manual backprop, and an Adam optimizer. No autodiff
library (no torch/jax) -- this keeps the research-project subproject
dependency-light and fully self-contained.

Architecture: configurable hidden widths, tanh or relu activation on hidden
layers, a linear (no activation) output layer (standard for regression of a
continuous flow map). Trained with mini-batch Adam on MSE loss.

Also provides an analytic Jacobian d(output)/d(input) via the chain rule
through the cached forward-pass activations -- used by lyapunov.py to
estimate the Lyapunov spectrum of the trained surrogate treated as a
discrete dynamical system.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def _tanh(z):
    return np.tanh(z)


def _tanh_prime(z):
    t = np.tanh(z)
    return 1.0 - t * t


def _relu(z):
    return np.maximum(0.0, z)


def _relu_prime(z):
    return (z > 0.0).astype(z.dtype)


_ACTIVATIONS = {
    "tanh": (_tanh, _tanh_prime),
    "relu": (_relu, _relu_prime),
}


class MLP:
    """A multilayer perceptron with a linear output layer.

    layer_sizes: e.g. [3, 32, 3] = input dim 3, one hidden layer of width
    32, output dim 3. Any number of hidden layers is supported.
    """

    def __init__(self, layer_sizes: List[int], activation: str = "tanh",
                 seed: int = 0):
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}")
        self.layer_sizes = list(layer_sizes)
        self.activation_name = activation
        self.act, self.act_prime = _ACTIVATIONS[activation]
        self.n_layers = len(layer_sizes) - 1  # number of weight matrices

        rng = np.random.default_rng(seed)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(self.n_layers):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            # Xavier/Glorot-style init, scaled a bit for tanh/relu stability.
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.W.append(rng.normal(0.0, scale, size=(fan_out, fan_in)))
            self.b.append(np.zeros(fan_out))

        # Adam moment buffers, lazily matched to W/b shapes.
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(bb) for bb in self.b]
        self._vb = [np.zeros_like(bb) for bb in self.b]
        self._adam_t = 0

    # ---- forward / backward on a batch, X shape (N, in_dim) ----

    def _forward_cache(self, X: np.ndarray):
        """Forward pass on a batch, caching z's and a's for backprop.

        Returns (a_list, z_list) where a_list[0] = X (input) and
        a_list[-1] = network output.
        """
        a = X
        a_list = [a]
        z_list = [None]  # placeholder so indices line up with a_list
        for i in range(self.n_layers):
            z = a @ self.W[i].T + self.b[i]
            is_last = (i == self.n_layers - 1)
            a = z if is_last else self.act(z)
            z_list.append(z)
            a_list.append(a)
        return a_list, z_list

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Batched forward pass. X: (N, in_dim) or (in_dim,). Returns
        matching-shape output."""
        single = (X.ndim == 1)
        X2 = X.reshape(1, -1) if single else X
        a_list, _ = self._forward_cache(X2)
        out = a_list[-1]
        return out[0] if single else out

    __call__ = forward

    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        pred = self.forward(X)
        return float(np.mean((pred - Y) ** 2))

    def _backward(self, X: np.ndarray, Y: np.ndarray):
        """Compute gradients of MSE loss w.r.t. all weights/biases via
        manual backprop. Returns (grad_W list, grad_b list)."""
        n = X.shape[0]
        a_list, z_list = self._forward_cache(X)
        pred = a_list[-1]
        # dL/d(output), MSE = mean over batch and output dims of (pred-y)^2
        delta = (2.0 / (n * Y.shape[1])) * (pred - Y)  # (N, out_dim)

        grad_W = [None] * self.n_layers
        grad_b = [None] * self.n_layers

        for i in reversed(range(self.n_layers)):
            a_prev = a_list[i]  # input to this layer, (N, fan_in)
            grad_W[i] = delta.T @ a_prev  # (fan_out, fan_in)
            grad_b[i] = delta.sum(axis=0)  # (fan_out,)
            if i > 0:
                # propagate delta to previous layer's activation output
                delta_prev_pre_act = delta @ self.W[i]  # (N, fan_in==prev fan_out)
                delta = delta_prev_pre_act * self.act_prime(z_list[i])
        return grad_W, grad_b

    def numeric_gradient(self, X: np.ndarray, Y: np.ndarray, eps: float = 1e-5):
        """Slow, exact finite-difference gradient check helper (used only in
        tests, not in training) -- perturbs every weight/bias entry."""
        grad_W = [np.zeros_like(w) for w in self.W]
        grad_b = [np.zeros_like(bb) for bb in self.b]
        base = self.loss(X, Y)
        for i in range(self.n_layers):
            it = np.nditer(self.W[i], flags=["multi_index"])
            for _ in it:
                idx = it.multi_index
                orig = self.W[i][idx]
                self.W[i][idx] = orig + eps
                lp = self.loss(X, Y)
                self.W[i][idx] = orig - eps
                lm = self.loss(X, Y)
                self.W[i][idx] = orig
                grad_W[i][idx] = (lp - lm) / (2 * eps)
            for j in range(self.b[i].shape[0]):
                orig = self.b[i][j]
                self.b[i][j] = orig + eps
                lp = self.loss(X, Y)
                self.b[i][j] = orig - eps
                lm = self.loss(X, Y)
                self.b[i][j] = orig
                grad_b[i][j] = (lp - lm) / (2 * eps)
        _ = base
        return grad_W, grad_b

    def _adam_step(self, grad_W, grad_b, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self._adam_t += 1
        t = self._adam_t
        for i in range(self.n_layers):
            self._mW[i] = beta1 * self._mW[i] + (1 - beta1) * grad_W[i]
            self._vW[i] = beta2 * self._vW[i] + (1 - beta2) * (grad_W[i] ** 2)
            mhat = self._mW[i] / (1 - beta1 ** t)
            vhat = self._vW[i] / (1 - beta2 ** t)
            self.W[i] -= lr * mhat / (np.sqrt(vhat) + eps)

            self._mb[i] = beta1 * self._mb[i] + (1 - beta1) * grad_b[i]
            self._vb[i] = beta2 * self._vb[i] + (1 - beta2) * (grad_b[i] ** 2)
            mhat_b = self._mb[i] / (1 - beta1 ** t)
            vhat_b = self._vb[i] / (1 - beta2 ** t)
            self.b[i] -= lr * mhat_b / (np.sqrt(vhat_b) + eps)

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int,
              batch_size: int = 32, lr: float = 1e-3,
              seed: int = 0, verbose: bool = False) -> List[float]:
        """Mini-batch Adam training. Returns the per-epoch loss history."""
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        history = []
        for epoch in range(epochs):
            perm = rng.permutation(n)
            Xs, Ys = X[perm], Y[perm]
            for start in range(0, n, batch_size):
                xb = Xs[start:start + batch_size]
                yb = Ys[start:start + batch_size]
                grad_W, grad_b = self._backward(xb, yb)
                self._adam_step(grad_W, grad_b, lr)
            epoch_loss = self.loss(X, Y)
            history.append(epoch_loss)
            if verbose and (epoch % max(1, epochs // 10) == 0):
                print(f"epoch {epoch:4d}  loss {epoch_loss:.6e}")
        return history

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Analytic Jacobian d(output)/d(input) at a single point x, via
        the chain rule through cached forward-pass pre-activations. Exact
        (not finite-difference), and cheap: O(n_layers) matrix products.
        """
        x = np.asarray(x, dtype=float)
        a_list, z_list = self._forward_cache(x.reshape(1, -1))
        # J starts as the last (linear) layer's weight matrix.
        J = self.W[-1]
        for i in range(self.n_layers - 2, -1, -1):
            z_next = z_list[i + 1][0]  # pre-activation of layer i+1 (the one after i)
            d = self.act_prime(z_next)
            J = (J * d[None, :]) @ self.W[i]
        return J

    def jacobian_fd(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Finite-difference Jacobian fallback (used for cross-checking
        the analytic `jacobian` method in tests)."""
        x = np.asarray(x, dtype=float)
        dim = x.shape[0]
        out_dim = self.layer_sizes[-1]
        J = np.zeros((out_dim, dim))
        for j in range(dim):
            step = np.zeros(dim)
            step[j] = eps
            fp = self.forward(x + step)
            fm = self.forward(x - step)
            J[:, j] = (fp - fm) / (2 * eps)
        return J
