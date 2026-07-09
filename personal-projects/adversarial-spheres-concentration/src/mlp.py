"""A from-scratch (no autodiff) two-hidden-layer MLP binary classifier.

Forward: x -> leaky_relu(W1 x + b1) -> leaky_relu(W2 h1 + b2) -> logit = w3.h2+b3
-> sigmoid. Backward is hand-derived reverse-mode chain rule (no autodiff
library), trained with a manually implemented Adam optimizer. Gradients are
checked against finite differences in tests/test_mlp.py.

Hidden activations use (leaky) ReLU rather than tanh deliberately: points on the
two spheres are, coordinate-wise, zero-mean random projections that differ
between classes only in *variance* (the radius), not in mean. tanh is an odd
function, so E[tanh(z)] carries almost no signal about the variance of a
zero-mean z near initialization (a vanishing-gradient / high "Hermite rank"
problem), which was empirically observed to stall training as d grew. ReLU's
asymmetry gives E[relu(z)] a direct dependence on std(z), restoring a usable
gradient signal from a random initialization at every dimension tested.
"""

import numpy as np

_LEAK = 0.01


def _sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def _lrelu(z):
    return np.where(z >= 0, z, _LEAK * z)


def _lrelu_grad(z):
    return np.where(z >= 0, 1.0, _LEAK)


class MLP:
    def __init__(self, d_in, d_hidden, rng):
        scale1 = np.sqrt(2.0 / d_in)
        scale2 = np.sqrt(2.0 / d_hidden)
        self.W1 = rng.normal(scale=scale1, size=(d_in, d_hidden))
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.normal(scale=scale2, size=(d_hidden, d_hidden))
        self.b2 = np.zeros(d_hidden)
        self.w3 = rng.normal(scale=scale2, size=d_hidden)
        self.b3 = 0.0
        self._adam_state = None

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.w3, np.array([self.b3])]

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2, self.w3, b3 = params
        self.b3 = float(b3[0])

    def forward(self, x, cache=False):
        """x: (N, d_in) -> logits: (N,). If cache, also return intermediates."""
        z1 = x @ self.W1 + self.b1
        h1 = _lrelu(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = _lrelu(z2)
        logit = h2 @ self.w3 + self.b3
        if cache:
            return logit, (x, z1, h1, z2, h2)
        return logit

    def predict_proba(self, x):
        return _sigmoid(self.forward(x))

    def predict(self, x):
        return (self.predict_proba(x) >= 0.5).astype(int)

    def loss_and_grad(self, x, y, l2=0.0):
        """Binary cross-entropy loss + gradients w.r.t. all params, averaged over batch."""
        n = x.shape[0]
        logit, (xin, z1, h1, z2, h2) = self.forward(x, cache=True)
        p = _sigmoid(logit)
        eps = 1e-12
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        reg = 0.0
        if l2 > 0:
            reg = l2 * (np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.w3**2))
            loss += reg

        # dL/dlogit for mean BCE-with-sigmoid is (p - y)/n
        dlogit = (p - y) / n

        dw3 = h2.T @ dlogit + 2 * l2 * self.w3
        db3 = np.sum(dlogit)

        dh2 = np.outer(dlogit, self.w3)
        dz2 = dh2 * _lrelu_grad(z2)

        dW2 = h1.T @ dz2 + 2 * l2 * self.W2
        db2 = np.sum(dz2, axis=0)

        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * _lrelu_grad(z1)

        dW1 = xin.T @ dz1 + 2 * l2 * self.W1
        db1 = np.sum(dz1, axis=0)

        grads = [dW1, db1, dW2, db2, dw3, np.array([db3])]
        return loss, grads

    def input_grad(self, x, y_target):
        """Gradient of per-example BCE loss (against y_target) w.r.t. input x.

        Used for gradient-based adversarial search; batched over x's rows,
        each row uses the corresponding y_target entry (no averaging).
        """
        logit, (xin, z1, h1, z2, h2) = self.forward(x, cache=True)
        p = _sigmoid(logit)
        dlogit = p - y_target  # per-example, unaveraged
        dh2 = np.outer(dlogit, self.w3)
        dz2 = dh2 * _lrelu_grad(z2)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * _lrelu_grad(z1)
        dx = dz1 @ self.W1.T
        return dx

    def fit(self, x, y, epochs=300, lr=0.01, batch_size=256, l2=1e-4, rng=None,
            x_val=None, y_val=None, patience=30, verbose=False):
        rng = rng or np.random.default_rng()
        n = x.shape[0]
        params = self.params()
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        best_val = np.inf
        best_params = [p.copy() for p in params]
        stall = 0

        for epoch in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                _, grads = self.loss_and_grad(x[idx], y[idx], l2=l2)
                t += 1
                params = self.params()
                new_params = []
                for p, g, mi, vi in zip(params, grads, m, v):
                    mi[:] = beta1 * mi + (1 - beta1) * g
                    vi[:] = beta2 * vi + (1 - beta2) * (g**2)
                    mhat = mi / (1 - beta1**t)
                    vhat = vi / (1 - beta2**t)
                    new_params.append(p - lr * mhat / (np.sqrt(vhat) + eps))
                self.set_params(new_params)

            if x_val is not None:
                val_loss, _ = self.loss_and_grad(x_val, y_val, l2=0.0)
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_params = [p.copy() for p in self.params()]
                    stall = 0
                else:
                    stall += 1
                if stall >= patience:
                    break
                if verbose and epoch % 50 == 0:
                    print(f"epoch {epoch} val_loss {val_loss:.4f}")

        if x_val is not None:
            self.set_params(best_params)
        return self
