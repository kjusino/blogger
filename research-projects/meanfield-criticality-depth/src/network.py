"""From-scratch NumPy tanh MLPs used to empirically test the mean-field
predictions in ``meanfield.py``: no autodiff library, manual forward pass,
manual backprop, hand-rolled Adam. Everything here is intentionally small
(width in the tens-to-low-hundreds, depth up to a couple hundred) so a full
grid sweep runs in minutes on a CPU.

Three empirical probes are implemented against the *same* random-init
recipe (W ~ N(0, sigma_w2 / fan_in), b ~ N(0, sigma_b2)):

1. ``propagate_correlation``  -- forward-pass correlation decay of two
   nearby inputs through a single wide random network (tests chi_1 / xi_c).
2. ``propagate_gradient_norms`` -- backprop signal-norm growth/decay at
   initialization (tests that the same chi_1 governs vanishing/exploding
   gradients).
3. ``train_classifier`` -- actually trains a depth-L network with Adam on a
   synthetic classification task (tests whether xi_c predicts trainable
   depth, not just short-horizon signal propagation).
"""

import numpy as np


def init_layers(depth, width, input_dim, sigma_w2, sigma_b2, rng):
    """Sample weights/biases for `depth` tanh layers (first layer maps
    input_dim -> width, the rest width -> width)."""
    weights, biases = [], []
    fan_in = input_dim
    for _ in range(depth):
        W = rng.normal(0.0, np.sqrt(sigma_w2 / fan_in), size=(width, fan_in))
        b = rng.normal(0.0, np.sqrt(sigma_b2), size=(width,))
        weights.append(W)
        biases.append(b)
        fan_in = width
    return weights, biases


def forward(X, weights, biases):
    """Returns (pre_activations, activations), each a list of length depth,
    activations[-1] being the network's final hidden representation."""
    pre_acts, acts = [], []
    y = X
    for W, b in zip(weights, biases):
        z = y @ W.T + b
        y = np.tanh(z)
        pre_acts.append(z)
        acts.append(y)
    return pre_acts, acts


def propagate_correlation(depth, width, sigma_w2, sigma_b2, seed, q_star, c0=0.9998):
    """Feed two inputs with pre-set variance q_star and correlation c0 into a
    *single* random-weight network draw (self-averaging over width for the
    network size used here) and track the empirical cross-input correlation
    of the pre-activations at every layer.

    Returns an array of length `depth`, c_hat[l] = corr(z1^l, z2^l).
    """
    rng = np.random.default_rng(seed)
    weights, biases = init_layers(depth, width, width, sigma_w2, sigma_b2, rng)

    x1 = rng.normal(0.0, 1.0, size=width)
    x1 = x1 / np.linalg.norm(x1) * np.sqrt(q_star * width)
    noise = rng.normal(0.0, 1.0, size=width)
    noise -= (noise @ x1) / (x1 @ x1) * x1  # orthogonalize
    noise = noise / np.linalg.norm(noise) * np.sqrt(q_star * width)
    x2 = c0 * x1 + np.sqrt(max(1.0 - c0 ** 2, 0.0)) * noise

    X = np.stack([x1, x2], axis=0)
    pre_acts, _ = forward(X, weights, biases)

    c_hat = np.empty(depth)
    for l, z in enumerate(pre_acts):
        z1, z2 = z[0], z[1]
        denom = np.linalg.norm(z1) * np.linalg.norm(z2)
        c_hat[l] = float(z1 @ z2 / denom) if denom > 0 else 1.0
    return c_hat


def propagate_gradient_norms(depth, width, sigma_w2, sigma_b2, seed, batch=64):
    """Backprop a random linear readout direction through a freshly
    initialized network (no training) and track the mean squared backprop
    signal ``dz^l`` at each layer, as a function of depth from the output.

    Returns an array of length `depth`, g[l] = mean(dz_{depth-l}^2) so index
    0 is the layer closest to the output and index depth-1 is closest to the
    input -- i.e. it reads as "distance travelled backward from the loss."
    """
    rng = np.random.default_rng(seed)
    weights, biases = init_layers(depth, width, width, sigma_w2, sigma_b2, rng)
    X = rng.normal(0.0, 1.0, size=(batch, width))
    pre_acts, acts = forward(X, weights, biases)

    v = rng.normal(0.0, 1.0, size=width)
    dy = np.tile(v, (batch, 1))

    signal = np.empty(depth)
    for i in range(depth - 1, -1, -1):
        z = pre_acts[i]
        dz = dy * (1.0 - np.tanh(z) ** 2)
        signal[depth - 1 - i] = float(np.mean(dz ** 2))
        if i > 0:
            dy = dz @ weights[i]
    return signal


def make_dataset(n, dim, seed, margin=3.0):
    """Two isotropic Gaussian clusters separated along the first coordinate;
    linearly separable in the raw input, used purely as a probe of whether a
    given random network still preserves separable signal after `depth`
    random nonlinear layers."""
    rng = np.random.default_rng(seed)
    n_pos = n // 2
    n_neg = n - n_pos
    X_pos = rng.normal(0.0, 1.0, size=(n_pos, dim))
    X_pos[:, 0] += margin
    X_neg = rng.normal(0.0, 1.0, size=(n_neg, dim))
    X_neg[:, 0] -= margin
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


class _SGDMomentum:
    """Plain momentum SGD -- deliberately *not* Adam. Adam's per-parameter
    gradient normalization masks the raw vanishing/exploding gradient
    magnitudes that the mean-field theory predicts (a tiny but consistently
    directioned gradient still produces a full-sized Adam step), which
    defeats the point of using trainability as a probe of signal
    propagation. Momentum SGD keeps update size tied to gradient magnitude,
    so it fails (loss diverges or gradients vanish to numerical zero) at
    exactly the depths the theory predicts."""

    def __init__(self, shapes, lr=0.05, momentum=0.9):
        self.lr, self.momentum = lr, momentum
        self.v = [np.zeros(s) for s in shapes]

    def step(self, params, grads):
        for i, (p, g) in enumerate(zip(params, grads)):
            self.v[i] = self.momentum * self.v[i] + g
            p -= self.lr * self.v[i]


def _bce_with_logits(logits, y):
    logits = np.clip(logits, -30, 30)
    p = 1.0 / (1.0 + np.exp(-logits))
    eps = 1e-12
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return loss, p


def train_classifier(
    depth,
    width,
    sigma_w2,
    sigma_b2,
    seed,
    n_train=256,
    n_steps=200,
    batch_size=64,
    lr=0.05,
    momentum=0.9,
    success_loss_ratio=0.5,
    success_acc=0.85,
):
    """Train a depth-`depth` tanh MLP + linear readout with plain momentum
    SGD on the synthetic classification task. Returns a dict with the
    initial/final loss and accuracy and a boolean `trainable` flag.

    `trainable` is defined as: final training loss drops to at most
    `success_loss_ratio` of the initial loss AND final training accuracy
    reaches at least `success_acc`. Both conditions guard against a network
    that "trains" by barely nudging the loss without actually separating
    the classes (which can happen with badly-scaled logits).
    """
    rng = np.random.default_rng(seed)
    X, y = make_dataset(n_train, width, seed=seed * 7919 + 1)
    weights, biases = init_layers(depth, width, width, sigma_w2, sigma_b2, rng)
    W_out = rng.normal(0.0, np.sqrt(1.0 / width), size=(1, width))
    b_out = np.zeros(1)

    shapes = [w.shape for w in weights] + [b.shape for b in biases] + [W_out.shape, b_out.shape]
    opt = _SGDMomentum(shapes, lr=lr, momentum=momentum)

    def forward_full(Xb):
        pre_acts, acts = forward(Xb, weights, biases)
        logits = (acts[-1] @ W_out.T + b_out).ravel()
        return pre_acts, acts, logits

    _, _, logits0 = forward_full(X)
    initial_loss, p0 = _bce_with_logits(logits0, y)

    n = X.shape[0]
    for step in range(n_steps):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        Xb, yb = X[idx], y[idx]
        pre_acts, acts, logits = forward_full(Xb)
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        dlogits = (p - yb) / Xb.shape[0]

        dWout = dlogits[None, :] @ acts[-1]
        dbout = np.array([dlogits.sum()])
        dy = dlogits[:, None] @ W_out

        dWs, dbs = [None] * depth, [None] * depth
        for i in range(depth - 1, -1, -1):
            z = pre_acts[i]
            dz = dy * (1.0 - np.tanh(z) ** 2)
            prev_act = acts[i - 1] if i > 0 else Xb
            dWs[i] = dz.T @ prev_act
            dbs[i] = dz.sum(axis=0)
            if i > 0:
                dy = dz @ weights[i]

        params = weights + biases + [W_out, b_out]
        grads = dWs + dbs + [dWout, dbout]
        if not all(np.all(np.isfinite(g)) for g in grads):
            # exploding-gradient blow-up (expected deep in the chaotic phase):
            # stop early and report as not trainable rather than propagate NaNs.
            return {
                "initial_loss": float(initial_loss),
                "final_loss": float("inf"),
                "final_acc": 0.5,
                "trainable": False,
            }
        opt.step(params, grads)

    _, _, logits_final = forward_full(X)
    final_loss, p_final = _bce_with_logits(logits_final, y)
    if not np.isfinite(final_loss):
        final_loss, final_acc = float("inf"), 0.5
    else:
        final_acc = float(np.mean((p_final > 0.5) == (y > 0.5)))

    trainable = bool(
        np.isfinite(final_loss)
        and (final_loss <= success_loss_ratio * max(initial_loss, 1e-8))
        and (final_acc >= success_acc)
    )
    return {
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "final_acc": final_acc,
        "trainable": trainable,
    }
