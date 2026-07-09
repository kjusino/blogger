import numpy as np
import pytest

from src.mlp import MLP


def _numeric_grad(model, x, y, param_idx, h=1e-5, l2=0.0):
    params = model.params()
    p = params[param_idx]
    grad = np.zeros_like(p)
    it = np.nditer(p, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = p[idx]
        p[idx] = orig + h
        model.set_params(params)
        loss_plus, _ = model.loss_and_grad(x, y, l2=l2)
        p[idx] = orig - h
        model.set_params(params)
        loss_minus, _ = model.loss_and_grad(x, y, l2=l2)
        p[idx] = orig
        model.set_params(params)
        grad[idx] = (loss_plus - loss_minus) / (2 * h)
        it.iternext()
    return grad


@pytest.mark.parametrize("param_idx", [0, 1, 2, 3, 4, 5])
def test_backprop_gradient_matches_finite_differences(param_idx):
    rng = np.random.default_rng(42)
    d_in, d_hidden = 4, 5
    model = MLP(d_in, d_hidden, rng)
    x = rng.normal(size=(6, d_in))
    y = rng.integers(0, 2, size=6).astype(float)

    _, analytic_grads = model.loss_and_grad(x, y, l2=1e-3)
    numeric = _numeric_grad(model, x, y, param_idx, l2=1e-3)

    analytic = analytic_grads[param_idx]
    max_abs_err = np.max(np.abs(analytic - numeric))
    denom = np.maximum(np.max(np.abs(analytic)), np.max(np.abs(numeric)))
    rel_err = max_abs_err / max(denom, 1e-8)
    assert rel_err < 1e-4, f"param {param_idx}: rel_err={rel_err}"


def test_input_grad_matches_finite_differences():
    rng = np.random.default_rng(7)
    d_in, d_hidden = 5, 6
    model = MLP(d_in, d_hidden, rng)
    x = rng.normal(size=(1, d_in))
    y_target = np.array([1.0])

    analytic = model.input_grad(x, y_target)[0]

    h = 1e-5
    numeric = np.zeros(d_in)
    for i in range(d_in):
        xp = x.copy()
        xp[0, i] += h
        logit_p = model.forward(xp)[0]
        xm = x.copy()
        xm[0, i] -= h
        logit_m = model.forward(xm)[0]
        p_plus = 1 / (1 + np.exp(-logit_p))
        p_minus = 1 / (1 + np.exp(-logit_m))
        loss_plus = -(y_target[0] * np.log(p_plus + 1e-12) + (1 - y_target[0]) * np.log(1 - p_plus + 1e-12))
        loss_minus = -(y_target[0] * np.log(p_minus + 1e-12) + (1 - y_target[0]) * np.log(1 - p_minus + 1e-12))
        numeric[i] = (loss_plus - loss_minus) / (2 * h)

    max_abs_err = np.max(np.abs(analytic - numeric))
    assert max_abs_err < 1e-4


def test_fit_reduces_loss_and_separates_linearly_separable_data():
    rng = np.random.default_rng(0)
    n = 200
    x0 = rng.normal(loc=-2.0, scale=0.5, size=(n, 2))
    x1 = rng.normal(loc=2.0, scale=0.5, size=(n, 2))
    x = np.concatenate([x0, x1])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    perm = rng.permutation(2 * n)
    x, y = x[perm], y[perm]

    model = MLP(2, 8, rng)
    loss_before, _ = model.loss_and_grad(x, y)
    model.fit(x, y, epochs=200, lr=0.05, batch_size=64, l2=1e-4, rng=rng)
    loss_after, _ = model.loss_and_grad(x, y)

    assert loss_after < loss_before
    acc = np.mean(model.predict(x) == y.astype(int))
    assert acc > 0.95
