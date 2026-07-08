import numpy as np

from grok.optim import AdamW


def test_adamw_decreases_quadratic_loss():
    rng = np.random.default_rng(0)
    params = {"x": rng.normal(size=5)}
    opt = AdamW(params, lr=0.1, weight_decay=0.0)

    def loss_and_grad(p):
        return float(np.sum(p["x"] ** 2)), {"x": 2 * p["x"]}

    losses = []
    for _ in range(200):
        loss, grads = loss_and_grad(params)
        losses.append(loss)
        opt.step(params, grads)
    assert losses[-1] < losses[0] * 1e-3


def test_adamw_weight_decay_shrinks_params_at_zero_gradient():
    params = {"x": np.array([1.0, -1.0, 2.0])}
    opt = AdamW(params, lr=0.1, weight_decay=0.5)
    zero_grad = {"x": np.zeros(3)}
    before = params["x"].copy()
    opt.step(params, zero_grad)
    # with zero gradient, only the decoupled weight-decay term acts
    assert np.all(np.abs(params["x"]) < np.abs(before))


def test_adamw_state_shapes_match_params():
    rng = np.random.default_rng(0)
    params = {"a": rng.normal(size=(3, 4)), "b": rng.normal(size=(2,))}
    opt = AdamW(params)
    assert opt.m["a"].shape == (3, 4)
    assert opt.v["b"].shape == (2,)


def test_adamw_step_counter_increments():
    params = {"x": np.zeros(3)}
    opt = AdamW(params)
    grads = {"x": np.ones(3)}
    for expected in range(1, 6):
        opt.step(params, grads)
        assert opt.t == expected
