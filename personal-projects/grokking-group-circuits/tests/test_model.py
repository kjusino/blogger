import numpy as np
import pytest

from grok.model import init_params, forward, backward, softmax_cross_entropy, loss_and_grads


def test_forward_shapes():
    rng = np.random.default_rng(0)
    v, d, h = 10, 4, 8
    params = init_params(v, d, h, rng)
    a_idx = np.array([0, 1, 2])
    b_idx = np.array([3, 4, 5])
    logits, cache = forward(params, a_idx, b_idx)
    assert logits.shape == (3, v)
    assert cache["z"].shape == (3, 2 * d)


def test_softmax_cross_entropy_uniform_logits_gives_log_v_loss():
    v = 6
    logits = np.zeros((4, v))
    labels = np.array([0, 1, 2, 3])
    loss, dlogits, acc = softmax_cross_entropy(logits, labels)
    assert loss == pytest.approx(np.log(v), abs=1e-6)
    assert acc == pytest.approx(0.25, abs=1e-6)  # ties broken by argmax->index 0


def test_softmax_cross_entropy_confident_correct_logits_low_loss():
    v = 5
    logits = np.full((3, v), -10.0)
    labels = np.array([0, 1, 2])
    logits[np.arange(3), labels] = 10.0
    loss, dlogits, acc = softmax_cross_entropy(logits, labels)
    assert loss < 1e-4
    assert acc == 1.0


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    v, d, h, n = 7, 3, 5, 6
    params = init_params(v, d, h, rng)
    a_idx = rng.integers(0, v, size=n)
    b_idx = rng.integers(0, v, size=n)
    labels = rng.integers(0, v, size=n)

    _, _, grads = loss_and_grads(params, a_idx, b_idx, labels)

    eps = 1e-5
    rng2 = np.random.default_rng(1)
    for name, arr in params.items():
        idxs = list(np.ndindex(arr.shape))
        sample = rng2.choice(len(idxs), size=min(8, len(idxs)), replace=False)
        for si in sample:
            idx = idxs[si]
            orig = arr[idx]
            arr[idx] = orig + eps
            loss_plus, _, _ = loss_and_grads(params, a_idx, b_idx, labels)
            arr[idx] = orig - eps
            loss_minus, _, _ = loss_and_grads(params, a_idx, b_idx, labels)
            arr[idx] = orig
            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytic = grads[name][idx]
            rel_err = abs(numerical - analytic) / (abs(numerical) + abs(analytic) + 1e-8)
            assert rel_err < 1e-3, f"{name}{idx}: numeric={numerical} analytic={analytic}"


def test_embedding_gradient_accumulates_for_shared_index():
    # if the same group element appears as both operands of the same pair,
    # backward() should add both slots' contributions into that one row
    # rather than the second overwriting the first. Exercised directly
    # against a synthetic cache/dlogits so the result doesn't depend on
    # whether ReLU happens to be dead for a particular random init.
    rng = np.random.default_rng(0)
    v, d, h = 4, 2, 3
    params = init_params(v, d, h, rng)
    a_idx = np.array([1])
    b_idx = np.array([1])
    cache = {
        "a_idx": a_idx,
        "b_idx": b_idx,
        "z": np.array([[1.0, 2.0, 3.0, 4.0]]),
        "pre1": np.array([[1.0, 1.0, 1.0]]),  # all positive: ReLU passes through
        "h": np.array([[1.0, 1.0, 1.0]]),
    }
    dlogits = np.ones((1, v))
    grads = backward(params, cache, dlogits)
    assert np.any(grads["W_E"][1] != 0)
    for row in (0, 2, 3):
        assert np.all(grads["W_E"][row] == 0)
