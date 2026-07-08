"""A minimal 2-layer MLP for group-operation classification, with the
forward/backward pass written out by hand (no autodiff framework) so the
whole training pipeline has no dependency beyond numpy.

Architecture: shared embedding table W_E (one row per group element, used for
both operands) -> concat -> Linear -> ReLU -> Linear -> softmax over group
elements. This is the same family of architecture (embed, mix, unembed) used
in the grokking literature for MLPs on modular arithmetic; it's simple enough
to backprop by hand and to later interrogate the embedding table `W_E` for
representation-theoretic structure (see grok/spectral.py).
"""
import numpy as np


def init_params(vocab_size, emb_dim, hidden_dim, rng):
    return {
        "W_E": rng.normal(0, 0.02, size=(vocab_size, emb_dim)),
        "W1": rng.normal(0, 1.0 / np.sqrt(2 * emb_dim), size=(2 * emb_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim),
        "W2": rng.normal(0, 1.0 / np.sqrt(hidden_dim), size=(hidden_dim, vocab_size)),
        "b2": np.zeros(vocab_size),
    }


def forward(params, a_idx, b_idx):
    e_a = params["W_E"][a_idx]
    e_b = params["W_E"][b_idx]
    z = np.concatenate([e_a, e_b], axis=1)
    pre1 = z @ params["W1"] + params["b1"]
    h = np.maximum(pre1, 0.0)
    logits = h @ params["W2"] + params["b2"]
    cache = {"a_idx": a_idx, "b_idx": b_idx, "z": z, "pre1": pre1, "h": h}
    return logits, cache


def softmax_cross_entropy(logits, labels):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.mean(np.log(probs[np.arange(n), labels] + 1e-12))
    dlogits = probs.copy()
    dlogits[np.arange(n), labels] -= 1.0
    dlogits /= n
    acc = float(np.mean(np.argmax(logits, axis=1) == labels))
    return float(loss), dlogits, acc


def backward(params, cache, dlogits):
    h, z, pre1 = cache["h"], cache["z"], cache["pre1"]
    grads = {
        "W2": h.T @ dlogits,
        "b2": dlogits.sum(axis=0),
    }
    dh = dlogits @ params["W2"].T
    dpre1 = dh * (pre1 > 0)
    grads["W1"] = z.T @ dpre1
    grads["b1"] = dpre1.sum(axis=0)
    dz = dpre1 @ params["W1"].T

    d = params["W_E"].shape[1]
    de_a, de_b = dz[:, :d], dz[:, d:]
    grad_we = np.zeros_like(params["W_E"])
    np.add.at(grad_we, cache["a_idx"], de_a)
    np.add.at(grad_we, cache["b_idx"], de_b)
    grads["W_E"] = grad_we
    return grads


def loss_and_grads(params, a_idx, b_idx, labels):
    logits, cache = forward(params, a_idx, b_idx)
    loss, dlogits, acc = softmax_cross_entropy(logits, labels)
    grads = backward(params, cache, dlogits)
    return loss, acc, grads
