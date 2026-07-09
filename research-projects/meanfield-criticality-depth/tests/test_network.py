import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from meanfield import fixed_point_q, q_map, tanh
from network import (
    forward,
    init_layers,
    make_dataset,
    propagate_correlation,
    propagate_gradient_norms,
    train_classifier,
)


def test_init_layers_shapes():
    weights, biases = init_layers(depth=5, width=32, input_dim=32, sigma_w2=1.0, sigma_b2=0.1,
                                   rng=np.random.default_rng(0))
    assert len(weights) == 5 and len(biases) == 5
    for W, b in zip(weights, biases):
        assert W.shape == (32, 32)
        assert b.shape == (32,)


def test_forward_pass_shapes_and_bounded_output():
    rng = np.random.default_rng(0)
    weights, biases = init_layers(depth=4, width=16, input_dim=16, sigma_w2=1.0, sigma_b2=0.1, rng=rng)
    X = rng.normal(size=(10, 16))
    pre_acts, acts = forward(X, weights, biases)
    assert len(pre_acts) == 4 and len(acts) == 4
    for z, y in zip(pre_acts, acts):
        assert z.shape == (10, 16)
        assert y.shape == (10, 16)
        assert np.all(np.abs(y) <= 1.0)  # tanh is bounded


def test_forward_pass_variance_matches_meanfield_prediction():
    # Monte Carlo check: empirical pre-activation variance of a random-init
    # layer should match the theoretical one-step q_map, for a wide-enough
    # network, to within a generous statistical tolerance.
    #
    # The *first* pre-activation z^1 = W^1 X + b^1 is a linear function of
    # the raw (un-squashed) input, so q^1 = sigma_w2 * Var(X) + sigma_b2
    # with no phi() involved -- phi only enters once something has already
    # passed through a previous activation. So this checks the *second*
    # layer, z^2 = W^2 phi(z^1) + b^2, against q_map applied to the
    # (measured) variance of z^1, which is exactly the standard recursion.
    sigma_w2, sigma_b2 = 1.5, 0.2
    width = 2000
    rng = np.random.default_rng(1)
    weights, biases = init_layers(depth=2, width=width, input_dim=width, sigma_w2=sigma_w2,
                                   sigma_b2=sigma_b2, rng=rng)
    X = rng.normal(size=(1, width))  # single input with unit variance per coordinate
    pre_acts, _ = forward(X, weights, biases)
    q1_empirical = float(np.var(pre_acts[0]))
    q2_empirical = float(np.var(pre_acts[1]))
    theoretical_q2 = q_map(q1_empirical, sigma_w2, sigma_b2, tanh)
    assert q2_empirical == pytest.approx(theoretical_q2, rel=0.15)


def test_correlation_propagation_ordered_phase_increases_toward_one():
    sigma_w2, sigma_b2 = 0.8, 0.05
    q_star = fixed_point_q(sigma_w2, sigma_b2, tanh)
    c_hat = propagate_correlation(depth=40, width=300, sigma_w2=sigma_w2, sigma_b2=sigma_b2,
                                   seed=0, q_star=q_star, c0=0.999)
    assert c_hat[-1] > c_hat[0]
    assert c_hat[-1] == pytest.approx(1.0, abs=1e-3)


def test_correlation_propagation_chaotic_phase_decreases_from_one():
    sigma_w2, sigma_b2 = 3.2, 0.1
    q_star = fixed_point_q(sigma_w2, sigma_b2, tanh)
    c_hat = propagate_correlation(depth=60, width=300, sigma_w2=sigma_w2, sigma_b2=sigma_b2,
                                   seed=0, q_star=q_star, c0=0.999)
    assert c_hat[-1] < c_hat[0]
    assert c_hat[-1] < 0.9  # should have visibly separated from the input correlation


def test_gradient_norms_decay_in_ordered_phase_and_grow_in_chaotic_phase():
    g_ordered = propagate_gradient_norms(depth=40, width=150, sigma_w2=0.8, sigma_b2=0.05, seed=0)
    g_chaotic = propagate_gradient_norms(depth=40, width=150, sigma_w2=3.2, sigma_b2=0.1, seed=0)
    assert g_ordered[-1] < g_ordered[0] * 1e-3
    assert g_chaotic[-1] > g_chaotic[0] * 1e3


def test_make_dataset_is_balanced_and_labels_match_shift_direction():
    X, y = make_dataset(200, dim=10, seed=0, margin=3.0)
    assert X.shape == (200, 10)
    assert set(np.unique(y).tolist()) == {0.0, 1.0}
    assert abs(y.sum() - 100) <= 1  # balanced up to odd/even rounding
    assert X[y == 1, 0].mean() > X[y == 0, 0].mean()  # positive class shifted toward +margin


def test_shallow_network_trains_regardless_of_regime():
    for sigma_w2, sigma_b2 in [(0.8, 0.05), (1.9861, 0.1), (3.2, 0.1)]:
        result = train_classifier(depth=2, width=32, sigma_w2=sigma_w2, sigma_b2=sigma_b2, seed=0)
        assert result["trainable"]
        assert result["final_acc"] >= 0.95


def test_very_deep_ordered_network_fails_to_train():
    # Deep in the ordered phase, backprop signal vanishes to numerical zero
    # long before 150 layers, so no layer other than the readout can adapt
    # and accuracy should not clear the shallow-network bar.
    result = train_classifier(depth=150, width=32, sigma_w2=0.5, sigma_b2=0.02, seed=0)
    assert not result["trainable"]


def test_exploding_gradients_are_caught_not_propagated_as_nan():
    # Deep + strongly chaotic: gradients should blow up numerically. The
    # function must report this as a clean failure, not raise or return NaN.
    result = train_classifier(depth=200, width=32, sigma_w2=6.0, sigma_b2=0.3, seed=0, n_steps=50)
    assert np.isfinite(result["final_loss"]) or result["final_loss"] == float("inf")
    assert not result["trainable"]
