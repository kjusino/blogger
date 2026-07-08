import numpy as np
import pytest
from scipy.linalg import hadamard

from src.sketches import countsketch, fwht, gaussian_sketch, next_pow2, srht_sketch


def test_next_pow2():
    assert next_pow2(1) == 1
    assert next_pow2(2) == 2
    assert next_pow2(3) == 4
    assert next_pow2(17) == 32
    assert next_pow2(256) == 256


@pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
def test_fwht_matches_reference_hadamard_matrix(n):
    H = hadamard(n).astype(float)
    got = fwht(np.eye(n))
    np.testing.assert_allclose(got, H, atol=1e-9)


def test_fwht_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        fwht(np.zeros((6, 3)))


def test_fwht_is_self_inverse_up_to_scaling():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 5))
    y = fwht(fwht(x)) / 32.0
    np.testing.assert_allclose(y, x, atol=1e-9)


def test_fwht_preserves_norm_up_to_sqrt_n():
    rng = np.random.default_rng(1)
    n = 64
    x = rng.standard_normal((n, 3))
    y = fwht(x)
    np.testing.assert_allclose(
        np.linalg.norm(y, axis=0), np.sqrt(n) * np.linalg.norm(x, axis=0), rtol=1e-9
    )


@pytest.mark.parametrize("sketch_fn", [gaussian_sketch, countsketch])
def test_sketch_output_shape(sketch_fn):
    rng = np.random.default_rng(2)
    A = rng.standard_normal((100, 7))
    out = sketch_fn(A, 15, rng)
    assert out.shape == (15, 7)


def test_srht_output_shape_handles_non_power_of_two_n():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((100, 7))  # 100 is not a power of two
    out = srht_sketch(A, 15, rng)
    assert out.shape == (15, 7)


def test_gaussian_sketch_isometry_in_expectation():
    """E[S^T S] = I_n for the Gaussian sketch -- an unbiased estimator of squared
    column norms, averaged over many independent draws."""
    rng = np.random.default_rng(4)
    n, k, trials = 6, 40, 4000
    A = np.eye(n)
    acc = np.zeros((n, n))
    for _ in range(trials):
        SA = gaussian_sketch(A, k, rng)
        acc += SA.T @ SA
    est = acc / trials
    np.testing.assert_allclose(est, np.eye(n), atol=0.1)


def test_countsketch_isometry_in_expectation():
    rng = np.random.default_rng(5)
    n, k, trials = 6, 40, 4000
    A = np.eye(n)
    acc = np.zeros((n, n))
    for _ in range(trials):
        SA = countsketch(A, k, rng)
        acc += SA.T @ SA
    est = acc / trials
    np.testing.assert_allclose(est, np.eye(n), atol=0.1)


def test_srht_full_sketch_is_exactly_orthogonal():
    """When k == n_pad (no actual subsampling, D+H only reorders/rotates), the SRHT
    sketch is an exact orthogonal transform for ANY subspace -- distortion should be
    ~0 to floating-point precision, regardless of the rng draw."""
    rng = np.random.default_rng(6)
    n = 32  # already a power of two
    Q, _ = np.linalg.qr(rng.standard_normal((n, 5)))
    SQ = srht_sketch(Q, n, rng)
    sv = np.linalg.svd(SQ, compute_uv=False)
    np.testing.assert_allclose(sv, np.ones_like(sv), atol=1e-9)


def test_srht_no_precondition_is_plain_row_sampling():
    A = np.random.default_rng(7).standard_normal((64, 4))
    out = srht_sketch(A, 10, np.random.default_rng(99), precondition=False)
    idx = np.random.default_rng(99).choice(64, size=10, replace=False)
    expected = A[idx] * np.sqrt(64 / 10)
    np.testing.assert_allclose(out, expected)
