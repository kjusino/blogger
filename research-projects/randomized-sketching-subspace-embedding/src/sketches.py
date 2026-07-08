"""Random linear sketches used to test the Johnson-Lindenstrauss / subspace-embedding
guarantee: for a matrix A with n rows, an n x k random sketch S should satisfy

    (1 - eps) ||A x||^2 <= ||S^T A x||^2 <= (1 + eps) ||A x||^2   for all x

with high probability once k = Omega(d / eps^2), where d = rank(A) (or a dimension
bound on the subspace of interest). Each function here returns S^T A (shape k x d)
directly rather than the sketch operator itself.
"""

import numpy as np


def next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def fwht(x):
    """Fast Walsh-Hadamard transform along axis 0. x.shape[0] must be a power of two.
    Fully vectorized: log2(n) passes, each an O(n * prod(rest_shape)) numpy op.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n & (n - 1) != 0:
        raise ValueError("fwht requires x.shape[0] to be a power of two")
    rest_shape = x.shape[1:]
    out = x.copy()
    h = 1
    while h < n:
        out = out.reshape((n // (2 * h), 2, h) + rest_shape)
        a, b = out[:, 0], out[:, 1]
        out = np.concatenate((a + b, a - b), axis=1)
        out = out.reshape((n,) + rest_shape)
        h *= 2
    return out


def gaussian_sketch(A, k, rng):
    """Dense Gaussian JL sketch: S is k x n i.i.d. N(0, 1/k). O(n * d * k) time."""
    n = A.shape[0]
    S = rng.standard_normal((k, n)) / np.sqrt(k)
    return S @ A


def countsketch(A, k, rng):
    """CountSketch (Clarkson-Woodruff): each row hashed into one of k buckets with a
    random sign, buckets summed. O(nnz(A)) time -- independent of k and of d for a
    dense A, this is O(n * d).
    """
    n, d = A.shape
    buckets = rng.integers(0, k, size=n)
    signs = rng.choice([-1.0, 1.0], size=n)
    SA = np.zeros((k, d))
    np.add.at(SA, buckets, A * signs[:, None])
    return SA


def srht_sketch(A, k, rng, precondition=True):
    """Subsampled Randomized Hadamard Transform: flatten row leverage with a random
    sign flip + Hadamard mix (O(n * d * log n)), then uniformly subsample k rows
    (O(k) extra). With precondition=False this degrades to *plain uniform row
    sampling* -- used as an ablation to show the D/H mixing step is load-bearing,
    not decorative, on subspaces with concentrated leverage.
    """
    n = A.shape[0]
    n_pad = next_pow2(n)
    if n_pad != n:
        A = np.vstack([A, np.zeros((n_pad - n, A.shape[1]))])
    if precondition:
        signs = rng.choice([-1.0, 1.0], size=n_pad)
        A = fwht(A * signs[:, None]) / np.sqrt(n_pad)
    idx = rng.choice(n_pad, size=k, replace=False)
    return A[idx] * np.sqrt(n_pad / k)


SKETCHES = {
    "gaussian": lambda A, k, rng: gaussian_sketch(A, k, rng),
    "srht": lambda A, k, rng: srht_sketch(A, k, rng, precondition=True),
    "countsketch": lambda A, k, rng: countsketch(A, k, rng),
}
