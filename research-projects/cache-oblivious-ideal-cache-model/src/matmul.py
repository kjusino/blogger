"""
Three n x n matrix-multiplication algorithms, instrumented against a shared
IdealCache (src.cache_sim), used to test the miss-count bounds from:

  - Hong & Kung (1981), "I/O complexity: the red-blue pebble game" -- the
    classic Theta(n^3) worst case for a naive, unblocked triple loop.
  - Frigo, Leiserson, Prokop & Ramachandran (1999), "Cache-oblivious
    algorithms" -- Theta(n^3 / (B * sqrt(M))) for both a cache-aware blocked
    algorithm (tile size Theta(sqrt(M))) and a cache-oblivious recursive
    algorithm that never references B or M.

All three write into a fresh, zero-initialized C and are numerically
equivalent; only their cache behavior differs.
"""

from .cache_sim import Matrix


def _get(arr, stride, i, j):
    return arr.get(i * stride + j)


def _set(arr, stride, i, j, value):
    arr.set(i * stride + j, value)


def multiply_naive(A, B, C):
    """Textbook i, j, k loop order. A's rows are scanned contiguously (good
    spatial locality); B is scanned column-by-column with stride n (poor
    spatial locality, and no temporal reuse across (i, j) pairs). The
    running sum is accumulated in a Python local ("register"), matching the
    literature convention of not counting register traffic as cache
    traffic. Expected: Theta(n^3) misses, dominated by B, roughly
    independent of B (block size) and M (cache size) for n^2 > M."""
    n = A.n
    a_arr, a_s = A.arr, A.n
    b_arr, b_s = B.arr, B.n
    c_arr, c_s = C.arr, C.n
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += _get(a_arr, a_s, i, k) * _get(b_arr, b_s, k, j)
            _set(c_arr, c_s, i, j, s)


def multiply_blocked(A, B, C, tile):
    """Cache-aware three-level-blocked matmul: partitions the i, j, k
    iteration space into tile x tile x tile blocks. With tile =
    Theta(sqrt(M)) (so that an A-tile, B-tile, and C-tile of size
    tile^2 words each fit simultaneously in an M-word cache), each block's
    work is done almost entirely out of cache: Theta(n^3 / (B * sqrt(M)))
    misses."""
    if tile <= 0:
        raise ValueError("tile must be positive")
    n = A.n
    a_arr, a_s = A.arr, A.n
    b_arr, b_s = B.arr, B.n
    c_arr, c_s = C.arr, C.n
    for ii in range(0, n, tile):
        i_end = min(ii + tile, n)
        for jj in range(0, n, tile):
            j_end = min(jj + tile, n)
            for kk in range(0, n, tile):
                k_end = min(kk + tile, n)
                for i in range(ii, i_end):
                    for j in range(jj, j_end):
                        s = _get(c_arr, c_s, i, j) if kk > 0 else 0.0
                        for k in range(kk, k_end):
                            s += _get(a_arr, a_s, i, k) * _get(b_arr, b_s, k, j)
                        _set(c_arr, c_s, i, j, s)


def multiply_oblivious(A, B, C, base_case=8):
    """Cache-oblivious recursive matmul (the "split the largest dimension"
    variant, e.g. MIT 6.172): recursively halves whichever of the three
    dimensions (rows of A/C, shared inner dimension, columns of B/C) is
    currently largest, bottoming out in a base-case triple loop once all
    three dimensions are <= base_case. Never references B or M -- yet the
    FLPR99 theorem predicts the same Theta(n^3 / (B * sqrt(M))) bound as
    the *optimally tuned* blocked algorithm above, for any tall cache
    (M = Omega(B^2))."""
    n = A.n
    a_arr, a_s = A.arr, A.n
    b_arr, b_s = B.arr, B.n
    c_arr, c_s = C.arr, C.n

    def rec(ar, ac, br, bc, cr, cc, m, k, p):
        if m <= base_case and k <= base_case and p <= base_case:
            for i in range(m):
                for j in range(p):
                    s = 0.0
                    for t in range(k):
                        s += _get(a_arr, a_s, ar + i, ac + t) * _get(
                            b_arr, b_s, br + t, bc + j
                        )
                    cur = _get(c_arr, c_s, cr + i, cc + j)
                    _set(c_arr, c_s, cr + i, cc + j, cur + s)
            return
        if m >= k and m >= p:
            m1 = m // 2
            rec(ar, ac, br, bc, cr, cc, m1, k, p)
            rec(ar + m1, ac, br, bc, cr + m1, cc, m - m1, k, p)
        elif p >= m and p >= k:
            p1 = p // 2
            rec(ar, ac, br, bc, cr, cc, m, k, p1)
            rec(ar, ac, br, bc + p1, cr, cc + p1, m, k, p - p1)
        else:
            k1 = k // 2
            rec(ar, ac, br, bc, cr, cc, m, k1, p)
            rec(ar, ac + k1, br + k1, bc, cr, cc, m, k - k1, p)

    rec(0, 0, 0, 0, 0, 0, n, n, n)


ALGORITHMS = {
    "naive": lambda A, B, C, tile, base_case: multiply_naive(A, B, C),
    "blocked": lambda A, B, C, tile, base_case: multiply_blocked(A, B, C, tile),
    "oblivious": lambda A, B, C, tile, base_case: multiply_oblivious(
        A, B, C, base_case
    ),
}


def default_tile(M, n=None):
    """Standard tile-size choice so that 3 tiles of size t^2 fit in an
    M-word cache: t = floor(sqrt(M / 3)). If `n` is given, snaps down to
    the largest divisor of n that does not exceed that bound, so blocking
    is exact (no ragged boundary tiles) -- ragged tiles are a real, known
    source of noise in tiled-matmul performance, but they are a distinct
    phenomenon from the asymptotic bound this study measures, so we
    control for it here."""
    t = max(int((M / 3) ** 0.5), 1)
    if n is not None:
        for cand in range(min(t, n), 0, -1):
            if n % cand == 0:
                return cand
    return t
