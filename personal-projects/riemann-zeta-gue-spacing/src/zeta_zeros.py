"""Computes nontrivial zeta zeros via mpmath's Riemann-Siegel + Turing's
method implementation (rigorous zero counting -- no missed or spurious
zeros, unconditional on the Riemann Hypothesis)."""

import mpmath


def zero_heights(n_start, count, dps=25):
    """Imaginary parts of nontrivial zeta zeros #n_start .. #(n_start+count-1),
    1-indexed to match mpmath.zetazero's convention. Returns a plain list of
    floats, sorted ascending (zetazero is monotonic in n).
    """
    if n_start < 1:
        raise ValueError("n_start is 1-indexed; the first zero is n_start=1")
    if count < 1:
        raise ValueError("count must be >= 1")
    mpmath.mp.dps = dps
    return [float(mpmath.im(mpmath.zetazero(n))) for n in range(n_start, n_start + count)]
