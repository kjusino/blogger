"""Computes nontrivial zeta zeros via mpmath's Riemann-Siegel + Turing's
method implementation (rigorous zero counting -- no missed or spurious
zeros, unconditional on the Riemann Hypothesis)."""

import json
import os

import mpmath


def zero_heights(n_start, count, dps=25, cache_path=None):
    """Imaginary parts of nontrivial zeta zeros #n_start .. #(n_start+count-1),
    1-indexed to match mpmath.zetazero's convention. Returns a plain list of
    floats, sorted ascending (zetazero is monotonic in n).

    If `cache_path` is given, results are read from / written to that JSON
    file so a re-run (e.g. after fixing a downstream analysis bug) doesn't
    re-pay the cost of the highest windows, which can take many minutes.
    """
    if n_start < 1:
        raise ValueError("n_start is 1-indexed; the first zero is n_start=1")
    if count < 1:
        raise ValueError("count must be >= 1")

    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("n_start") == n_start and cached.get("count") == count:
            return cached["heights"]

    mpmath.mp.dps = dps
    heights = [float(mpmath.im(mpmath.zetazero(n))) for n in range(n_start, n_start + count)]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"n_start": n_start, "count": count, "heights": heights}, f)

    return heights
