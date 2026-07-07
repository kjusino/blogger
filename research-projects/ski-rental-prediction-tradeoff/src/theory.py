"""Closed-form Robustness and Consistency for the lambda-buy-threshold family.

Both formulas below are original derivations for this project (they are not
transcribed from Purohit, Svitkina & Kumar 2018 or any other paper -- see
README.md for the field-founding citation and an explicit disclaimer that
these specific formulas are not a reproduction of it).

Derivation sketch (also verified numerically in
tests/test_theory_vs_brute_force.py against a from-scratch brute-force
search -- see brute_force.py):

tau(y, lam, b), viewed as a function of y for fixed (lam, b), is
non-decreasing in y (its "continuous" pre-rounding value is
tau_star_real + lam * y for y < b, and the constant tau_star_real for
y >= b, where tau_star_real = 1 + (1 - lam) * (b - 1)). So its minimum over
y is attained at y = 0 (and re-attained, at the same value, for every
y >= b): tau_min(lam, b) = round(1 + (1 - lam) * (b - 1)) =: tau_star.

For a fixed tau, cost(x, tau, b) / OPT(x, b) equals 1 for x < tau, equals
((tau - 1) + b) / x for tau <= x < b (decreasing in x, so maximized at
x = tau), and equals ((tau - 1) + b) / b for x >= b (constant, and no
larger than the x = tau value since tau <= b). So the worst-case x for a
fixed tau is x = tau, giving ratio 1 + (b - 1) / tau.

Robustness is the worst ratio over BOTH x and y jointly, i.e. over all tau
achievable by some y, then the worst x for that tau -- which is realized at
the minimizing tau = tau_star:

    Robustness(lam, b) = 1 + (b - 1) / tau_star(lam, b)

Consistency instead fixes y = x (a perfect predictor) and asks for the
worst x. Because tau now moves together with x, the worst case is no
longer at a boundary in the same clean way; the formula below is the
closed form for the *continuous* (pre-rounding) crossover, and is validated
against a from-scratch integer brute-force search with a stated tolerance
(rounding can shift the crossover by about one integer day, which changes
the ratio by O(1/b) -- see the brute-force test module for the exact
tolerance and its justification, and for the honest reporting of
observed discrepancies rather than a forced match):

    Consistency(lam, b) = 1 + (1 - lam) * (b - 1) / b

Sanity checks (also exercised in tests/test_algorithm.py and
tests/test_theory_vs_brute_force.py):
  - lam = 0: tau_star = b, so Robustness = 1 + (b-1)/b -> 2 as b -> inf.
    This recovers the classical tight 2-competitive deterministic
    ski-rental bound.
  - lam = 1: tau_star = 1, so Robustness = b (unbounded as b -> inf):
    blindly trusting the predictor has no worst-case guarantee.
  - lam = 1: Consistency = 1 (matches OPT exactly when the prediction is
    always right).
  - lam = 0: Consistency = 1 + (b-1)/b -> 2 (no better than the classical
    worst case, since lam = 0 ignores the prediction entirely).
"""

from __future__ import annotations


def tau_star(lam: float, b: int) -> int:
    """The buy-threshold at the "hardest" prediction (y = 0, or any y >= b).

    This is the pointwise minimum of tau(y, lam, b) over all y >= 0.
    """
    if b < 1:
        raise ValueError(f"b must be >= 1, got {b}")
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    t = round(1 + (1 - lam) * (b - 1))
    return int(min(max(t, 1), b))


def robustness_formula(lam: float, b: int) -> float:
    """Closed-form worst-case (over both x and y) competitive ratio."""
    ts = tau_star(lam, b)
    return 1.0 + (b - 1) / ts


def consistency_formula(lam: float, b: int) -> float:
    """Closed-form worst-case (over x, with a perfect predictor y = x)
    competitive ratio."""
    if b < 1:
        raise ValueError(f"b must be >= 1, got {b}")
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    return 1.0 + (1 - lam) * (b - 1) / b


def robustness_tolerance(lam: float, b: int) -> float:
    """Tolerance for |brute-force Robustness - closed-form Robustness|.

    As derived in this module's docstring, tau(y, lam, b) is non-decreasing
    in y with minimum exactly tau_star at y = 0 (reproduced exactly, not
    approximately, by round(1 + (1-lam)*(b-1)) -- the same expression used
    for both y = 0 and y >= b), and for a fixed tau the worst x is exactly
    x = tau_star, which is always an integer in the brute-force scan range
    [1, 4b] since tau_star in [1, b]. So brute force and the closed form
    should agree to floating-point precision, not just approximately; this
    tolerance is a small numerical-noise cushion, not a modeling fudge
    factor.
    """
    return 1e-6


def consistency_tolerance(lam: float, b: int) -> float:
    """Tolerance for |brute-force Consistency - closed-form Consistency|.

    Unlike Robustness, the Consistency worst case (y = x, i.e. tau moves
    together with x) is not at a fixed boundary: it occurs near where
    tau(x, lam, b) crosses x, and round() can shift that integer crossover
    by up to ~0.5 days relative to the continuous crossover the closed form
    implicitly assumes. Near the crossover, cost and OPT both scale like
    O(b), and a half-day shift in where you buy changes the ratio by about
    0.5/b absolute.

    This was confirmed empirically, not just estimated: across the full
    validation grid (b in {10, 50, 200, 1000}, lam in 0.0..1.0), the
    observed max |theory - brute_force| * b was ~0.4999999999999... at
    every b (see results/theory_vs_bruteforce.csv) -- i.e. the discrepancy
    tracks 0.5/b almost exactly. We use 1/b (a 2x safety margin over that
    observed 0.5/b law), with a small floor for b so tiny it isn't
    represented in the validation grid. If a future run produced a genuine
    mismatch beyond this, the honest response is to trust the brute-force
    number and say so, not to loosen this tolerance to hide it.
    """
    return max(1.0 / b, 0.02)
