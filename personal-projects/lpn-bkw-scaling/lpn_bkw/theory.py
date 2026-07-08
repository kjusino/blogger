"""Textbook complexity accounting for classical (pivot-and-eliminate) BKW.

This module encodes the standard "back of the envelope" BKW cost model found
in expositions of Blum-Kalai-Wasserman (2003) / Levieil-Fouque (2006):

  * Partition the n secret bits into `a = n / b` blocks of `b` bits.
  * Recovering one target block requires eliminating the other `a - 1`
    blocks. Each elimination level XORs matching samples together
    (pivot-and-eliminate), which costs ~2^b "wasted" pivot samples per
    level and squares the noise bias of every surviving sample.
  * After `a - 1` levels the surviving samples have bias
    delta_final = delta_0 ^ (2 ** (a - 1)), where delta_0 = 1 - 2*tau.
  * A final correlation/majority-vote solve over the 2^b candidates for the
    target block needs M_final ~ C / delta_final^2 samples to succeed with
    good probability (Chernoff-type bound, C a confidence constant).
  * The whole secret is recovered with `a` independent passes (one target
    block per pass), so total query complexity is a * (M_final + (a-1)*2^b).

This module implements exactly that model so it can be compared against
measurements from `lpn_bkw.bkw`, which implements the actual algorithm.
"""

import math


def bias(tau: float) -> float:
    """delta_0 = 1 - 2*tau, the correlation of a single raw LPN sample."""
    return 1.0 - 2.0 * tau


def final_bias(tau: float, a: int) -> float:
    """Bias remaining after a-1 pairwise-XOR elimination levels."""
    if a < 1:
        raise ValueError("a must be >= 1")
    return bias(tau) ** (2 ** (a - 1))


def required_final_samples(tau: float, a: int, confidence_const: float = 20.0,
                            floor: int = 50) -> int:
    """M_final: samples needed at the last level for a reliable solve."""
    fb = final_bias(tau, a)
    denom = fb * fb
    if denom <= 0:
        return float("inf")
    return max(floor, math.ceil(confidence_const / denom))


def queries_per_pass(n: int, b: int, tau: float, confidence_const: float = 20.0,
                      margin: float = 1.3) -> float:
    """Theoretical raw-query budget for ONE target-block recovery pass."""
    if n % b != 0:
        raise ValueError("this study requires n to be a multiple of b")
    a = n // b
    m_final = required_final_samples(tau, a, confidence_const)
    pivot_cost = (a - 1) * (2 ** b)
    return m_final + margin * pivot_cost


def total_queries(n: int, b: int, tau: float, confidence_const: float = 20.0,
                   margin: float = 1.3) -> float:
    """Theoretical total query budget across all a recovery passes."""
    a = n // b
    return a * queries_per_pass(n, b, tau, confidence_const, margin)


def optimal_b(n: int, tau: float, b_candidates, confidence_const: float = 20.0,
              margin: float = 1.3) -> int:
    """argmin over b_candidates (divisors of n) of total_queries(n, b, tau)."""
    valid = [b for b in b_candidates if n % b == 0]
    if not valid:
        raise ValueError("no candidate window size divides n")
    return min(valid, key=lambda b: total_queries(n, b, tau, confidence_const, margin))
