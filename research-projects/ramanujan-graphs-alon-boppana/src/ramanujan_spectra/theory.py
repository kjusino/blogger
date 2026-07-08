"""The two theoretical predictions under test.

1. Alon-Boppana bound (Alon 1986; quantified by Nilli 1991; see Hoory,
   Linial & Wigderson, "Expander graphs and their applications", Bull. AMS
   43(4) 2006, Theorem 5.2): for any sequence of d-regular graphs G_n with
   n -> infinity,

       liminf_{n -> infinity} lambda_2(G_n) >= 2*sqrt(d-1)

   A d-regular graph is called *Ramanujan* if it meets this bound at finite
   n, i.e. lambda(G) = max_{i>=2} |lambda_i(G)| <= 2*sqrt(d-1). This makes
   2*sqrt(d-1) the best possible asymptotic spectral gap a family of
   bounded-degree expanders can have.

2. Friedman's theorem (Friedman, "A proof of Alon's second eigenvalue
   conjecture and related problems", Memoirs of the AMS 195, 2008; earlier
   announced 2003): a uniformly random d-regular graph on n vertices is
   *asymptotically almost Ramanujan* -- for every eps > 0,

       P( lambda(G_n) <= 2*sqrt(d-1) + eps )  ->  1   as n -> infinity.

   This project's experiment (experiment.py) tests both predictions purely
   computationally: does the empirical distribution of lambda(G) for random
   d-regular graphs (a) stay at or above 2*sqrt(d-1) (per Alon-Boppana,
   asymptotically) and (b) concentrate ever closer to 2*sqrt(d-1) as n grows
   (per Friedman)?
"""
from __future__ import annotations

import math


def alon_boppana_bound(d: int) -> float:
    """The Alon-Boppana constant 2*sqrt(d-1) for d-regular graphs."""
    if d < 2:
        raise ValueError("the Alon-Boppana bound requires d >= 2")
    return 2.0 * math.sqrt(d - 1)


def is_ramanujan(lambda2_abs: float, d: int, tol: float = 1e-9) -> bool:
    """Whether a graph with this lambda(G) value meets the Ramanujan bound."""
    return lambda2_abs <= alon_boppana_bound(d) + tol


def within_epsilon_of_bound(lambda2_abs: float, d: int, eps: float) -> bool:
    """Friedman-style near-Ramanujan check: lambda(G) <= 2*sqrt(d-1) + eps."""
    return lambda2_abs <= alon_boppana_bound(d) + eps


def close_to_bound(lambda2_abs: float, d: int, eps: float) -> bool:
    """Two-sided concentration check: |lambda(G) - 2*sqrt(d-1)| <= eps.

    Unlike within_epsilon_of_bound (which only rules out overshooting the
    bound), this also requires lambda(G) not to sit far *below* the bound
    either -- the metric that actually tracks the empirical finding here,
    since mean lambda(G) approaches 2*sqrt(d-1) from below as n grows."""
    return abs(lambda2_abs - alon_boppana_bound(d)) <= eps
