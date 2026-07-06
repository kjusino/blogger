"""Core experiment logic: measuring empirical sample complexity of the two
testers in testers.py and fitting power-law scaling exponents in n.

Everything here is orchestration over the primitives in distributions.py,
testers.py and theory.py — no numerical claims are hard-coded; all numbers
come from actually drawing samples and running the testers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .distributions import FAMILIES
from .testers import TESTERS
from .theory import PREDICTED_M


def generate_samples(
    family: str, n: int, epsilon: float, m: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw m i.i.d. samples from the given family ("uniform" or a key of
    distributions.FAMILIES) over domain size n."""
    if family == "uniform":
        return rng.integers(0, n, size=m)
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    p = FAMILIES[family](n, epsilon, rng)
    return rng.choice(n, size=m, p=p)


def power_at_m(
    tester_name: str,
    family: str,
    n: int,
    epsilon: float,
    m: int,
    trials: int,
    rng: np.random.Generator,
) -> float:
    """Fraction of `trials` independent draws of m samples for which the
    named tester rejects H0 (declares "far")."""
    tester = TESTERS[tester_name]
    rejects = 0
    for _ in range(trials):
        samples = generate_samples(family, n, epsilon, m, rng)
        if tester(samples, n, epsilon):
            rejects += 1
    return rejects / trials


@dataclass
class M50Result:
    n: int
    epsilon: float
    tester: str
    family: str
    m50: float
    false_positive_rate: float
    m_grid: list = field(default_factory=list)
    power_grid: list = field(default_factory=list)
    advantage_grid: list = field(default_factory=list)
    bracketed: bool = True


def find_m50(
    tester_name: str,
    family: str,
    n: int,
    epsilon: float,
    trials: int,
    rng: np.random.Generator,
    start_m: int = 2,
    max_doublings: int = 24,
    advantage_threshold: float = 0.5,
) -> M50Result:
    """Find the smallest sample size m at which the named tester actually
    *discriminates* family from uniform, via doubling search on the
    discrimination advantage

        advantage(m) = P[reject | family] - P[reject | uniform]

    rather than on raw rejection power. Raw power alone is misleading at
    very small m: with m << n, an empirical histogram is sparse regardless
    of the true source, so a naive L1-threshold tester rejects *both*
    uniform and far distributions with high probability there (both TPR and
    FPR are near 1) — a spurious "detection" with no real signal. Requiring
    the *advantage* (TPR - FPR) to cross 0.5 correctly discounts that
    regime, since advantage is near 0 until the tester actually starts
    telling the two hypotheses apart.

    m_grid/power_grid/advantage_grid record the doubling trajectory for
    plotting; theory.PREDICTED_M is used only for reporting, never to steer
    this search.
    """
    m_grid, power_grid, advantage_grid = [], [], []
    m = start_m
    prev_m, prev_advantage, prev_fpr = None, None, None
    m50 = None
    fpr = None
    bracketed = False
    for _ in range(max_doublings):
        power = power_at_m(tester_name, family, n, epsilon, m, trials, rng)
        fpr_here = power_at_m(tester_name, "uniform", n, epsilon, m, trials, rng)
        advantage = power - fpr_here
        m_grid.append(m)
        power_grid.append(power)
        advantage_grid.append(advantage)
        if advantage >= advantage_threshold:
            if prev_m is None:
                m50, fpr = float(m), fpr_here  # can't bracket from below start_m
            else:
                t = (advantage_threshold - prev_advantage) / (advantage - prev_advantage)
                log_m0, log_m1 = np.log(prev_m), np.log(m)
                m50 = float(np.exp(log_m0 + t * (log_m1 - log_m0)))
                fpr = prev_fpr + t * (fpr_here - prev_fpr)
                bracketed = True
            break
        prev_m, prev_advantage, prev_fpr = m, advantage, fpr_here
        m *= 2
    if m50 is None:
        # Advantage never reached the threshold within max_doublings; report
        # the boundary so it still plots, flagged via bracketed=False.
        m50, fpr = float(m), fpr_here

    return M50Result(
        n=n,
        epsilon=epsilon,
        tester=tester_name,
        family=family,
        m50=m50,
        false_positive_rate=fpr,
        m_grid=m_grid,
        power_grid=power_grid,
        advantage_grid=advantage_grid,
        bracketed=bracketed,
    )


def fit_power_law(ns, m50s):
    """Fit log(m50) = slope*log(n) + intercept by least squares. Returns
    (slope, intercept, r_squared)."""
    log_n = np.log(np.asarray(ns, dtype=float))
    log_m = np.log(np.asarray(m50s, dtype=float))
    slope, intercept = np.polyfit(log_n, log_m, 1)
    pred = slope * log_n + intercept
    ss_res = np.sum((log_m - pred) ** 2)
    ss_tot = np.sum((log_m - np.mean(log_m)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r_squared)


def power_curve(
    tester_name: str,
    family: str,
    n: int,
    epsilon: float,
    m_grid,
    trials: int,
    rng: np.random.Generator,
):
    """Power of `tester_name` against `family` at each m in m_grid."""
    return [
        power_at_m(tester_name, family, n, epsilon, int(m), trials, rng)
        for m in m_grid
    ]
