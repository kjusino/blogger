"""Monte Carlo sweeps that compare empirical spiked-covariance statistics
against the theoretical BBP predictions."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats

from .model import sample_spiked_covariance_data, sample_covariance, top_eigenpair
from .theory import bbp_threshold, theoretical_top_eigenvalue, theoretical_alignment_sq


@dataclass
class TrialAggregate:
    c: float
    n: int
    p: int
    lam: float
    lam_over_threshold: float
    trials: int
    mean_eig: float
    std_eig: float
    ci95_eig: float
    theory_eig: float
    rel_err_eig: float
    mean_align: float
    std_align: float
    ci95_align: float
    theory_align: float
    abs_err_align: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_trials(n: int, p: int, lam: float, trials: int, rng: np.random.Generator):
    """Run `trials` independent draws, returning arrays of top eigenvalues
    and squared alignments to the true spike direction."""
    eigs = np.empty(trials)
    aligns = np.empty(trials)
    for t in range(trials):
        X, v = sample_spiked_covariance_data(n, p, lam, rng)
        S = sample_covariance(X)
        eig, u = top_eigenpair(S)
        eigs[t] = eig
        aligns[t] = float(np.dot(u, v)) ** 2
    return eigs, aligns


def _ci95(values: np.ndarray) -> float:
    """Half-width of a 95% CI on the mean, via the t-distribution."""
    n = len(values)
    if n < 2:
        return float("nan")
    se = values.std(ddof=1) / np.sqrt(n)
    return float(stats.t.ppf(0.975, df=n - 1) * se)


def sweep_lambda_grid(
    p: int,
    c_values,
    lam_ratios,
    trials: int,
    seed: int,
) -> list[TrialAggregate]:
    """For each aspect ratio c in c_values, sweep lam over lam_ratios *
    bbp_threshold(c), so that every c is probed at the same *relative*
    distance from its own threshold."""
    rng = np.random.default_rng(seed)
    results = []
    for c in c_values:
        n = max(int(round(p / c)), p + 1)
        thr = bbp_threshold(c)
        for ratio in lam_ratios:
            lam = ratio * thr
            eigs, aligns = run_trials(n, p, lam, trials, rng)
            theory_eig = float(theoretical_top_eigenvalue(lam, c))
            theory_align = float(theoretical_alignment_sq(lam, c))
            results.append(
                TrialAggregate(
                    c=c,
                    n=n,
                    p=p,
                    lam=lam,
                    lam_over_threshold=ratio,
                    trials=trials,
                    mean_eig=float(eigs.mean()),
                    std_eig=float(eigs.std(ddof=1)),
                    ci95_eig=_ci95(eigs),
                    theory_eig=theory_eig,
                    rel_err_eig=abs(float(eigs.mean()) - theory_eig) / theory_eig,
                    mean_align=float(aligns.mean()),
                    std_align=float(aligns.std(ddof=1)),
                    ci95_align=_ci95(aligns),
                    theory_align=theory_align,
                    abs_err_align=abs(float(aligns.mean()) - theory_align),
                )
            )
    return results


def estimate_detection_threshold(
    p: int, c: float, lam_ratios_fine, trials: int, seed: int, crossing: float = 0.05
):
    """Sweep a fine lam grid for a single c and locate, by linear
    interpolation of the empirical mean alignment curve, the lam at which
    the alignment first exceeds `crossing`. Returns (lam_hat, thr_theory)."""
    rng = np.random.default_rng(seed)
    n = max(int(round(p / c)), p + 1)
    thr = bbp_threshold(c)
    lams = np.array([r * thr for r in lam_ratios_fine])
    mean_aligns = np.empty(len(lams))
    for i, lam in enumerate(lams):
        _, aligns = run_trials(n, p, lam, trials, rng)
        mean_aligns[i] = aligns.mean()

    order = np.argsort(lams)
    lams_sorted = lams[order]
    aligns_sorted = mean_aligns[order]

    lam_hat = None
    for i in range(1, len(lams_sorted)):
        if aligns_sorted[i - 1] < crossing <= aligns_sorted[i]:
            x0, x1 = lams_sorted[i - 1], lams_sorted[i]
            y0, y1 = aligns_sorted[i - 1], aligns_sorted[i]
            lam_hat = x0 + (crossing - y0) * (x1 - x0) / (y1 - y0)
            break
    return lam_hat, thr, lams_sorted, aligns_sorted


def finite_size_scaling(
    c: float, lam_ratio: float, p_values, trials: int, seed: int
):
    """Fix (c, lam/threshold), vary p (with n = p/c), and track how the
    empirical top eigenvalue's error against the asymptotic theory shrinks
    as the system grows."""
    rng = np.random.default_rng(seed)
    thr = bbp_threshold(c)
    lam = lam_ratio * thr
    theory_eig = float(theoretical_top_eigenvalue(lam, c))
    rows = []
    for p in p_values:
        n = max(int(round(p / c)), p + 1)
        eigs, _ = run_trials(n, p, lam, trials, rng)
        rows.append(
            {
                "p": p,
                "n": n,
                "c": c,
                "lam": lam,
                "lam_over_threshold": lam_ratio,
                "mean_eig": float(eigs.mean()),
                "std_eig": float(eigs.std(ddof=1)),
                "theory_eig": theory_eig,
                "abs_err_eig": abs(float(eigs.mean()) - theory_eig),
            }
        )
    return rows
