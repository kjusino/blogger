"""Closed-form BBP (Baik-Ben Arous-Peche) phase transition predictions.

For the rank-one spiked covariance model Sigma = I_p + lam * v v^T with
n i.i.d. Gaussian samples and aspect ratio c = p/n -> (0, 1) fixed as
n, p -> infinity, the sample covariance S = (1/n) X^T X obeys (Baik,
Ben Arous & Peche 2005 for the complex/Wishart case; Baik & Silverstein
2006 and Paul 2007 for the real spiked-covariance case):

  * Marchenko-Pastur bulk edge (lam = 0):  (1 + sqrt(c))^2
  * BBP threshold:                          lam* = sqrt(c)

  For lam <= lam*  ("subcritical"):
      top eigenvalue      -> (1 + sqrt(c))^2      (merges into the MP bulk)
      |<u_hat, v>|^2       -> 0                    (no detectable alignment)

  For lam > lam*  ("supercritical"):
      top eigenvalue      -> (1 + lam) * (1 + c / lam)
      |<u_hat, v>|^2       -> (1 - c / lam^2) / (1 + c / lam)

Both quantities are continuous at lam = lam* (verified in tests/test_theory.py),
which is the standard consistency check for this formula.
"""

from __future__ import annotations

import numpy as np


def bbp_threshold(c: float) -> float:
    if c <= 0:
        raise ValueError(f"c must be > 0, got {c}")
    return np.sqrt(c)


def mp_edge(c: float) -> float:
    if c <= 0:
        raise ValueError(f"c must be > 0, got {c}")
    return (1.0 + np.sqrt(c)) ** 2


def theoretical_top_eigenvalue(lam, c: float):
    """Asymptotic top sample-eigenvalue as a function of spike strength lam."""
    lam = np.asarray(lam, dtype=float)
    if np.any(lam < 0):
        raise ValueError("lam must be >= 0")
    thr = bbp_threshold(c)
    edge = mp_edge(c)
    safe_lam = np.where(lam > 0, lam, 1.0)
    supercritical = (1.0 + lam) * (1.0 + c / safe_lam)
    return np.where(lam <= thr, edge, supercritical)


def theoretical_alignment_sq(lam, c: float):
    """Asymptotic squared cosine alignment |<u_hat, v>|^2 between the sample
    top eigenvector and the true spike direction, as a function of lam."""
    lam = np.asarray(lam, dtype=float)
    if np.any(lam < 0):
        raise ValueError("lam must be >= 0")
    thr = bbp_threshold(c)
    safe_lam = np.where(lam > 0, lam, 1.0)
    supercritical = (1.0 - c / safe_lam**2) / (1.0 + c / safe_lam)
    supercritical = np.clip(supercritical, 0.0, 1.0)
    return np.where(lam <= thr, 0.0, supercritical)
