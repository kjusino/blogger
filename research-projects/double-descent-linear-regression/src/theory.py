"""Closed-form finite-sample and asymptotic risk of the min-norm/OLS
estimator under isotropic Gaussian design.

Derivation (see README for the full write-up). Model: x ~ N(0, I_p),
y = x.beta0 + eps, eps ~ N(0, sigma2), ||beta0||^2 = r2.

Underparameterized (p < n): unique OLS solution, unbiased since beta0 lies
exactly in the span of the p features. Its excess risk is pure variance:

    R(n, p) = sigma2 * p / (n - p - 1)          [requires n - p - 1 > 0]

which follows from E[tr((X^T X)^-1)] = p / (n - p - 1) for a p x p Wishart
matrix with n degrees of freedom.

Overparameterized (p > n): the min-norm interpolator is
    beta_hat = X^+ y = P_row(X) beta0 + X^T (X X^T)^-1 eps
where P_row(X) projects onto the (random, n-dimensional) row space of X.
Its risk decomposes into two exact, uncorrelated terms:

    bias^2   = E[||(I - P_row(X)) beta0||^2] = r2 * (1 - n / p)
               (row(X) is a uniformly random n-dim subspace of R^p, so the
               expected squared component of any fixed unit vector outside
               it is exactly the dimension ratio 1 - n/p)

    variance = sigma2 * E[tr((X X^T)^-1)] = sigma2 * n / (p - n - 1)
               (X X^T is an n x n Wishart matrix with p degrees of freedom)

    R(n, p) = sigma2 * n / (p - n - 1) + r2 * (1 - n / p)   [requires p - n - 1 > 0]

Both formulas are exact for any finite (n, p) satisfying the stated
constraint -- they are not large-sample approximations. Their gamma = p/n
limits (gamma < 1 and gamma > 1 respectively) give the familiar asymptotic
double-descent curve, which diverges as gamma -> 1 from either side.
"""

import numpy as np


def exact_risk(n: int, p: int, r2: float, sigma2: float) -> float:
    if p < n:
        if n - p - 1 <= 0:
            return np.inf
        return sigma2 * p / (n - p - 1)
    elif p > n:
        if p - n - 1 <= 0:
            return np.inf
        return sigma2 * n / (p - n - 1) + r2 * (1 - n / p)
    else:
        return np.inf


def bias_variance_exact(n: int, p: int, r2: float, sigma2: float):
    """Return (bias_squared, variance) matching `exact_risk`'s decomposition."""
    if p <= n:
        bias2 = 0.0
        variance = sigma2 * p / (n - p - 1) if n - p - 1 > 0 else np.inf
    else:
        bias2 = r2 * (1 - n / p)
        variance = sigma2 * n / (p - n - 1) if p - n - 1 > 0 else np.inf
    return bias2, variance


def asymptotic_risk(gamma: float, r2: float, sigma2: float) -> float:
    """gamma = p / n limit of `exact_risk`."""
    if gamma < 1:
        return sigma2 * gamma / (1 - gamma)
    elif gamma > 1:
        return sigma2 / (gamma - 1) + r2 * (1 - 1 / gamma)
    else:
        return np.inf
