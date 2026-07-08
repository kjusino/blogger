"""Fit the conjectured 1/log(X) decay of the consecutive-prime digit bias.

Lemke Oliver and Soundararajan's heuristic (built on the Hardy-Littlewood
k-tuple conjecture) predicts that the deviation of the same-last-digit
frequency from its asymptotic value of 1/4 shrinks like c / ln(X) as the
scale X grows -- slowly enough that the bias remains large and detectable
even at astronomically large X, which is why it went unnoticed until 2016
despite primes being studied for centuries.
"""

from typing import Dict

import numpy as np
from scipy import stats


def fit_inverse_log_decay(scales: np.ndarray, biases: np.ndarray) -> Dict[str, float]:
    """Ordinary least squares fit of bias ~ slope * (1 / ln(scale)) + intercept.

    A slope close to the conjectured proportionality constant and a
    positive, statistically significant slope both count as evidence for
    the 1/log(X) decay law; an intercept close to 0 is expected since the
    bias should vanish as scale -> infinity (1/ln(scale) -> 0).
    """
    x = 1.0 / np.log(scales)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, biases)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "stderr": float(std_err),
    }
