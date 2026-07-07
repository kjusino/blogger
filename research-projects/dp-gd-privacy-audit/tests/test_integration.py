"""Ground-truth validation: at T=1, the mechanism's effect on
theta_final[0] - theta0[0] differs between IN/OUT worlds by exactly a mean
shift of lr*C with common std lr*sigma*C -- a textbook Gaussian mechanism
with "effective sigma" equal to the same sigma parameter used by the RDP
accountant (the lr*C scaling cancels in the signal-to-noise ratio).

This means the T=1 empirical eps_lower (Monte Carlo audit) must satisfy, as
N grows:
  - eps_lower <= epsilon_theory + small_slack  (ALWAYS -- a hard DP
    correctness invariant; if this fails, there's a real bug in the
    accountant, mechanism, or audit code)
  - eps_lower > 0.3 * epsilon_theory  (sanity that the attack isn't
    trivially weak at T=1)
"""
import numpy as np
import pytest

from src.accountant import epsilon_from_rdp
from src.audit import run_membership_audit
from src.data import make_dataset

SLACK = 0.5  # small additive slack for Monte Carlo / threshold-sweep noise


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_t1_audit_recovers_theory_within_bounds(sigma):
    delta = 1e-5
    C = 1.0
    lr = 0.1
    N = 4000

    data_rng = np.random.default_rng(0)
    X, y = make_dataset(n=100, d=5, rng=data_rng)
    theta0 = np.zeros(5)

    epsilon_theory, _ = epsilon_from_rdp(sigma, T=1, delta=delta)

    trial_rng = np.random.default_rng(hash(("t1_audit", sigma)) % (2 ** 32))
    result = run_membership_audit(X, y, theta0, T=1, C=C, sigma=sigma, lr=lr, N=N, delta=delta, rng=trial_rng)
    eps_lower = result["eps_lower"]

    assert result["mean_shift_is_negative"]
    assert eps_lower <= epsilon_theory + SLACK, (
        f"eps_lower ({eps_lower}) exceeded epsilon_theory ({epsilon_theory}) + slack "
        f"-- this violates a hard DP correctness invariant, indicating a real bug."
    )
    assert eps_lower > 0.3 * epsilon_theory, (
        f"eps_lower ({eps_lower}) is too weak relative to epsilon_theory ({epsilon_theory}) "
        f"-- the audit attack should be reasonably tight at T=1."
    )
