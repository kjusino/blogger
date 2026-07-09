"""Replay loop, pseudo-regret computation, and theoretical bound formulas."""

import math

import numpy as np


def run_bandit(algo, reward_table):
    """Replay ``algo`` against a precomputed (T, K) full-information table.

    The algorithm only ever sees ``reward_table[t, arm_played]`` — the
    bandit-feedback restriction — via ``select_arm``/``update``. Returns a
    dict with the per-round arm played, the reward received, the
    cumulative regret trace against the best fixed arm in hindsight
    (computed from the full table, which the algorithm never sees), and
    the final scalar regret.
    """
    horizon, k = reward_table.shape
    arms_played = np.empty(horizon, dtype=np.int64)
    rewards_received = np.empty(horizon, dtype=np.float64)

    for t in range(horizon):
        arm = algo.select_arm()
        reward = reward_table[t, arm]
        algo.update(arm, reward)
        arms_played[t] = arm
        rewards_received[t] = reward

    cumulative_arm_totals = np.cumsum(reward_table, axis=0)
    best_fixed_cumulative = cumulative_arm_totals.max(axis=1)
    algo_cumulative = np.cumsum(rewards_received)
    regret_trace = best_fixed_cumulative - algo_cumulative

    return {
        "arms_played": arms_played,
        "rewards_received": rewards_received,
        "regret_trace": regret_trace,
        "final_regret": float(regret_trace[-1]),
    }


def run_bandit_dynamic(algo, reward_table, mean_table):
    """Replay ``algo`` and score it against the best arm *at each round*.

    This is "dynamic" or "tracking" pseudo-regret: at every round t it
    compares the true mean of the arm played to the true mean of whichever
    arm is best at that specific t (``mean_table[t].max()``), using the
    environment's *expected* rewards rather than realized noisy outcomes
    for both sides of the comparison. In a stationary environment the
    per-round best arm never changes, so this coincides with the
    realized-reward weak regret from ``run_bandit``. In a non-stationary
    (switching) environment it is a strictly harder, more standard
    benchmark than "best single fixed arm over the whole horizon" — see
    the module docstring in ``src/experiment.py`` for why that fixed-arm
    benchmark turned out to be too weak to be informative here.
    """
    horizon, k = reward_table.shape
    arms_played = np.empty(horizon, dtype=np.int64)
    rewards_received = np.empty(horizon, dtype=np.float64)

    for t in range(horizon):
        arm = algo.select_arm()
        reward = reward_table[t, arm]
        algo.update(arm, reward)
        arms_played[t] = arm
        rewards_received[t] = reward

    per_round_best_mean = mean_table.max(axis=1)
    per_round_played_mean = mean_table[np.arange(horizon), arms_played]
    regret_trace = np.cumsum(per_round_best_mean - per_round_played_mean)

    return {
        "arms_played": arms_played,
        "rewards_received": rewards_received,
        "regret_trace": regret_trace,
        "final_regret": float(regret_trace[-1]),
    }


def exp3_bound(n_arms, horizon):
    """Auer et al. (2002) Corollary 3.2 expected weak-regret bound."""
    return 2.0 * math.sqrt(math.e - 1) * math.sqrt(n_arms * horizon * math.log(n_arms))


def ucb1_bound(gaps, horizon):
    """Auer, Cesa-Bianchi, Fischer (2002) Theorem 1 expected regret bound.

    ``gaps`` are the suboptimality gaps Delta_i > 0 of the non-best arms.
    """
    gaps = np.asarray(gaps, dtype=np.float64)
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return 0.0
    return float(np.sum(8.0 * math.log(horizon) / gaps + (1.0 + math.pi ** 2 / 3.0) * gaps))


def fit_power_law(x, y):
    """Least-squares fit of y = c * x^a via log(y) = a*log(x) + log(c).

    Returns (exponent, r_squared). Non-positive y values are dropped
    before the fit (regret can be exactly 0 at t=0).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = (x > 0) & (y > 0)
    log_x, log_y = np.log(x[mask]), np.log(y[mask])
    exponent, intercept = np.polyfit(log_x, log_y, 1)
    predicted = exponent * log_x + intercept
    ss_res = np.sum((log_y - predicted) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(exponent), float(r_squared)


def fit_log_law(x, y):
    """Least-squares fit of y = a * ln(x) + b. Returns (a, b, r_squared)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = x > 0
    log_x, y = np.log(x[mask]), y[mask]
    a, b = np.polyfit(log_x, y, 1)
    predicted = a * log_x + b
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(a), float(b), float(r_squared)
