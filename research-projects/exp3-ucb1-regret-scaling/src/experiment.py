"""Grid runners tying bandits + environments + regret together.

Every grid point uses a *fresh* algorithm instance tuned for the horizon
of that specific run (EXP3's gamma depends on the target T). This costs
more compute than reusing checkpoints from one long run, but keeps every
data point directly comparable to the theoretical, per-horizon bound
formulas in ``src/regret.py`` — no approximation from reusing a
longer-horizon tuning at an earlier checkpoint.

Two environment families are used for two different questions:

* ``StochasticBernoulliEnv`` (one persistently-best arm, i.i.d. rewards)
  drives the EXP3 scaling-law experiments (``exp3_scaling_over_T`` /
  ``exp3_scaling_over_K``) and the "who wins on friendly ground"
  comparison. Because there is a single, never-changing best arm, the
  best-fixed-arm-in-hindsight benchmark is unambiguous and every
  algorithm's regret against it is (in expectation) non-negative — this
  is what makes it a clean setting to check a scaling *exponent* against.

* ``SwitchingBernoulliEnv`` drives the "who wins on adversarial ground"
  comparison, scored with *dynamic* (per-round-best-arm) pseudo-regret
  via ``src/regret.run_bandit_dynamic`` rather than realized weak regret
  against a single best-fixed-arm. An earlier version of this project
  used the fixed-arm benchmark here and found it uninformative: by
  construction every arm in ``SwitchingBernoulliEnv`` is "good" for an
  equal share of rounds, so the fixed-arm benchmark is only barely above
  what *any* policy gets on average, and both algorithms "beat" it just
  by playing more than one arm — that comparison couldn't distinguish
  good tracking from no tracking at all. Dynamic regret (harder: it
  compares to the best arm at every single round, not one fixed arm over
  the whole horizon) is the standard notion in the non-stationary-bandit
  literature and is what ``algorithm_comparison("switching", ...)`` uses.
  See the README for the (counterintuitive) result.
"""

import numpy as np

from src.bandits import EXP3, UCB1
from src.environments import StochasticBernoulliEnv, SwitchingBernoulliEnv
from src.regret import run_bandit, run_bandit_dynamic


def _build_algo(algo_name, k, horizon, algo_rng):
    if algo_name == "EXP3":
        return EXP3(k, horizon, algo_rng)
    if algo_name == "UCB1":
        return UCB1(k)
    raise ValueError(algo_name)


def stochastic_run(means, horizon, seed, algo_name):
    env_rng = np.random.default_rng(seed * 2 + 1)
    algo_rng = np.random.default_rng(seed * 2 + 2)
    env = StochasticBernoulliEnv(means, env_rng)
    table = env.reward_table(horizon)
    algo = _build_algo(algo_name, len(means), horizon, algo_rng)
    return run_bandit(algo, table)


def switching_run(n_arms, horizon, delta, num_segments, seed, algo_name):
    """Dynamic (per-round-best-arm) pseudo-regret — see run_bandit_dynamic."""
    env_rng = np.random.default_rng(seed * 2 + 1)
    algo_rng = np.random.default_rng(seed * 2 + 2)
    segment_length = max(1, horizon // num_segments)
    env = SwitchingBernoulliEnv(n_arms, segment_length, delta, env_rng)
    table = env.reward_table(horizon)
    means = env.mean_table(horizon)
    algo = _build_algo(algo_name, n_arms, horizon, algo_rng)
    return run_bandit_dynamic(algo, table, means)


def _one_best_arm_means(n_arms, delta):
    means = np.full(n_arms, 0.5 - delta / 2.0)
    means[0] = 0.5 + delta / 2.0
    return means


def exp3_scaling_over_T(n_arms, t_values, delta, n_seeds, base_seed=1):
    """Fresh EXP3 runs on the stationary env for each T in t_values."""
    means = _one_best_arm_means(n_arms, delta)
    records = []
    for horizon in t_values:
        for seed in range(n_seeds):
            run_seed = base_seed * 100_000 + seed * 97 + horizon
            result = stochastic_run(means, horizon, run_seed, "EXP3")
            records.append({"K": n_arms, "T": horizon, "seed": seed, "regret": result["final_regret"]})
    return records


def exp3_scaling_over_K(k_values, horizon, delta, n_seeds, base_seed=2):
    """Fresh EXP3 runs on the stationary env for each K in k_values, fixed T."""
    records = []
    for n_arms in k_values:
        means = _one_best_arm_means(n_arms, delta)
        for seed in range(n_seeds):
            run_seed = base_seed * 100_000 + seed * 97 + n_arms
            result = stochastic_run(means, horizon, run_seed, "EXP3")
            records.append({"K": n_arms, "T": horizon, "seed": seed, "regret": result["final_regret"]})
    return records


def algorithm_comparison(env_kind, n_arms, t_values, delta, n_seeds, base_seed, num_segments=4):
    """Fresh EXP3 + UCB1 runs at each T in t_values, on a stochastic or switching env."""
    records = []
    means = _one_best_arm_means(n_arms, delta)
    for horizon in t_values:
        for algo_name in ("EXP3", "UCB1"):
            for seed in range(n_seeds):
                run_seed = base_seed * 100_000 + seed * 97 + horizon
                if env_kind == "stochastic":
                    result = stochastic_run(means, horizon, run_seed, algo_name)
                elif env_kind == "switching":
                    result = switching_run(n_arms, horizon, delta, num_segments, run_seed, algo_name)
                else:
                    raise ValueError(env_kind)
                records.append(
                    {
                        "env": env_kind,
                        "algo": algo_name,
                        "K": n_arms,
                        "T": horizon,
                        "seed": seed,
                        "regret": result["final_regret"],
                    }
                )
    return records
