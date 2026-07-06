"""Experiment grid: does Hoeffding-bound-driven query redundancy let L*
recover its noiseless learning-success rate under a persistently noisy
membership oracle, and at what query-complexity cost?"""

from __future__ import annotations

import math
import random

from .dfa import DFA, equivalent, random_dfa
from .lstar import learn, LStarNonConvergence
from .oracle import (
    ExactEquivalenceOracle,
    NoisyMembershipOracle,
    RedundantMembershipOracle,
    hoeffding_repetitions,
)

STRATEGIES = ("none", "fixed5", "adaptive")
DELTA_Q = 1e-4  # per-query target failure probability for the adaptive strategy


def make_targets(sizes=(5, 7, 10, 13), alphabet_size=2, master_seed=2024):
    """Deterministically generate one random target DFA per size in `sizes`."""
    targets = []
    for i, n in enumerate(sizes):
        rng = random.Random(master_seed * 1000 + i)
        dfa = random_dfa(n, alphabet_size, rng)
        targets.append({"target_id": i, "target_num_states": n, "dfa": dfa})
    return targets


def build_oracle(target: DFA, noise_rate: float, strategy: str, rng: random.Random):
    noisy = NoisyMembershipOracle(target, noise_rate, rng)
    if strategy == "none":
        return RedundantMembershipOracle(noisy, repetitions=1)
    if strategy == "fixed5":
        return RedundantMembershipOracle(noisy, repetitions=5)
    if strategy == "adaptive":
        return RedundantMembershipOracle(
            noisy, repetitions_fn=lambda p: hoeffding_repetitions(p, DELTA_Q)
        )
    raise ValueError(f"unknown strategy {strategy!r}")


def run_single(target_row: dict, noise_rate: float, strategy: str, seed: int,
               max_equivalence_queries: int = 200, max_states_factor: int = 8):
    target = target_row["dfa"]
    rng = random.Random(seed)
    oracle = build_oracle(target, noise_rate, strategy, rng)
    equivalence = ExactEquivalenceOracle(target)

    theoretical_k = hoeffding_repetitions(noise_rate, DELTA_Q)

    record = {
        "target_id": target_row["target_id"],
        "target_num_states": target_row["target_num_states"],
        "noise_rate": noise_rate,
        "strategy": strategy,
        "seed": seed,
        "theoretical_k": theoretical_k,
    }

    try:
        hypothesis, stats = learn(
            oracle, equivalence, alphabet_size=target.alphabet_size,
            max_states=max(30, max_states_factor * target_row["target_num_states"]),
            max_equivalence_queries=max_equivalence_queries,
        )
        success = equivalent(hypothesis, target)
        record.update({
            "success": success,
            "hypothesis_num_states": hypothesis.num_states,
            "equivalence_queries": stats["equivalence_queries"],
            "converged": True,
        })
    except LStarNonConvergence:
        record.update({
            "success": False,
            "hypothesis_num_states": None,
            "equivalence_queries": None,
            "converged": False,
        })

    record["distinct_queries"] = oracle.distinct_queries
    record["raw_queries"] = oracle.raw_queries
    return record


def run_grid(noise_rates, strategies=STRATEGIES, sizes=(5, 7, 10, 13),
             num_seeds=10, seed_offset=0):
    targets = make_targets(sizes=sizes)
    results = []
    for target_row in targets:
        for noise_rate in noise_rates:
            for strategy in strategies:
                for s in range(num_seeds):
                    seed = seed_offset + s
                    results.append(run_single(target_row, noise_rate, strategy, seed))
    return results
