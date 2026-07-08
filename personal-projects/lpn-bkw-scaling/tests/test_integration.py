import random

from lpn_bkw.bkw import attack
from lpn_bkw.experiment import run_config, find_required_confidence_const
from lpn_bkw import theory


def test_end_to_end_attack_recovers_known_secret_noiseless():
    n, b, tau = 12, 4, 0.0
    secret = 0b110010101101
    res = attack(n, b, tau, secret=secret, rng=random.Random(0))
    assert res["success"]
    assert res["secret_hat"] == secret
    assert res["hamming_distance"] == 0


def test_end_to_end_attack_high_success_rate_with_calibrated_budget():
    # A comfortably feasible configuration (a=2 elimination levels): the
    # theory-prescribed query budget at a generous confidence constant
    # should recover the secret almost always.
    n, b, tau = 12, 6, 0.1
    summary = run_config(n, b, tau, trials=25, confidence_const=150, seed=123)
    assert summary["success_rate"] >= 0.9
    assert summary["mean_total_queries"] > 0


def test_success_rate_degrades_with_deeper_elimination_at_fixed_budget():
    # Same window size and confidence constant, but more elimination levels
    # (bigger n) should make the theory-prescribed budget less reliable,
    # since the required-samples formula assumes independence that the
    # pivot-and-eliminate reduction does not fully deliver.
    b, tau, c = 4, 0.1, 20
    shallow = run_config(n=8, b=b, tau=tau, trials=25, confidence_const=c, seed=1)
    deep = run_config(n=20, b=b, tau=tau, trials=25, confidence_const=c, seed=2)
    assert shallow["success_rate"] >= deep["success_rate"]


def test_find_required_confidence_const_returns_reasonable_value():
    n, b, tau = 12, 6, 0.1
    required, trace = find_required_confidence_const(
        n, b, tau, candidate_consts=[20, 50, 100, 200], trials=15, target_success=0.85, seed=5,
    )
    assert required is not None
    assert any(c == required for c, _ in trace)


def test_theory_module_agrees_with_empirical_query_budget():
    n, b, tau, c = 16, 8, 0.1, 100
    summary = run_config(n, b, tau, trials=5, confidence_const=c, seed=9)
    predicted = theory.total_queries(n, b, tau, confidence_const=c)
    # mean_total_queries is the realized sum of the (deterministic, formula
    # driven) per-pass budgets, so it should match the theoretical value
    # up to integer rounding across a_levels passes.
    a_levels = n // b
    assert abs(summary["mean_total_queries"] - predicted) <= a_levels
