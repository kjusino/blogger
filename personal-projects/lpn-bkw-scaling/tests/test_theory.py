import math

import pytest

from lpn_bkw import theory


def test_bias_endpoints():
    assert theory.bias(0.0) == 1.0
    assert theory.bias(0.25) == 0.5
    assert abs(theory.bias(0.5)) < 1e-12


def test_final_bias_decreases_with_more_levels():
    tau = 0.1
    b1 = theory.final_bias(tau, a=1)
    b2 = theory.final_bias(tau, a=2)
    b3 = theory.final_bias(tau, a=3)
    assert b1 > b2 > b3 > 0
    # a=1 means zero elimination levels -> bias unchanged.
    assert b1 == pytest.approx(theory.bias(tau))
    # each extra level squares the bias.
    assert b2 == pytest.approx(b1 ** 2)
    assert b3 == pytest.approx(b2 ** 2)


def test_required_final_samples_grows_as_bias_shrinks():
    tau = 0.1
    m2 = theory.required_final_samples(tau, a=2)
    m4 = theory.required_final_samples(tau, a=4)
    m6 = theory.required_final_samples(tau, a=6)
    assert m2 <= m4 <= m6


def test_required_final_samples_matches_formula_above_floor():
    tau = 0.2
    a = 3
    c = 20.0
    expected = math.ceil(c / theory.final_bias(tau, a) ** 2)
    got = theory.required_final_samples(tau, a, confidence_const=c, floor=1)
    assert got == expected


def test_queries_per_pass_rejects_non_divisor():
    with pytest.raises(ValueError):
        theory.queries_per_pass(10, 3, 0.1)


def test_total_queries_is_a_passes_of_per_pass_cost():
    n, b, tau = 16, 4, 0.1
    per_pass = theory.queries_per_pass(n, b, tau)
    total = theory.total_queries(n, b, tau)
    assert total == pytest.approx(per_pass * (n // b))


def test_optimal_b_picks_a_divisor_of_n():
    n, tau = 24, 0.1
    candidates = [2, 3, 4, 6, 8, 12]
    best = theory.optimal_b(n, tau, candidates)
    assert best in candidates
    assert n % best == 0


def test_optimal_b_raises_when_no_candidate_divides_n():
    with pytest.raises(ValueError):
        theory.optimal_b(10, 0.1, [3, 7])
