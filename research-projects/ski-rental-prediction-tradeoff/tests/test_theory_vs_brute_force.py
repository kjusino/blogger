import pytest

from src.theory import (
    robustness_formula,
    consistency_formula,
    robustness_tolerance,
    consistency_tolerance,
)
from src.brute_force import brute_force_robustness, brute_force_consistency

B_GRID = [10, 50, 200, 1000]
LAMBDA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@pytest.mark.parametrize("b", B_GRID)
@pytest.mark.parametrize("lam", LAMBDA_GRID)
def test_robustness_theory_matches_brute_force(b, lam):
    theory = robustness_formula(lam, b)
    bf, bf_x, bf_y = brute_force_robustness(lam, b)
    tol = robustness_tolerance(lam, b)
    assert abs(theory - bf) <= tol, (
        f"Robustness mismatch at b={b}, lam={lam}: theory={theory}, "
        f"brute_force={bf} (argmax x={bf_x}, y={bf_y}), tol={tol}"
    )


@pytest.mark.parametrize("b", B_GRID)
@pytest.mark.parametrize("lam", LAMBDA_GRID)
def test_consistency_theory_matches_brute_force(b, lam):
    theory = consistency_formula(lam, b)
    bf, bf_x = brute_force_consistency(lam, b)
    tol = consistency_tolerance(lam, b)
    assert abs(theory - bf) <= tol, (
        f"Consistency mismatch at b={b}, lam={lam}: theory={theory}, "
        f"brute_force={bf} (argmax x={bf_x}), tol={tol}"
    )


class TestClassicalBoundSanityCheck:
    def test_robustness_at_lambda_zero_approaches_two(self):
        # This is the core, must-not-be-broken sanity check: the
        # no-predictions (lambda=0) strategy is the classical tight
        # 2-competitive deterministic algorithm.
        for b in [10, 100, 1000, 10000]:
            r = robustness_formula(0.0, b)
            assert r < 2.0
            assert r == pytest.approx(2.0, abs=2.0 / b + 1e-6)

    def test_robustness_is_monotonically_worse_with_larger_b_at_lambda_zero(self):
        vals = [robustness_formula(0.0, b) for b in [10, 100, 1000, 10000]]
        assert all(v1 <= v2 for v1, v2 in zip(vals, vals[1:]))

    def test_robustness_unbounded_at_lambda_one(self):
        # lambda=1 (blind trust): Robustness(1, b) = b -- no worst-case
        # guarantee at all as b grows.
        for b in [10, 100, 1000]:
            assert robustness_formula(1.0, b) == pytest.approx(float(b))

    def test_consistency_perfect_at_lambda_one(self):
        for b in [10, 100, 1000]:
            assert consistency_formula(1.0, b) == pytest.approx(1.0)

    def test_consistency_approaches_two_at_lambda_zero(self):
        for b in [10, 100, 1000, 10000]:
            c = consistency_formula(0.0, b)
            assert c == pytest.approx(2.0, abs=2.0 / b + 1e-6)


class TestBruteForceInternalConsistency:
    def test_brute_force_robustness_argmax_x_equals_tau_star(self):
        # From the derivation in theory.py: the worst-case x for Robustness
        # is exactly x = tau_star (achieved at y=0 or any y>=b).
        from src.theory import tau_star

        for b in [10, 50, 200]:
            for lam in [0.0, 0.3, 0.7, 1.0]:
                _, bf_x, bf_y = brute_force_robustness(lam, b)
                assert bf_x == tau_star(lam, b)

    def test_brute_force_ratio_never_below_one(self):
        for b in [10, 50]:
            for lam in [0.0, 0.5, 1.0]:
                r, _, _ = brute_force_robustness(lam, b)
                assert r >= 1.0
                c, _ = brute_force_consistency(lam, b)
                assert c >= 1.0
