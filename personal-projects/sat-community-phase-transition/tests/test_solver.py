import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cnf import CNF
from src.solver import solve


def _check_assignment(cnf, assignment):
    for clause in cnf.clauses:
        if any((assignment[abs(lit)] if lit > 0 else not assignment[abs(lit)]) for lit in clause):
            continue
        return False
    return True


def test_trivially_satisfiable_single_clause():
    cnf = CNF(n_vars=3, clauses=[(1, 2, 3)])
    result = solve(cnf)
    assert result.satisfiable
    assert _check_assignment(cnf, result.assignment)


def test_empty_clause_is_unsatisfiable():
    # A clause list containing an empty clause can't be built through the
    # generators (always length 3), but the solver must still handle it
    # correctly as a direct contradiction.
    cnf = CNF(n_vars=1, clauses=[()])
    result = solve(cnf)
    assert not result.satisfiable


def test_all_eight_sign_patterns_over_3_vars_is_unsatisfiable():
    # Every possible sign pattern of a clause over the same 3 variables:
    # no assignment can satisfy all 8, since each one rules out exactly the
    # single assignment it contradicts (this is the textbook maximally
    # constrained 3-CNF over 3 variables).
    clauses = [
        (1, 2, 3), (1, 2, -3), (1, -2, 3), (1, -2, -3),
        (-1, 2, 3), (-1, 2, -3), (-1, -2, 3), (-1, -2, -3),
    ]
    cnf = CNF(n_vars=3, clauses=clauses)
    result = solve(cnf)
    assert not result.satisfiable


def _brute_force_satisfiable(cnf):
    for bits in range(2 ** cnf.n_vars):
        assignment = {v: bool((bits >> (v - 1)) & 1) for v in range(1, cnf.n_vars + 1)}
        if _check_assignment(cnf, assignment):
            return True
    return False


def test_solver_matches_brute_force_oracle_on_random_small_instances():
    import random

    from src.cnf import random_3sat

    rng = random.Random(123)
    for trial in range(40):
        n = rng.randint(3, 8)
        m = rng.randint(1, 25)
        cnf = random_3sat(n_vars=n, n_clauses=m, rng=rng)
        expected = _brute_force_satisfiable(cnf)
        result = solve(cnf)
        assert result.satisfiable == expected, f"trial {trial}: n={n} m={m}"
        if result.satisfiable:
            assert _check_assignment(cnf, result.assignment)


def test_satisfiable_instance_returns_valid_assignment():
    clauses = [
        (1, 2, -3),
        (-1, 2, 3),
        (1, -2, 3),
        (-1, -2, -3),
    ]
    cnf = CNF(n_vars=3, clauses=clauses)
    result = solve(cnf)
    assert result.satisfiable
    assert _check_assignment(cnf, result.assignment)
    assert set(result.assignment.keys()) == {1, 2, 3}


def test_decision_cap_is_honored():
    # An artificially tiny cap on a nontrivial formula must abort early
    # rather than run the search to completion.
    import random

    from src.cnf import random_3sat

    cnf = random_3sat(n_vars=40, n_clauses=180, rng=random.Random(42))
    result = solve(cnf, decision_cap=1)
    assert result.decisions <= 2  # allowed to exceed the cap by one check
    assert result.hit_cap or result.decisions <= 1


def test_decisions_and_backtracks_are_nonnegative():
    import random

    from src.cnf import random_3sat

    cnf = random_3sat(n_vars=20, n_clauses=85, rng=random.Random(7))
    result = solve(cnf)
    assert result.decisions >= 0
    assert result.backtracks >= 0
