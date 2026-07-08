"""A from-scratch DPLL SAT solver instrumented to report search effort.

This is deliberately a plain Davis-Putnam-Logemann-Loveland solver (unit
propagation + branching), not a modern CDCL engine with clause learning or
restarts. That is the point: the classical phase-transition experiments
(Mitchell/Selman/Levesque 1992, Crawford/Auton 1996) measure search effort
with exactly this kind of solver, because CDCL's learned clauses would
confound "how structurally hard is this formula" with "how good is the
learning heuristic." Decision count is the standard hardness proxy used in
that literature in place of implementation-dependent wall-clock time.
"""

from dataclasses import dataclass, field


@dataclass
class SolveResult:
    satisfiable: bool
    decisions: int = 0
    backtracks: int = 0
    hit_cap: bool = False
    assignment: dict = field(default_factory=dict)


class _DecisionCapExceeded(Exception):
    pass


def solve(cnf, decision_cap=200_000):
    """Decide satisfiability of a CNF instance via DPLL.

    Returns a SolveResult with the verdict and search-effort counters.
    If the branching-decision count would exceed `decision_cap`, the search
    is aborted early and `hit_cap=True` is set (a safety valve against
    pathological worst-case blowup; satisfiable/assignment are meaningless
    in that case).
    """
    counters = {"decisions": 0, "backtracks": 0}
    assignment = {}

    # clauses as lists of ints for in-place-friendly iteration
    clauses = [list(c) for c in cnf.clauses]

    try:
        ok = _dpll(clauses, cnf.n_vars, assignment, counters, decision_cap)
    except _DecisionCapExceeded:
        return SolveResult(
            satisfiable=False,
            decisions=counters["decisions"],
            backtracks=counters["backtracks"],
            hit_cap=True,
        )

    if ok:
        # DPLL only assigns variables it needed to touch; any variable left
        # untouched once every clause is satisfied is a genuine "don't care"
        # and can take any value. Fill those in so callers get a total
        # assignment (a complete model), not just the partial witness.
        for v in range(1, cnf.n_vars + 1):
            assignment.setdefault(v, False)

    return SolveResult(
        satisfiable=ok,
        decisions=counters["decisions"],
        backtracks=counters["backtracks"],
        assignment=dict(assignment) if ok else {},
    )


def _clause_status(clause, assignment):
    """Returns ('satisfied', None), ('unit', literal), ('conflict', None),
    or ('unresolved', None) for a clause under a partial assignment."""
    unassigned = []
    for lit in clause:
        v = abs(lit)
        if v not in assignment:
            unassigned.append(lit)
            continue
        val = assignment[v]
        lit_true = val if lit > 0 else not val
        if lit_true:
            return "satisfied", None
    if not unassigned:
        return "conflict", None
    if len(unassigned) == 1:
        return "unit", unassigned[0]
    return "unresolved", None


def _unit_propagate(clauses, assignment):
    """Repeatedly assign forced unit-clause literals. Returns the list of
    variables assigned during this call (for backtracking), or None if a
    conflict was derived."""
    assigned_here = []
    changed = True
    while changed:
        changed = False
        for clause in clauses:
            status, lit = _clause_status(clause, assignment)
            if status == "conflict":
                for v in assigned_here:
                    del assignment[v]
                return None
            if status == "unit":
                v = abs(lit)
                assignment[v] = lit > 0
                assigned_here.append(v)
                changed = True
    return assigned_here


def _pick_branch_variable(clauses, assignment):
    """Most-frequent-remaining-literal heuristic: choose the unassigned
    variable appearing in the most not-yet-satisfied clauses."""
    counts = {}
    for clause in clauses:
        status, _ = _clause_status(clause, assignment)
        if status == "satisfied":
            continue
        for lit in clause:
            v = abs(lit)
            if v not in assignment:
                counts[v] = counts.get(v, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _all_satisfied(clauses, assignment):
    return all(_clause_status(c, assignment)[0] == "satisfied" for c in clauses)


def _dpll(clauses, n_vars, assignment, counters, decision_cap):
    assigned_here = _unit_propagate(clauses, assignment)
    if assigned_here is None:
        return False

    if _all_satisfied(clauses, assignment):
        return True

    var = _pick_branch_variable(clauses, assignment)
    if var is None:
        # No unsatisfied clause has an unassigned literal, but not all
        # clauses satisfied => every remaining clause is a conflict.
        for v in assigned_here:
            del assignment[v]
        return False

    counters["decisions"] += 1
    if counters["decisions"] > decision_cap:
        raise _DecisionCapExceeded()

    for value in (True, False):
        assignment[var] = value
        if _dpll(clauses, n_vars, assignment, counters, decision_cap):
            return True
        del assignment[var]
        counters["backtracks"] += 1

    for v in assigned_here:
        del assignment[v]
    return False
