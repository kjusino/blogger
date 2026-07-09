import numpy as np

from src.experiment import (
    run_cost_convergence_check,
    run_extreme_sweep,
    run_main_sweep,
)


def test_main_sweep_small_scale_never_violates_the_birkhoff_bound():
    """End-to-end smoke test of the primary success criterion (M1): the
    fitted empirical contraction rate must never exceed the theoretical
    Birkhoff bound (Franklin & Lorenz 1989), for every instance/eps tested.
    Run at reduced scale (fewer eps points, smaller n) to keep this fast.
    """
    eps_values = np.geomspace(1.5, 0.2, 4)
    records = run_main_sweep(n=14, m=12, eps_values=eps_values, max_iter=3000, tol=1e-10)

    assert len(records) > 0
    violations = [r for r in records if r.bound_violated]
    assert violations == [], f"Birkhoff bound violated by: {violations}"

    for r in records:
        assert r.converged, f"Sinkhorn failed to converge: {r}"
        assert 0.0 <= r.kappa_theory <= 1.0
        # rate_empirical can be NaN when convergence is so fast (large eps)
        # that fewer than min_points tail residuals are available to fit --
        # that's an expected, correctly-flagged "unfittable" outcome, not a
        # violation of anything.
        if np.isfinite(r.rate_empirical):
            assert 0.0 <= r.rate_empirical <= 1.0
        if r.tightness is not None:
            assert 0.0 <= r.tightness <= 1.0 + 1e-6


def test_main_sweep_covers_all_configured_families():
    eps_values = np.array([1.0, 0.3])
    records = run_main_sweep(n=10, m=10, eps_values=eps_values, max_iter=2000, tol=1e-9)
    families_seen = {r.family for r in records}
    assert families_seen == {"random_points", "clustered_points", "grid_1d", "iid_random"}


def test_main_sweep_record_count_matches_seeds_times_eps():
    eps_values = np.array([1.0, 0.5, 0.25])
    records = run_main_sweep(n=8, m=8, eps_values=eps_values, max_iter=1500, tol=1e-9)
    # 6 + 6 + 1 + 6 = 19 (n_seeds) instances x 3 eps values
    assert len(records) == 19 * 3


def test_extreme_sweep_runs_and_produces_finite_iteration_counts():
    eps_values = np.geomspace(0.15, 0.06, 3)
    records = run_extreme_sweep(
        n=12, m=12, eps_values=eps_values, max_iter=2000, tol=1e-8, n_seeds=2
    )
    assert len(records) == 2 * 2 * 3  # 2 families x 2 seeds x 3 eps
    for r in records:
        assert r.n_iter >= 1
        assert r.n_iter <= 2000


def test_cost_convergence_check_gap_shrinks_toward_zero():
    points = run_cost_convergence_check(
        n=10, eps_values=np.geomspace(1.0, 0.05, 5), max_iter=8000, tol=1e-10
    )
    gaps = [p.gap for p in points]
    assert all(g >= -1e-6 for g in gaps)  # entropic cost never beats exact OT
    assert gaps[-1] < gaps[0]
    assert gaps[-1] < 0.05
