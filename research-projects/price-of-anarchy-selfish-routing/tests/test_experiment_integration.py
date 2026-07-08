import numpy as np

from src.experiment import run_topology_battery, run_worst_case_convergence
from src.theory import poa_bound


def test_topology_battery_never_violates_the_theoretical_bound():
    """The core scientific claim, exercised at small scale for a fast test:
    no randomly generated topology, of any of the tested polynomial
    degrees, should ever exceed the topology-independent PoA bound."""
    results = run_topology_battery(degrees=[1, 2, 3], n_trials=8, max_edges=8, seed=42)
    for p, battery in results.items():
        assert battery.n_violations == 0, (
            f"degree {p}: max empirical PoA {battery.max_poa:.4f} "
            f"exceeds bound {battery.bound:.4f}"
        )
        assert len(battery.trials) == 8
        assert all(t.poa >= 1.0 - 1e-6 for t in battery.trials)


def test_topology_battery_is_deterministic_given_a_seed():
    r1 = run_topology_battery(degrees=[1], n_trials=5, max_edges=6, seed=99)
    r2 = run_topology_battery(degrees=[1], n_trials=5, max_edges=6, seed=99)
    np.testing.assert_allclose(r1[1].poas, r2[1].poas)


def test_worst_case_convergence_peaks_near_the_bound():
    b_grid = np.linspace(0.01, 3.0, 25)
    results = run_worst_case_convergence(degrees=[1, 2], b_grid=b_grid)
    for p, points in results.items():
        peak = max(pt.poa for pt in points)
        assert peak <= poa_bound(p) + 1e-6
        # a 25-point grid should get reasonably close to the true peak
        assert peak >= poa_bound(p) - 0.05
