import numpy as np

from src.experiment import (run_construction_sweep, run_control_sweep,
                             run_search_refinement, staircase_hard_instance,
                             summarize_trend)
from src.matching import max_matching_size


def test_construction_sweep_end_to_end_small():
    rng = np.random.default_rng(0)
    rows = run_construction_sweep([4, 6, 8], rng, eval_trials=200)

    assert len(rows) == 3
    for row in rows:
        n = row["n"]
        assert row["exact_finite_floor"] > row["asymptotic_floor"]
        # H0: mean RANKING ratio should not sit meaningfully below the
        # theorem's exact finite-n floor.
        assert row["mean_meets_exact_floor"], (
            f"n={n}: mean {row['ranking_adversarial_mean']:.4f} fell below "
            f"the exact floor {row['exact_finite_floor']:.4f} by more than "
            f"the tolerance"
        )
        # H3: greedy on this exact instance is (deterministically) 1/2.
        assert abs(row["greedy_adversarial"] - 0.5) < 1e-9
        # H2: random arrival order (ROM) should beat adversarial order.
        assert row["rom_minus_adversarial_lo"] > 0
        assert 0.0 <= row["ranking_adversarial_mean"] <= 1.0
        assert 0.0 <= row["ranking_rom_mean"] <= 1.0


def test_staircase_hard_instance_has_perfect_matching():
    for n in [3, 5, 10]:
        graph, order = staircase_hard_instance(n)
        assert max_matching_size(graph) == n
        assert sorted(order) == list(range(n))
        assert order == list(reversed(range(n)))


def test_search_refinement_cannot_meaningfully_beat_the_construction():
    rng = np.random.default_rng(1)
    rows, histories = run_search_refinement([6, 10], rng, n_iterations=40, trials_per_eval=20)
    assert len(rows) == 2
    for row in rows:
        n = row["n"]
        # The construction is already near the theorem's floor; local search
        # should not be able to drive the score meaningfully further below
        # the exact floor (a large drop would suggest a bug, since it would
        # violate KVV's proven worst-case guarantee).
        assert row["best_score"] >= row["exact_finite_floor"] - 0.1
        assert histories[n][-1] <= histories[n][0]


def test_control_sweep_easy_graphs_have_high_ratio():
    rng = np.random.default_rng(2)
    rows = run_control_sweep([20, 40], [0.3, 0.6], rng, eval_trials=40)
    complete_rows = [r for r in rows if r["family"] == "complete_bipartite"]
    assert len(complete_rows) == 2
    for r in complete_rows:
        assert r["ranking_mean"] == 1.0
        assert r["greedy"] == 1.0

    random_rows = [r for r in rows if r["family"] == "random_bipartite"]
    assert len(random_rows) == 4
    for r in random_rows:
        assert 0.0 <= r["ranking_mean"] <= 1.0


def test_summarize_trend_shape():
    rng = np.random.default_rng(3)
    rows = run_construction_sweep([4, 6, 8], rng, eval_trials=150)
    summary = summarize_trend(rows)
    assert set(summary) == {
        "trend_slope", "trend_intercept", "trend_r2", "asymptotic_floor",
        "all_means_meet_exact_finite_floor", "max_gap_to_exact_floor",
        "rom_significantly_better_at_every_n",
    }
    assert isinstance(summary["all_means_meet_exact_finite_floor"], bool)
    assert summary["all_means_meet_exact_finite_floor"]
