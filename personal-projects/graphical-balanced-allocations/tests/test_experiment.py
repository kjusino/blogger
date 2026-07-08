import numpy as np

from src import experiment, graphs


def test_run_family_trials_returns_expected_shapes():
    actual_n, mean_gap, gaps = experiment.run_family_trials("regular3", 40, trials=5, seed=0)
    assert actual_n == 40
    assert len(gaps) == 5
    assert all(g >= 0 for g in gaps)
    assert not np.isnan(mean_gap)


def test_torus_actual_n_may_differ_from_requested():
    # grid_2d_graph(side, side) yields side*side nodes, which won't equal
    # an arbitrary requested n; run_family_trials must report the real count.
    actual_n, _, gaps = experiment.run_family_trials("torus", 40, trials=2, seed=0)
    G, _ = graphs.generate_graph("torus", 40, seed=0)
    assert actual_n == G.number_of_nodes()
    assert len(gaps) == 2


def test_run_baseline_trials_one_choice_and_two_choice():
    one = experiment.run_baseline_trials("one_choice", 100, trials=5, seed=0)
    two = experiment.run_baseline_trials("classical_two_choice", 100, trials=5, seed=0)
    assert len(one) == 5 and len(two) == 5
    assert all(g >= 0 for g in one)
    assert all(g >= 0 for g in two)


def test_run_sweep_end_to_end_small():
    families = ("complete", "cycle")
    ns = (64, 128, 256)
    trials = 20
    raw_df, summary_df = experiment.run_sweep(families=families, ns=ns, trials=trials, seed=0)

    expected_rows = len(families) * len(ns) * trials + 2 * len(ns) * trials  # + one_choice + two_choice baselines
    assert len(raw_df) == expected_rows
    assert not raw_df["max_load_gap"].isna().any()

    assert set(summary_df["family"]) == set(families) | {"one_choice", "classical_two_choice"}
    for col in ("mean_gap", "std_gap"):
        assert not summary_df[col].isna().any()

    # sanity check on the theory: pooled across n, the complete graph (a
    # near-perfect expander) should have a smaller mean max-load gap than
    # the cycle (the worst-expanding family in the sweep). Pooled rather
    # than per-n to stay robust to trial noise at any single small n.
    complete_total = summary_df[summary_df.family == "complete"]["mean_gap"].sum()
    cycle_total = summary_df[summary_df.family == "cycle"]["mean_gap"].sum()
    assert complete_total < cycle_total
