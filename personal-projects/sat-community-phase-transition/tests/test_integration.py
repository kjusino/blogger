"""End-to-end integration test: run a tiny sweep through the same code path
as run_experiment.py (generation -> solving -> modularity -> aggregation ->
plotting) and sanity-check the results, without depending on any of the
full-scale numbers reported in README.md.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis import aggregate_by_alpha_mu, load_rows, sorted_unique
from src.experiment import run_sweep, write_csv
from src.plots import generate_all_figures


def test_full_pipeline_runs_and_produces_sane_output():
    alphas = [2.0, 4.267, 7.0]  # far below, at, and far above the classical threshold
    mus = [1.0, 0.0]
    trials = 6

    rows = run_sweep(alphas, mus, trials, n_vars=18, n_communities=3, seed=0)

    assert len(rows) == len(alphas) * len(mus) * trials
    for row in rows:
        assert row["decisions"] >= 0
        assert row["backtracks"] >= 0
        assert row["runtime_sec"] >= 0
        assert not row["hit_cap"]
        assert -1.0 <= row["modularity_q"] <= 1.0

    agg = aggregate_by_alpha_mu(rows)

    # far under-constrained instances (alpha=2.0) should be satisfiable far
    # more often than far over-constrained ones (alpha=7.0), for both mu.
    for mu in mus:
        assert agg[(2.0, mu)]["p_sat"] >= agg[(7.0, mu)]["p_sat"]

    # community-local instances (mu=0.0) should never be *harder* than
    # uniform-random ones (mu=1.0) by an unreasonable margin at the easy
    # end of the spectrum -- a loose smoke check, not the headline claim
    # (that comparison is made rigorously, with real sample sizes, in
    # run_experiment.py's full sweep and reported in README.md).
    assert agg[(2.0, 0.0)]["median_decisions"] < 10_000


def test_pipeline_writes_valid_csv_and_figures(tmp_path=None):
    import shutil

    tmp_dir = tempfile.mkdtemp()
    try:
        rows = run_sweep([3.0, 4.267, 5.5], [1.0, 0.0], trials=5, n_vars=16,
                          n_communities=2, seed=1)
        csv_path = os.path.join(tmp_dir, "results.csv")
        write_csv(rows, csv_path)

        assert os.path.exists(csv_path)
        reloaded = load_rows(csv_path)
        assert len(reloaded) == len(rows)
        assert sorted_unique(reloaded, "alpha") == [3.0, 4.267, 5.5]

        figures_dir = os.path.join(tmp_dir, "figures")
        os.makedirs(figures_dir)
        stats = generate_all_figures(reloaded, figures_dir)

        expected_figures = [
            "fig1_satisfiability_transition.png",
            "fig2_hardness_peak.png",
            "fig3_modularity_vs_mu.png",
            "fig4_hardness_vs_modularity_scatter.png",
        ]
        for name in expected_figures:
            path = os.path.join(figures_dir, name)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

        assert "pearson_r" in stats
        assert -1.0 <= stats["pearson_r"] <= 1.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
