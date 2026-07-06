"""End-to-end integration test: runs the experiment pipeline in quick/
reduced mode and asserts it completes without crashing, produces a
non-empty results table with the expected columns, and (with figure
generation enabled) produces at least one figure file. This is the "runs
to completion autonomously" proof for the full grid-sweep + plotting code
path, using a tiny grid so it finishes in well under 30 seconds.
"""
import os
import tempfile

from src.experiment import GridSpec, quick_grid_spec, run_grid, RESULT_FIELDS
from src.plotting import generate_all_figures


def test_quick_grid_runs_end_to_end_and_produces_results_table():
    spec = quick_grid_spec()
    with tempfile.TemporaryDirectory() as tmpdir:
        rows = run_grid(spec, out_dir=tmpdir, verbose=False, quick=True)

        assert len(rows) > 0
        for row in rows:
            assert set(row.keys()) == set(RESULT_FIELDS)

        csv_path = os.path.join(tmpdir, "grid_results.csv")
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            content = f.read()
        assert "train_size" in content.splitlines()[0]
        assert len(content.splitlines()) == len(rows) + 1  # header + rows


def test_quick_grid_expected_number_of_configs():
    spec = quick_grid_spec()
    expected = (len(spec.train_sizes) * len(spec.noise_levels)
                * len(spec.hidden_widths) * len(spec.seeds))
    rows = run_grid(spec, verbose=False, quick=True)
    assert len(rows) == expected


def test_figure_generation_produces_png_files():
    spec = quick_grid_spec()
    rows = run_grid(spec, verbose=False, quick=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = generate_all_figures(rows, tmpdir, representative=None)
        produced = os.listdir(tmpdir)
        assert len(produced) >= 1
        assert all(name.endswith(".png") for name in produced)
        assert "pearson_r_logmse_vs_lambda1_error" in summary
