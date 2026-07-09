import json
import os

from run_experiment import main


def test_quick_pipeline_end_to_end(tmp_path, monkeypatch):
    import run_experiment as re_module

    monkeypatch.setattr(re_module, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(re_module, "FIGURES_DIR", str(tmp_path / "figures"))

    summary = main(quick=True)

    # Headline scaling exponents are in the right ballpark (loose bounds --
    # the quick grid is small, so this is a smoke test, not a precision check).
    assert 0.2 < summary["majority_total_influence_exponent"] < 0.8
    assert summary["kkl_bound_ever_violated"] is False
    assert summary["kkl_min_ratio_observed"] > 0

    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    for name in [
        "majority_scaling.csv",
        "tribes_scaling.csv",
        "noise_sensitivity.csv",
        "noise_sensitivity_vs_n.csv",
        "kkl_bound_check.csv",
        "summary.json",
    ]:
        assert (results_dir / name).exists(), f"missing {name}"

    for name in [
        "majority_scaling.png",
        "tribes_scaling.png",
        "noise_sensitivity.png",
        "noise_sensitivity_vs_n.png",
        "kkl_bound_check.png",
    ]:
        assert (figures_dir / name).exists(), f"missing {name}"

    with open(results_dir / "summary.json") as fh:
        on_disk = json.load(fh)
    assert on_disk["kkl_bound_ever_violated"] is False
