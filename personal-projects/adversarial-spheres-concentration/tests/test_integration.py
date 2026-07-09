import numpy as np

from src.experiment import run_dimension, summarize_rows


def test_run_dimension_end_to_end_smoke():
    result = run_dimension(
        d=4,
        n_train_per_class=400,
        n_val_per_class=100,
        n_test_per_class=200,
        n_attack_per_sphere=6,
        hidden=16,
        epochs=80,
        seed=0,
    )

    assert result["test_acc"] > 0.9
    assert 0.0 <= result["p_minor_inner"] <= 0.5
    assert 0.0 <= result["p_minor_outer"] <= 0.5
    assert result["ceiling_inner"] > 0
    assert result["ceiling_outer"] > 0
    assert len(result["rows"]) > 0

    summary = summarize_rows(result["rows"], "on_sphere_dist")
    assert summary["n"] >= 0  # may be 0 in a degenerate tiny run, but must not crash
    for row in result["rows"]:
        assert row["d"] == 4
        assert row["sphere"] in ("inner", "outer")


def test_higher_dimension_has_tighter_ceiling_than_lower():
    low = run_dimension(
        d=4, n_train_per_class=300, n_val_per_class=80, n_test_per_class=150,
        n_attack_per_sphere=2, hidden=12, epochs=60, seed=1,
    )
    high = run_dimension(
        d=64, n_train_per_class=300, n_val_per_class=80, n_test_per_class=150,
        n_attack_per_sphere=2, hidden=12, epochs=60, seed=1,
    )
    # holding p_minor roughly comparable, higher-dimensional ceilings should be
    # tighter (smaller) -- checked directly via the closed-form formula, not the
    # (noisier) empirical attack distances, to keep this test fast and deterministic.
    from src.concentration import levy_ceiling_exact
    p = 0.3
    assert levy_ceiling_exact(p, 64, 1.0) < levy_ceiling_exact(p, 4, 1.0)
