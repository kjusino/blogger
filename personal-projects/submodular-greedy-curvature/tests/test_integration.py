from src.experiment import compute_summary, run_sweep


def test_end_to_end_quick_sweep_structure():
    """A small end-to-end run should produce well-formed records and a
    summary with all four metrics present, regardless of pass/fail."""
    records = run_sweep(
        n_values=(8, 10),
        redundancy_mults=(0.0, 0.3, 1.0, 5.0),
        seeds_per_config=2,
        k_fracs=(0.25, 0.5),
        base_seed=42,
    )
    assert len(records) == 2 * 4 * 2 * 2  # n * redundancy_mults * seeds * k

    required_keys = {
        "instance_id", "n", "redundancy_mult", "seed", "curvature", "k",
        "k_over_n", "opt_val", "greedy_val", "ratio", "curvature_bound",
        "worst_case_bound",
    }
    for r in records:
        assert required_keys <= r.keys()
        assert 0.0 <= r["curvature"] <= 1.0 + 1e-9
        assert r["ratio"] > 0.0

    summary, per_instance = compute_summary(records)
    for key in ("m1_validity", "m2_informativeness", "m3_superior_predictor",
                "m4_tightness_at_extremes", "overall_pass"):
        assert key in summary
    assert len(per_instance) == 2 * 4 * 2  # n * redundancy_mults * seeds


def test_end_to_end_quick_sweep_never_violates_curvature_bound():
    """The core validity check (M1) on a real (if small) run: the proven
    theorem must hold on every single trial, not just on average."""
    records = run_sweep(
        n_values=(8, 12),
        redundancy_mults=(0.0, 0.2, 1.0, 5.0, 20.0),
        seeds_per_config=3,
        k_fracs=(0.3, 0.5),
        base_seed=777,
    )
    summary, _ = compute_summary(records)
    assert summary["m1_validity"]["n_violations"] == 0
    assert summary["m1_validity"]["validity_rate"] == 1.0


def test_quick_sweep_reproduces_headline_qualitative_trend():
    """Even a small grid should show the two headline qualitative results:
    near-modular (mult=0 -> c=0) instances solved exactly, and the
    curvature bound never violated."""
    records = run_sweep(
        n_values=(10,),
        redundancy_mults=(0.0, 1.0, 10.0),
        seeds_per_config=4,
        k_fracs=(0.3, 0.5),
        base_seed=555,
    )
    summary, _ = compute_summary(records)
    assert summary["m4_tightness_at_extremes"]["low_curvature_mean_ratio"] >= 0.999
    assert summary["m1_validity"]["n_violations"] == 0


def test_determinism_same_seed_same_results():
    kwargs = dict(
        n_values=(10,),
        redundancy_mults=(0.2, 1.0),
        seeds_per_config=2,
        k_fracs=(0.4,),
        base_seed=99,
    )
    records_a = run_sweep(**kwargs)
    records_b = run_sweep(**kwargs)
    assert records_a == records_b
