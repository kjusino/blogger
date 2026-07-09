import numpy as np

from toriccode.simulate import run_trials
from toriccode.stabilizer import ToricLattice


def test_zero_physical_error_rate_gives_zero_logical_error_rate():
    lat = ToricLattice(5)
    rng = np.random.default_rng(42)
    result = run_trials(lat, p=0.0, decoder="mwpm", shots=20, rng=rng)
    assert result.logical_error_rate == 0.0


def test_logical_error_rate_increases_with_physical_error_rate():
    """A coarse sanity check, not a threshold estimate: for a fixed small
    code, low p should give a lower (or equal) logical error rate than high
    p, on average."""
    lat = ToricLattice(5)
    rng = np.random.default_rng(7)
    low = run_trials(lat, p=0.02, decoder="mwpm", shots=300, rng=rng)
    high = run_trials(lat, p=0.3, decoder="mwpm", shots=300, rng=rng)
    assert low.logical_error_rate < high.logical_error_rate


def test_larger_code_is_better_below_threshold():
    """Well below threshold, increasing code distance should reduce the
    logical error rate (finite-size scaling, checked with a fixed seed for
    reproducibility across enough shots to beat sampling noise)."""
    rng = np.random.default_rng(123)
    p = 0.03
    small = run_trials(ToricLattice(5), p=p, decoder="mwpm", shots=400, rng=rng)
    large = run_trials(ToricLattice(9), p=p, decoder="mwpm", shots=400, rng=rng)
    assert large.logical_error_rate <= small.logical_error_rate


def test_both_decoders_run_end_to_end_without_error():
    lat = ToricLattice(6)
    rng = np.random.default_rng(99)
    for decoder in ("mwpm", "union_find"):
        result = run_trials(lat, p=0.08, decoder=decoder, shots=50, rng=rng)
        assert result.shots == 50
        assert 0.0 <= result.logical_error_rate <= 1.0
        assert result.mean_decode_seconds > 0
