import numpy as np

from tda_phase_transitions.cycle_onset import cycle_onset_threshold, cycle_birth_rate


def test_cycle_onset_threshold_basic_quantile():
    # Births uniformly spaced 0..9, all long-lived (persistence=5)
    births = np.arange(10, dtype=float)
    deaths = births + 5.0
    dgm1 = np.column_stack([births, deaths])
    onset = cycle_onset_threshold(dgm1, birth_quantile=0.1, min_persistence=0.0)
    assert np.isclose(onset, np.quantile(births, 0.1))


def test_cycle_onset_threshold_filters_noise_by_persistence():
    # Half the bars are noise (persistence 0.01), half are significant (persistence 5)
    noise_births = np.linspace(0, 1, 20)
    noise = np.column_stack([noise_births, noise_births + 0.01])
    sig_births = np.linspace(5, 6, 20)
    significant = np.column_stack([sig_births, sig_births + 5.0])
    dgm1 = np.vstack([noise, significant])

    onset = cycle_onset_threshold(dgm1, birth_quantile=0.1, min_persistence=1.0)
    # Only the significant bars should count, so onset must be >= 5
    assert onset >= 5.0


def test_cycle_onset_threshold_none_when_empty():
    assert cycle_onset_threshold(np.zeros((0, 2))) is None


def test_cycle_onset_threshold_none_when_all_filtered():
    dgm1 = np.array([[0.0, 0.01], [1.0, 1.02]])
    assert cycle_onset_threshold(dgm1, min_persistence=10.0) is None


def test_cycle_birth_rate_cumulative_counts():
    dgm1 = np.array([[1.0, 5.0], [2.0, 6.0], [2.0, 7.0], [4.0, 8.0]])
    thresholds = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    rate = cycle_birth_rate(dgm1, thresholds)
    assert np.array_equal(rate, [0, 1, 3, 3, 4])


def test_cycle_birth_rate_empty_diagram():
    rate = cycle_birth_rate(np.zeros((0, 2)), [0.0, 1.0])
    assert np.array_equal(rate, [0.0, 0.0])
