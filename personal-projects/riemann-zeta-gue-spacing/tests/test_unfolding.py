import numpy as np
import pytest

from src.unfolding import smooth_counting_function, unfold_heights, spacings

# Known zeta zero heights (imaginary parts), computed once offline via
# mpmath.zetazero -- hardcoded here so these tests don't re-run the (slower)
# zero-finding routine just to check the unfolding math.
FIRST_FIVE_ZERO_HEIGHTS = [
    14.134725141734695,
    21.022039638771554,
    25.010857580145688,
    30.424876125859513,
    32.935061587739190,
]


def test_smooth_counting_function_is_increasing():
    values = [smooth_counting_function(t) for t in FIRST_FIVE_ZERO_HEIGHTS]
    assert np.all(np.diff(values) > 0)


def test_smooth_counting_function_near_first_zero_is_order_one():
    # N-bar(T) tracks the true count N(T) up to an O(log T) fluctuation;
    # at the very first zero the true count is 1, so N-bar should be an
    # O(1) quantity, not e.g. wildly negative or in the thousands.
    value = smooth_counting_function(FIRST_FIVE_ZERO_HEIGHTS[0])
    assert 0.0 < value < 2.0


def test_unfold_heights_matches_pointwise_smooth_counting_function():
    unfolded = unfold_heights(FIRST_FIVE_ZERO_HEIGHTS)
    expected = np.array([smooth_counting_function(t) for t in FIRST_FIVE_ZERO_HEIGHTS])
    assert unfolded == pytest.approx(expected)


def test_spacings_of_synthetic_unit_lattice():
    x = np.array([0.0, 1.0, 2.2, 2.9, 5.0])
    s = spacings(x)
    assert s == pytest.approx([1.0, 1.2, 0.7, 2.1])


def test_spacings_requires_at_least_two_points():
    with pytest.raises(ValueError):
        spacings(np.array([1.0]))


def test_unfolded_mean_spacing_close_to_one_over_many_zeros():
    # Mean spacing of the unfolded sequence should be close to 1 even over
    # just the first few dozen zeros -- this is the entire point of
    # unfolding. Uses a slightly larger, still-hardcoded reference set
    # (zeros #1-#30) computed once offline.
    heights = [
        14.134725141734695, 21.022039638771556, 25.01085758014569,
        30.424876125859512, 32.93506158773919, 37.586178158825675,
        40.9187190121475, 43.327073280915, 48.00515088116716,
        49.7738324776723, 52.970321477714464, 56.44624769706339,
        59.34704400260235, 60.83177852460981, 65.1125440480816,
        67.07981052949417, 69.54640171117398, 72.0671576744819,
        75.70469069908393, 77.1448400688748, 79.33737502024937,
        82.91038085408603, 84.73549298051705, 87.42527461312523,
        88.80911120763446, 92.49189927055849, 94.65134404051989,
        95.87063422824531, 98.83119421819369, 101.31785100573138,
    ]
    x = unfold_heights(heights)
    s = spacings(x)
    assert s.mean() == pytest.approx(1.0, abs=0.15)
