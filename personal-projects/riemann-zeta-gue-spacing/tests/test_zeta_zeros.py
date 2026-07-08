import pytest

from src.zeta_zeros import zero_heights

# Reference values computed independently (Odlyzko's published tables /
# standard references for the first few nontrivial zeta zeros).
KNOWN_FIRST_THREE = [14.134725141734695, 21.022039638771556, 25.01085758014569]


def test_zero_heights_matches_known_reference_values():
    heights = zero_heights(1, 3)
    assert heights == pytest.approx(KNOWN_FIRST_THREE, abs=1e-9)


def test_zero_heights_returns_requested_count_and_is_sorted():
    heights = zero_heights(5, 6)
    assert len(heights) == 6
    assert all(a < b for a, b in zip(heights, heights[1:]))


def test_zero_heights_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        zero_heights(0, 5)
    with pytest.raises(ValueError):
        zero_heights(1, 0)
