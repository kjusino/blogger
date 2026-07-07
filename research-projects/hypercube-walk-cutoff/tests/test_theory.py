import numpy as np
import pytest

from hypercube_cutoff import theory as th


def test_eigenvalues_endpoints():
    n = 37
    lam = th.eigenvalues(n)
    assert lam.shape == (n + 1,)
    assert lam[0] == pytest.approx(1.0)
    assert lam[1] == pytest.approx(1 - 1 / n)
    assert lam[-1] == pytest.approx(0.0)


def test_eigenvalues_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        th.eigenvalues(0)
    with pytest.raises(ValueError):
        th.eigenvalues(-3)


def test_chi_square_at_t0_equals_2n_minus_1():
    for n in [5, 10, 15]:
        assert th.chi_square(n, 0) == pytest.approx(2 ** n - 1, rel=1e-9)


def test_chi_square_matches_direct_double_precision_sum():
    """Cross-check the log-domain (logsumexp) implementation against a naive
    direct sum, in double precision, for n small enough that the naive sum
    doesn't overflow."""
    from scipy.special import comb
    n = 12
    for t in [0, 1, 5, 20]:
        k = np.arange(1, n + 1)
        lam = 1 - k / n
        direct = np.sum(comb(n, k) * lam ** (2 * t))
        assert th.chi_square(n, t) == pytest.approx(direct, rel=1e-6)


def test_chi_square_decreasing_in_t():
    n = 40
    values = th.chi_square(n, [0, 5, 20, 100, 500])
    assert np.all(np.diff(values) <= 0)


def test_chi_square_rejects_negative_t():
    with pytest.raises(ValueError):
        th.chi_square(10, -1)


def test_tv_upper_bound_nonnegative():
    n = 25
    bound = th.tv_upper_bound(n, [0, 1, 10, 100])
    assert np.all(bound >= 0)


def test_cutoff_time_formula():
    n = 500
    assert th.cutoff_time(n) == pytest.approx(0.5 * n * np.log(n))


def test_cutoff_time_rejects_small_n():
    with pytest.raises(ValueError):
        th.cutoff_time(1)
    with pytest.raises(ValueError):
        th.cutoff_time(0)


def test_rescale_unrescale_are_inverses():
    n = 300
    ts = np.array([10.0, 500.0, 2000.0, 9000.0])
    c = th.rescale(n, ts)
    back = th.unrescale(n, c)
    np.testing.assert_allclose(back, ts, rtol=1e-10)


def test_rescale_at_cutoff_time_is_zero():
    n = 777
    t_star = th.cutoff_time(n)
    assert th.rescale(n, t_star) == pytest.approx(0.0, abs=1e-9)


def test_limiting_profile_matches_asymptotic_chi_square_bound_for_large_n():
    """The limiting_profile closed form is derived as the n -> infinity limit
    of tv_upper_bound(n, unrescale(n, c)); check it's a good approximation
    already at a large but finite n."""
    n = 20000
    for c in [-1.0, 0.0, 1.0, 2.0, 4.0]:
        t = th.unrescale(n, c)
        exact_bound = min(float(th.tv_upper_bound(n, t)), 1.0)
        profile = min(float(th.limiting_profile(c)), 1.0)
        assert exact_bound == pytest.approx(profile, abs=0.01)


def test_limiting_profile_is_decreasing_in_c():
    c = np.linspace(-3, 6, 50)
    profile = th.limiting_profile(c)
    assert np.all(np.diff(profile) <= 1e-12)


def test_limiting_profile_approaches_zero_for_large_c():
    assert th.limiting_profile(40.0) == pytest.approx(0.0, abs=1e-6)


def test_rescale_and_unrescale_reject_small_n():
    with pytest.raises(ValueError):
        th.rescale(1, 5.0)
    with pytest.raises(ValueError):
        th.unrescale(1, 0.0)
