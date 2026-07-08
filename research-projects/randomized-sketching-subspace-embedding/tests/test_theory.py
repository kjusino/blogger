import numpy as np

from src.theory import fit_power_law, predicted_k, subspace_distortion


def test_predicted_k_matches_formula():
    d, eps, delta, const = 20, 0.2, 0.1, 4.0
    expected = int(np.ceil(const * (d + np.log(1.0 / delta)) / eps ** 2))
    assert predicted_k(d, eps, delta, const) == expected


def test_predicted_k_increases_with_d():
    assert predicted_k(50, 0.2) > predicted_k(10, 0.2)


def test_predicted_k_increases_as_eps_shrinks():
    assert predicted_k(10, 0.05) > predicted_k(10, 0.2)


def test_subspace_distortion_zero_for_exact_orthogonal_embedding():
    Q = np.eye(5)
    assert subspace_distortion(Q) == 0.0


def test_subspace_distortion_known_singular_values():
    # sigma = [0.8, 1.3] -> eps = max(1-0.64, 1.69-1) = max(0.36, 0.69) = 0.69
    SQ = np.diag([0.8, 1.3])
    assert np.isclose(subspace_distortion(SQ), 0.69)


def test_subspace_distortion_symmetric_case():
    # sigma = [0.9, 1.1] -> 1-0.81=0.19, 1.21-1=0.21 -> eps=0.21
    SQ = np.diag([0.9, 1.1])
    assert np.isclose(subspace_distortion(SQ), 0.21)


def test_fit_power_law_recovers_known_exponent():
    rng = np.random.default_rng(0)
    x = np.geomspace(10, 10000, 40)
    true_a, true_b = 3.0, -0.5
    y = true_a * x ** true_b * np.exp(rng.normal(0, 0.01, size=x.shape))
    a, b, r2 = fit_power_law(x, y)
    assert np.isclose(a, true_a, rtol=0.1)
    assert np.isclose(b, true_b, atol=0.05)
    assert r2 > 0.95
