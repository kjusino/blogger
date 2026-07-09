import numpy as np
import pytest

from src.attacks import general_l2_attack, on_sphere_attack, radial_attack


class LinearModel:
    """y = 1[w.x + b >= 0]; exposes the MLP-like interface attacks need."""

    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, x):
        return x @ self.w + self.b

    def predict_proba(self, x):
        z = self.forward(x)
        return 1 / (1 + np.exp(-z))

    def predict(self, x):
        return (self.forward(x) >= 0).astype(int)

    def input_grad(self, x, y_target):
        p = self.predict_proba(x)
        dlogit = p - y_target
        return np.outer(dlogit, self.w)


def test_general_l2_attack_matches_analytic_distance_to_hyperplane():
    rng = np.random.default_rng(0)
    w = np.array([3.0, 4.0])  # ||w|| = 5
    b = -5.0
    model = LinearModel(w, b)
    x0 = np.array([4.0, 4.0])  # w.x0+b = 12+16-5=23 > 0, class 1
    analytic_dist = abs(w @ x0 + b) / np.linalg.norm(w)

    found, dist = general_l2_attack(model, x0, y_true=1.0, eps_max=20.0, steps=60, n_bin=25)
    assert found
    assert abs(dist - analytic_dist) / analytic_dist < 0.02


def test_general_l2_attack_not_found_when_budget_too_small():
    w = np.array([1.0, 0.0])
    b = -10.0
    model = LinearModel(w, b)
    x0 = np.array([0.0, 0.0])  # far from boundary (distance 10)
    found, dist = general_l2_attack(model, x0, y_true=0.0, eps_max=1.0, steps=20, n_bin=10)
    assert not found
    assert dist == np.inf


def test_on_sphere_attack_finds_flip_on_a_hemisphere_classifier():
    # classifier: class 1 iff x[0] >= 0 (a great-circle boundary) -- on a sphere of
    # radius R, a point at the pole x=(R,0,...,0) is at the boundary already, but a
    # point near the "equator minus a bit" should require only a small on-sphere
    # move to cross x[0]=0.
    rng = np.random.default_rng(1)
    d, R = 6, 2.0
    w = np.zeros(d)
    w[0] = 1.0
    model = LinearModel(w, b=0.0)

    theta = np.radians(80)  # close to the equator (boundary)
    x0 = np.zeros(d)
    x0[0] = R * np.cos(theta)
    x0[1] = R * np.sin(theta)

    found, dist = on_sphere_attack(model, x0, y_true=1.0, radius=R, phi_max=np.pi, steps=30, n_bin=20)
    assert found
    # geodesic distance from angle theta to pi/2 is (pi/2 - theta); chord should be
    # close to (but not wildly larger than) that on-sphere separation
    expected_chord = 2 * R * np.sin((np.pi / 2 - theta) / 2)
    assert dist < expected_chord * 3 + 0.05


def test_radial_attack_finds_flip_for_radius_threshold_classifier():
    d = 5
    w = np.ones(d) / np.sqrt(d)  # detects norm along the all-ones direction
    model = LinearModel(w * 0 + 0, b=0.0)  # placeholder, overwritten below

    class RadiusModel:
        def __init__(self, threshold):
            self.threshold = threshold

        def predict(self, x):
            return (np.linalg.norm(x, axis=1) >= self.threshold).astype(int)

    model = RadiusModel(threshold=1.15)
    x0 = np.ones(d) / np.sqrt(d) * 1.0  # norm 1.0, class 0
    found, dist = radial_attack(model, x0, r_from=1.0, r_to=1.3, n_bin=30)
    assert found
    assert abs(dist - 0.15) < 1e-3
