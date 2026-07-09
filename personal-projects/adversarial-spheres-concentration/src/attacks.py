"""Minimum-perturbation adversarial search against a trained classifier.

Two attack families:
  - on_sphere_attack: constrained to stay exactly on the data sphere (moves along
    geodesics via the exact exponential map). This is the quantity Levy's lemma
    directly bounds: the nearest *on-manifold* decision-boundary crossing.
  - general_l2_attack: unconstrained in R^d (standard practical adversarial-example
    search), for comparison against the on-manifold notion.

Both use binary search over a perturbation budget, with a gradient-ascent inner
loop, and both report the smallest flip-inducing perturbation found (an upper
bound on the true minimal distance, standard practice for attack-based robustness
estimates).
"""

import numpy as np


def _flips(model, x_batch, y_true_batch):
    pred = model.predict(x_batch)
    return pred != y_true_batch.astype(int)


def on_sphere_attack(model, x0, y_true, radius, phi_max=np.pi, steps=25, n_bin=18):
    """Search for the nearest same-sphere point that flips the model's prediction.

    x0: (d,) starting point (already on the sphere of the given radius).
    y_true: scalar true label of x0 (used to define the loss-ascent direction:
        push logit away from the *model's own current prediction*, i.e. seek a
        decision-boundary crossing regardless of ground truth).
    Returns (found: bool, distance: float) where distance is the Euclidean chord
    length between x0 and the adversarial point found (np.inf if not found).
    """
    d = x0.shape[0]
    y0 = model.predict(x0[None, :])[0]
    x_hat0 = x0 / radius

    def run(phi):
        x_hat = x_hat0.copy()
        step_angle = phi / steps
        for _ in range(steps):
            x = radius * x_hat
            g = model.input_grad(x[None, :], np.array([1.0 - y0]))[0]
            descent = -g  # move to *minimize* loss against the flipped target
            g_tan = descent - (descent @ x_hat) * x_hat
            norm = np.linalg.norm(g_tan)
            if norm < 1e-12:
                break
            direction = g_tan / norm
            x_hat = np.cos(step_angle) * x_hat + np.sin(step_angle) * direction
            x_hat /= np.linalg.norm(x_hat)
        x_final = radius * x_hat
        cos_dist = np.clip((x_final @ x0) / (radius**2), -1, 1)
        geo_dist = np.arccos(cos_dist)
        chord = 2 * radius * np.sin(geo_dist / 2)
        flipped = model.predict(x_final[None, :])[0] != y0
        return flipped, chord

    lo, hi = 0.0, phi_max
    best_dist = np.inf
    found = False
    # exponential search to bracket a working hi if phi_max itself doesn't flip
    flipped_hi, dist_hi = run(hi)
    if flipped_hi:
        found = True
        best_dist = min(best_dist, dist_hi)
    for _ in range(n_bin):
        mid = (lo + hi) / 2
        flipped, dist = run(mid)
        if flipped:
            found = True
            best_dist = min(best_dist, dist)
            hi = mid
        else:
            lo = mid
    return found, best_dist


def general_l2_attack(model, x0, y_true, eps_max, steps=25, n_bin=18):
    """Unconstrained (off-manifold) minimum-L2-norm adversarial search."""
    d = x0.shape[0]
    y0 = model.predict(x0[None, :])[0]

    def run(eps):
        x = x0.copy()
        step_size = 2 * eps / steps
        for _ in range(steps):
            g = model.input_grad(x[None, :], np.array([1.0 - y0]))[0]
            descent = -g  # move to *minimize* loss against the flipped target
            norm = np.linalg.norm(descent)
            if norm < 1e-12:
                break
            x = x + step_size * descent / norm
            delta = x - x0
            dnorm = np.linalg.norm(delta)
            if dnorm > eps:
                x = x0 + delta * (eps / dnorm)
        flipped = model.predict(x[None, :])[0] != y0
        dist = np.linalg.norm(x - x0)
        return flipped, dist

    lo, hi = 0.0, eps_max
    best_dist = np.inf
    found = False
    flipped_hi, dist_hi = run(hi)
    if flipped_hi:
        found = True
        best_dist = min(best_dist, dist_hi)
    for _ in range(n_bin):
        mid = (lo + hi) / 2
        flipped, dist = run(mid)
        if flipped:
            found = True
            best_dist = min(best_dist, dist)
            hi = mid
        else:
            lo = mid
    return found, best_dist


def radial_attack(model, x0, r_from, r_to, n_bin=30):
    """Baseline sanity-check attack: search along the radial ray only."""
    y0 = model.predict(x0[None, :])[0]
    x_hat = x0 / r_from
    lo, hi = r_from, r_to
    flipped_hi = model.predict((hi * x_hat)[None, :])[0] != y0
    if not flipped_hi:
        return False, np.inf
    for _ in range(n_bin):
        mid = (lo + hi) / 2
        flipped = model.predict((mid * x_hat)[None, :])[0] != y0
        if flipped:
            hi = mid
        else:
            lo = mid
    return True, abs(hi - r_from)
