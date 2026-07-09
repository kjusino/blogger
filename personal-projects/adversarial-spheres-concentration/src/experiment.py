"""Per-dimension experiment: train a classifier on the concentric-spheres task,
measure its decision-region balance, and probe its robustness with three attacks
(on-sphere geodesic, general off-manifold L2, radial baseline). Also computes the
exact Levy-concentration robustness ceiling for comparison.
"""

import numpy as np
from scipy.stats import beta as beta_dist

from src.spheres import make_dataset, sample_sphere
from src.mlp import MLP
from src.attacks import on_sphere_attack, general_l2_attack, radial_attack
from src.concentration import levy_ceiling_exact


def minority_measure_upper_bound(model, x, conf=0.95):
    """Upper confidence bound on the minority decision-region measure.

    A finite Monte Carlo probe of n points can observe k=0 minority-class
    predictions even when the true measure is small but nonzero (e.g. true
    measure ~1/(10n) will show 0 minority hits most of the time). Feeding a raw
    MLE of 0 into the isoperimetric ceiling formula spuriously returns an
    infinite ceiling. Instead we use the one-sided Clopper-Pearson upper bound
    on the minority-class probability, so the ceiling stays a valid (if
    conservative) guarantee even in the near-perfect-separation regime.
    """
    pred = model.predict(x)
    n = len(pred)
    k1 = int(np.sum(pred == 1))
    k = min(k1, n - k1)
    alpha = 1 - conf
    upper = beta_dist.ppf(1 - alpha, k + 1, n - k) if k < n else 1.0
    return float(upper)


def run_dimension(
    d,
    r_inner=1.0,
    r_outer=1.3,
    n_train_per_class=3000,
    n_val_per_class=500,
    n_test_per_class=1500,
    n_probe_per_sphere=6000,
    n_attack_per_sphere=40,
    hidden=64,
    epochs=250,
    seed=0,
    delta=0.5,
):
    """Train a classifier, then test Levy's isoperimetric ceiling against attacks
    on the population it actually constrains: MINORITY decision-region points.

    Levy's lemma (see src/concentration.py) bounds how close points in the
    *smaller* of a classifier's two decision regions on a sphere must be to the
    boundary -- it says nothing about majority-region points, which a well-fit
    classifier is free to place robustly deep in the interior. So the attacked
    population here is: for each sphere, the points the classifier itself
    labels with whichever class is *less common* on that sphere (regardless of
    the sphere's ground-truth label) -- exactly the set Levy's lemma governs.
    """
    rng = np.random.default_rng(seed)

    x_train, y_train = make_dataset(n_train_per_class, d, r_inner, r_outer, rng)
    x_val, y_val = make_dataset(n_val_per_class, d, r_inner, r_outer, rng)
    x_test, y_test = make_dataset(n_test_per_class, d, r_inner, r_outer, rng)

    model = MLP(d, hidden, rng)
    model.fit(
        x_train, y_train, epochs=epochs, lr=0.01, batch_size=256, l2=1e-4,
        rng=rng, x_val=x_val, y_val=y_val, patience=25,
    )

    test_pred = model.predict(x_test)
    test_acc = float(np.mean(test_pred == y_test.astype(int)))

    gap = r_outer - r_inner
    rows = []
    sphere_stats = {}

    for name, radius, r_to in [
        ("inner", r_inner, r_outer + gap),
        ("outer", r_outer, max(r_inner - gap, 1e-3)),
    ]:
        probe = sample_sphere(n_probe_per_sphere, d, radius, rng)
        pred = model.predict(probe)
        frac1 = float(np.mean(pred == 1))
        minority_label = 1 if frac1 <= 0.5 else 0
        p_minor = minority_measure_upper_bound(model, probe)
        ceiling = levy_ceiling_exact(p_minor, d, radius, delta=delta)
        sphere_stats[name] = {"p_minor": p_minor, "ceiling": ceiling}

        minority_points = probe[pred == minority_label]
        n_avail = min(n_attack_per_sphere, len(minority_points))
        if n_avail > 0:
            idx = rng.choice(len(minority_points), size=n_avail, replace=False)
            for i in idx:
                x0 = minority_points[i]
                y0 = float(minority_label)
                found_os, dist_os = on_sphere_attack(model, x0, y0, radius, phi_max=np.pi, steps=25, n_bin=16)
                found_gl2, dist_gl2 = general_l2_attack(model, x0, y0, eps_max=3 * gap, steps=25, n_bin=16)
                found_rad, dist_rad = radial_attack(model, x0, r_from=radius, r_to=r_to, n_bin=25)
                rows.append({
                    "d": d,
                    "sphere": name,
                    "on_sphere_found": found_os,
                    "on_sphere_dist": dist_os,
                    "general_l2_found": found_gl2,
                    "general_l2_dist": dist_gl2,
                    "radial_found": found_rad,
                    "radial_dist": dist_rad,
                })

    return {
        "d": d,
        "test_acc": test_acc,
        "p_minor_inner": sphere_stats["inner"]["p_minor"],
        "p_minor_outer": sphere_stats["outer"]["p_minor"],
        "ceiling_inner": sphere_stats["inner"]["ceiling"],
        "ceiling_outer": sphere_stats["outer"]["ceiling"],
        "rows": rows,
    }


def summarize_rows(rows, key):
    vals = np.array([r[key] for r in rows if r[key.replace("_dist", "_found")] and np.isfinite(r[key])])
    if len(vals) == 0:
        return {"n": 0, "median": np.nan, "p25": np.nan, "p75": np.nan, "found_frac": 0.0}
    return {
        "n": len(vals),
        "median": float(np.median(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
        "found_frac": len(vals) / len(rows),
    }
