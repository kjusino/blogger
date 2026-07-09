"""Matplotlib figure generation for the adversarial-spheres concentration experiment."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_accuracy(summary_df, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summary_df["d"], summary_df["test_acc"], "o-", color="#1f77b4")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("dimension d")
    ax.set_ylabel("held-out test accuracy")
    ax.set_title("Classifier fit quality vs. dimension")
    ax.set_ylim(0.4, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_robustness_vs_dimension(summary_df, fits, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    d = summary_df["d"]

    ax.fill_between(d, summary_df["on_sphere_p25"], summary_df["on_sphere_p75"],
                     color="#1f77b4", alpha=0.15)
    ax.plot(d, summary_df["on_sphere_median"], "o-", color="#1f77b4",
            label="on-sphere attack (median, IQR band)")

    ax.plot(d, summary_df["general_l2_median"], "s--", color="#ff7f0e",
            label="general off-manifold L2 attack (median)")

    ceiling_avg = (summary_df["ceiling_inner"] + summary_df["ceiling_outer"]) / 2
    ax.plot(d, ceiling_avg, "k^-", label="exact Levy isoperimetric ceiling")

    if fits.get("on_sphere"):
        f = fits["on_sphere"]
        dd = np.array(sorted(d))
        ax.plot(dd, f["c"] * dd ** f["exponent"], color="#1f77b4", linestyle=":",
                linewidth=1, label=f"on-sphere fit: exp={f['exponent']:.2f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dimension d")
    ax.set_ylabel("minimal adversarial perturbation (Euclidean distance)")
    ax.set_title("Adversarial robustness vs. dimension: empirical attacks vs. theoretical ceiling")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_exponent_estimates(fits, path):
    labels, means, los, his = [], [], [], []
    for key, label in [("on_sphere", "on-sphere attack"), ("general_l2", "off-manifold L2 attack")]:
        f = fits.get(key)
        if not f or np.isnan(f["exponent"]):
            continue
        labels.append(label)
        means.append(f["exponent"])
        lo, hi = f["exponent_ci95"]
        los.append(f["exponent"] - lo)
        his.append(hi - f["exponent"])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ypos = np.arange(len(labels))
    ax.errorbar(means, ypos, xerr=[los, his], fmt="o", color="#1f77b4", capsize=4)
    ax.axvline(-0.5, color="crimson", linestyle="--", label="theoretical ceiling exponent (-0.5)")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("fitted power-law exponent (median distance ~ d^exponent)")
    ax.set_title("Fitted scaling exponents vs. isoperimetric prediction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distance_distributions(raw_df, low_d, high_d, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    for ax, d in zip(axes, [low_d, high_d]):
        mask = (raw_df["d"] == d) & raw_df["on_sphere_found"]
        vals = raw_df["on_sphere_dist"][mask]
        ax.hist(vals, bins=15, color="#1f77b4", alpha=0.8)
        ax.set_title(f"d = {d}")
        ax.set_xlabel("on-sphere adversarial distance")
    axes[0].set_ylabel("count")
    fig.suptitle("Concentration of adversarial distances: low- vs. high-dimensional")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ceiling_decomposition(summary_df, fixed_p_ceiling, fixed_p_minor, fits, path):
    """Disentangle the pure isoperimetric d-dependence (fixed minority measure,
    zero empirical noise) from the 'realized' ceiling that also inherits each
    dimension's own drifting, empirically-trained p_minor(d)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    d = summary_df["d"]
    realized = (summary_df["ceiling_inner"] + summary_df["ceiling_outer"]) / 2

    ax.plot(d, fixed_p_ceiling, "k-", linewidth=2,
            label=f"pure theory, p_minor fixed at {fixed_p_minor}")
    ax.plot(d, realized, "o--", color="#d62728",
            label="realized ceiling (each d's own trained p_minor)")
    ax.plot(d, summary_df["on_sphere_median"], "s:", color="#1f77b4",
            label="empirical on-sphere attack median")

    f = fits.get("ceiling_fixed_p_minor")
    if f and not np.isnan(f["exponent"]):
        ax.text(0.02, 0.05, f"pure-theory fitted exponent = {f['exponent']:.3f}  (predicted: -0.5)",
                transform=ax.transAxes, fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dimension d")
    ax.set_ylabel("distance")
    ax.set_title("Decomposing the ceiling: pure d^(-1/2) theory vs. realized-classifier effects")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_attack_type_comparison(summary_df, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    d = summary_df["d"]
    ax.plot(d, summary_df["on_sphere_median"], "o-", label="on-sphere (constrained)")
    ax.plot(d, summary_df["general_l2_median"], "s--", label="general L2 (unconstrained)")

    radial_found = np.nan_to_num(summary_df["radial_found_frac"], nan=0.0)
    if np.any(radial_found > 0):
        ax.plot(d, summary_df["radial_median"], "^:", label="radial (baseline)")
    else:
        ax.text(0.02, 0.05,
                "radial baseline: 0% of minority points had ANY radial flip\n"
                "(the learned boundary is not a simple radius threshold)",
                transform=ax.transAxes, fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dimension d")
    ax.set_ylabel("median adversarial distance")
    ax.set_title("Attack-type comparison: does the decision boundary\nbehave like a pure radial threshold?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
