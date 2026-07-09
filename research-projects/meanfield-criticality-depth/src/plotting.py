"""Figure generation. Kept separate from experiment.py so `run_experiment.py`
can regenerate plots from saved results without re-running the sweep."""

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meanfield import (
    correlation_map,
    critical_sigma_w2,
    fixed_point_q,
    tanh,
    tanh_prime,
)
from network import propagate_correlation, propagate_gradient_norms

PHASE_EXAMPLES = {
    "ordered": (0.8, 0.05, "#3b7dd8"),
    "critical": (1.9861, 0.1, "#e07b39"),
    "chaotic": (3.2, 0.1, "#c0392b"),
}


def plot_correlation_maps(path):
    fig, ax = plt.subplots(figsize=(6, 6))
    c_grid = np.linspace(0.0, 1.0, 200)
    for label, (sw2, sb2, color) in PHASE_EXAMPLES.items():
        q = fixed_point_q(sw2, sb2, tanh)
        f_vals = [correlation_map(c, q, sw2, sb2, tanh) for c in c_grid]
        ax.plot(c_grid, f_vals, color=color, lw=2, label=f"{label}  ($\\sigma_w^2$={sw2}, $\\sigma_b^2$={sb2})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="identity ($c_{l+1}=c_l$)")
    ax.set_xlabel("$c_l$")
    ax.set_ylabel("$c_{l+1} = f(c_l)$")
    ax.set_title("Theoretical correlation map by phase")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_signal_propagation(path, seed=0):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    depth = 70
    for label, (sw2, sb2, color) in PHASE_EXAMPLES.items():
        q = fixed_point_q(sw2, sb2, tanh)
        c_hat = propagate_correlation(depth, 400, sw2, sb2, seed, q)
        axes[0].plot(np.arange(depth), c_hat, color=color, lw=1.8, label=label)

        g = propagate_gradient_norms(depth, 200, sw2, sb2, seed)
        axes[1].semilogy(np.arange(depth), g, color=color, lw=1.8, label=label)

    axes[0].set_xlabel("layer $l$")
    axes[0].set_ylabel("empirical correlation $\\hat c^l$")
    axes[0].set_title("Forward correlation propagation (two nearby inputs)")
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("layers from output")
    axes[1].set_ylabel("mean squared backprop signal (log scale)")
    axes[1].set_title("Backprop gradient-signal scaling at init")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_phase_diagram(rows, sigma_w2_grid, sigma_b2_grid, path):
    depth_grid = np.array(
        [[r["max_trainable_depth"] for r in rows if r["sigma_w2"] == sw2 and r["sigma_b2"] == sb2][0]
         for sb2 in sigma_b2_grid for sw2 in sigma_w2_grid]
    ).reshape(len(sigma_b2_grid), len(sigma_w2_grid))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.pcolormesh(sigma_w2_grid, sigma_b2_grid, depth_grid, shading="nearest", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("empirical max trainable depth $L^*$")

    sb2_fine = np.linspace(sigma_b2_grid.min(), sigma_b2_grid.max(), 60)
    sw2_crit = [critical_sigma_w2(sb2, tanh, tanh_prime) for sb2 in sb2_fine]
    ax.plot(sw2_crit, sb2_fine, color="white", lw=2.2, label="theoretical critical line ($\\chi_1=1$)")

    ax.set_xlabel("$\\sigma_w^2$")
    ax.set_ylabel("$\\sigma_b^2$")
    ax.set_title("Trainable depth vs. mean-field phase diagram")
    ax.legend(loc="upper right", fontsize=9, facecolor="black", labelcolor="white")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_depth_vs_xi(rows, path):
    xi = np.array([r["xi_c_theory"] for r in rows])
    l_star = np.array([r["max_trainable_depth"] for r in rows])
    phase = np.array([r["phase"] for r in rows])
    finite = np.isfinite(xi) & (l_star > 0)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = {"ordered": "#3b7dd8", "chaotic": "#c0392b"}
    for ph in ["ordered", "chaotic"]:
        mask = finite & (phase == ph)
        ax.scatter(xi[mask], l_star[mask], color=colors[ph], label=ph, s=45, edgecolor="k", linewidth=0.4)

    if finite.sum() > 2:
        log_xi = np.log(xi[finite])
        log_l = np.log(l_star[finite].astype(float))
        coef = np.polyfit(log_xi, log_l, 1)
        xs = np.linspace(log_xi.min(), log_xi.max(), 50)
        r = np.corrcoef(log_xi, log_l)[0, 1]
        ax.plot(np.exp(xs), np.exp(np.polyval(coef, xs)), "k--", lw=1.5,
                 label=f"log-log fit (slope={coef[0]:.2f}, r={r:.2f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("theoretical correlation length $\\xi_c$ (layers)")
    ax.set_ylabel("empirical max trainable depth $L^*$")
    ax.set_title("Does $\\xi_c$ predict trainable depth?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
