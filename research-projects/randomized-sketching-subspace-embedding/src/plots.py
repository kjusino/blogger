"""Figure generation for the randomized-sketching experiment. Headless (Agg backend)."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .theory import fit_power_law

COLORS = {"gaussian": "#4C72B0", "srht": "#DD8452", "countsketch": "#55A868"}
LABELS = {"gaussian": "Gaussian", "srht": "SRHT", "countsketch": "CountSketch"}


def _by_sketch(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r["sketch"], []).append(r)
    for k in out:
        out[k] = sorted(out[k], key=lambda r: r[key])
    return out


def plot_threshold_success(results, outdir):
    rows, k0 = results["threshold"], results["k0_threshold"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, rs in _by_sketch(rows, "k_over_k0").items():
        x = [r["k_over_k0"] for r in rs]
        y = [r["success_rate"] for r in rs]
        ax.plot(x, y, "o-", color=COLORS[name], label=LABELS[name])
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, label=f"predicted k0 = {k0}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$k \, / \, k_0$  (predicted sample complexity)")
    ax.set_ylabel(f"success rate (distortion $\\leq$ {rows[0]['eps_target']})")
    ax.set_title("Subspace-embedding success rate vs. predicted threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_threshold_success.png"), dpi=150)
    plt.close(fig)


def plot_scaling_law(results, outdir):
    rows = results["scaling"]
    d = rows[0]["d"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fits = {}
    for name, rs in _by_sketch(rows, "k").items():
        x = np.array([r["k"] for r in rs])
        y = np.array([r["median_eps"] for r in rs])
        ax.plot(x, y, "o", color=COLORS[name], label=LABELS[name])
        a, b, r2 = fit_power_law(x, y)
        fits[name] = (a, b, r2)
        xf = np.linspace(x.min(), x.max(), 50)
        ax.plot(xf, a * xf ** b, "--", color=COLORS[name], alpha=0.6,
                 label=f"{LABELS[name]} fit: $\\epsilon \\propto k^{{{b:.2f}}}$ ($R^2$={r2:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sketch dimension $k$")
    ax.set_ylabel("median distortion $\\epsilon$")
    ax.set_title(f"Distortion scaling vs. sketch size (d = {d}); theory predicts $\\epsilon \\propto k^{{-0.5}}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_scaling_law.png"), dpi=150)
    plt.close(fig)
    return fits


def plot_coherence_ablation(results, outdir):
    rows = results["coherence"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    styles = {
        ("incoherent", "srht_precond"): dict(color="#4C72B0", ls="-", marker="o", label="SRHT (D+H precond), incoherent"),
        ("incoherent", "uniform_sampling"): dict(color="#4C72B0", ls="--", marker="s", label="uniform sampling, incoherent"),
        ("coherent", "srht_precond"): dict(color="#C44E52", ls="-", marker="o", label="SRHT (D+H precond), coherent"),
        ("coherent", "uniform_sampling"): dict(color="#C44E52", ls="--", marker="s", label="uniform sampling, coherent"),
    }
    grouped = {}
    for r in rows:
        grouped.setdefault((r["basis"], r["variant"]), []).append(r)
    for key, rs in grouped.items():
        rs = sorted(rs, key=lambda r: r["k"])
        x = [r["k"] for r in rs]
        y = [r["median_eps"] for r in rs]
        ax.plot(x, y, **styles[key])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sketch dimension $k$")
    ax.set_ylabel("median distortion $\\epsilon$")
    ax.set_title("Coherence ablation: does row-mixing before subsampling matter?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_coherence_ablation.png"), dpi=150)
    plt.close(fig)


def plot_least_squares(results, outdir):
    rows, k0 = results["least_squares"], results["k0_least_squares"]
    eps_target = rows[0]["eps_target"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, rs in _by_sketch(rows, "k_over_k0").items():
        x = [r["k_over_k0"] for r in rs]
        y_med = [r["median_rel_excess"] for r in rs]
        y_p90 = [r["p90_rel_excess"] for r in rs]
        ax.plot(x, y_med, "o-", color=COLORS[name], label=f"{LABELS[name]} (median)")
        ax.plot(x, y_p90, "^--", color=COLORS[name], alpha=0.5, label=f"{LABELS[name]} (p90)")
    ax.axhline(eps_target, color="gray", linestyle=":", label=f"target $\\epsilon$ = {eps_target}")
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k \, / \, k_0$  (predicted sample complexity, on $d+1$)")
    ax.set_ylabel(r"relative excess residual  $\|A\hat{x}-b\|/\|Ax^*-b\| - 1$")
    ax.set_title("Sketch-and-solve least squares: excess error vs. sketch size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_least_squares.png"), dpi=150)
    plt.close(fig)


def plot_timing_vs_k(results, outdir):
    rows = results["timing_vs_k"]
    n, d = rows[0]["n"], rows[0]["d"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, rs in _by_sketch(rows, "k").items():
        x = [r["k"] for r in rs]
        y = [r["time_s"] * 1000 for r in rs]
        ax.plot(x, y, "o-", color=COLORS[name], label=LABELS[name])
    ax.set_xlabel("sketch dimension $k$")
    ax.set_ylabel("construction time (ms)")
    ax.set_title(f"Sketch construction time vs. $k$ (n={n}, d={d}):\nGaussian scales with $k$, SRHT/CountSketch don't")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig5_timing_vs_k.png"), dpi=150)
    plt.close(fig)


def plot_timing_vs_n(results, outdir):
    rows = results["timing_vs_n"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fits = {}
    for name, rs in _by_sketch(rows, "n").items():
        x = np.array([r["n"] for r in rs])
        y = np.array([r["time_s"] for r in rs])
        a, b, r2 = fit_power_law(x, y)
        fits[name] = (a, b, r2)
        ax.plot(x, y, "o-", color=COLORS[name], label=f"{LABELS[name]} (slope={b:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of rows $n$")
    ax.set_ylabel("construction time (s)")
    ax.set_title("Sketch construction time vs. $n$ (log-log)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig6_timing_vs_n.png"), dpi=150)
    plt.close(fig)
    return fits


def make_all_plots(results, outdir):
    os.makedirs(outdir, exist_ok=True)
    plot_threshold_success(results, outdir)
    scaling_fits = plot_scaling_law(results, outdir)
    plot_coherence_ablation(results, outdir)
    plot_least_squares(results, outdir)
    plot_timing_vs_k(results, outdir)
    timing_fits = plot_timing_vs_n(results, outdir)
    return dict(scaling_fits=scaling_fits, timing_fits=timing_fits)
