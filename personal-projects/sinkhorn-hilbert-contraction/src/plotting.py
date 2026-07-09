"""Figure generation for the Sinkhorn/Birkhoff-contraction-rate experiment.

Colors follow a fixed categorical assignment (never cycled/re-derived per
plot) drawn from a validated colorblind-safe palette:
  random_points -> blue, clustered_points -> red,
  grid_1d -> green, iid_random -> violet.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILY_COLOR = {
    "random_points": "#2a78d6",
    "clustered_points": "#e34948",
    "grid_1d": "#008300",
    "iid_random": "#4a3aa7",
}
FAMILY_LABEL = {
    "random_points": "Random points (2D)",
    "clustered_points": "Clustered points (2D)",
    "grid_1d": "1D grid",
    "iid_random": "i.i.d. random cost",
}
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_rate_vs_bound_scatter(records, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    fittable = [r for r in records if r.tightness is not None]
    for family in FAMILY_COLOR:
        pts = [r for r in fittable if r.family == family]
        if not pts:
            continue
        x = [r.rate_empirical for r in pts]
        y = [r.kappa_theory for r in pts]
        ax.scatter(x, y, s=28, alpha=0.75, color=FAMILY_COLOR[family],
                   label=FAMILY_LABEL[family], edgecolors="white", linewidths=0.4, zorder=3)

    lims = [0, 1]
    ax.plot(lims, lims, linestyle="--", color=MUTED, linewidth=1.5, zorder=2,
            label="y = x (bound would be exactly tight)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Empirical fitted contraction rate")
    ax.set_ylabel(r"Theoretical Birkhoff bound $\kappa_{\mathrm{theory}} = \tanh(\Delta(K)/4)^2$")
    ax.set_title("Every measured instance sits on/below the diagonal:\nthe Birkhoff bound is never violated")
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_tightness_by_family(records, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    families = [f for f in FAMILY_COLOR if any(r.family == f and r.tightness is not None for r in records)]
    data = [[r.tightness for r in records if r.family == f and r.tightness is not None] for f in families]

    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                     medianprops=dict(color=INK, linewidth=1.5),
                     flierprops=dict(marker="o", markersize=3, alpha=0.4, markeredgewidth=0))
    for patch, family in zip(bp["boxes"], families):
        patch.set_facecolor(FAMILY_COLOR[family])
        patch.set_alpha(0.55)
        patch.set_edgecolor(FAMILY_COLOR[family])

    ax.set_xticks(range(1, len(families) + 1))
    ax.set_xticklabels([FAMILY_LABEL[f] for f in families], rotation=12, ha="right")
    ax.set_ylabel(r"Tightness $T = \log(\kappa_{\mathrm{theory}}) / \log(\mathrm{rate}_{\mathrm{empirical}})$")
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(1.0, linestyle="--", color=MUTED, linewidth=1.0)
    ax.set_title("Bound tightness by cost-matrix family\n(1 = tight, near 0 = very loose)")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_tightness_vs_eps_matched(records, path: Path) -> None:
    """Controlled comparison: median tightness per family at each eps value
    where *every* family has fittable-rate data (avoids the confound where
    slower-converging families get extra high-tightness samples at large
    eps simply because they're the only ones still fittable there).
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fittable = [r for r in records if r.tightness is not None]
    families = list(FAMILY_COLOR)
    eps_sets = {f: set(round(r.eps, 6) for r in fittable if r.family == f) for f in families}
    common_eps = sorted(set.intersection(*eps_sets.values())) if all(eps_sets.values()) else []

    for family in families:
        medians = []
        for eps in common_eps:
            vals = [r.tightness for r in fittable if r.family == family and round(r.eps, 6) == eps]
            medians.append(np.median(vals))
        ax.plot(common_eps, medians, marker="o", markersize=5, linewidth=1.8,
                color=FAMILY_COLOR[family], label=FAMILY_LABEL[family])

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\varepsilon$ (regularization, decreasing $\rightarrow$)")
    ax.set_ylabel("Median tightness at this $\\varepsilon$ (matched across families)")
    ax.set_title("Controlled comparison: bound tightness vs. $\\varepsilon$, same $\\varepsilon$ grid per family")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_rate_and_bound_vs_eps(records, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex=True, sharey=True)
    families = list(FAMILY_COLOR)
    for ax, family in zip(axes.flat, families):
        fam_records = [r for r in records if r.family == family]
        eps_vals = sorted(set(r.eps for r in fam_records))
        empirical_med, theory_med = [], []
        for eps in eps_vals:
            at_eps = [r for r in fam_records if r.eps == eps]
            rates = [r.rate_empirical for r in at_eps if np.isfinite(r.rate_empirical)]
            kappas = [r.kappa_theory for r in at_eps]
            empirical_med.append(np.median(rates) if rates else np.nan)
            theory_med.append(np.median(kappas))

        color = FAMILY_COLOR[family]
        ax.plot(eps_vals, theory_med, linestyle="--", color=MUTED, linewidth=1.8,
                label=r"$\kappa_{\mathrm{theory}}$ (Birkhoff bound)")
        ax.plot(eps_vals, empirical_med, marker="o", markersize=4, color=color,
                linewidth=1.8, label="empirical rate (median)")
        ax.set_xscale("log")
        ax.set_title(FAMILY_LABEL[family], fontsize=10, color=color)
        _style_axes(ax)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\varepsilon$ (regularization)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Contraction rate")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Empirical Sinkhorn contraction rate vs. the theoretical Birkhoff bound")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_extreme_iterations(records, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    for family in sorted(set(r.family for r in records)):
        fam_records = [r for r in records if r.family == family]
        eps_vals = sorted(set(r.eps for r in fam_records))
        mean_iters = [np.mean([r.n_iter for r in fam_records if r.eps == eps]) for eps in eps_vals]
        ax.plot(eps_vals, mean_iters, marker="o", markersize=5, linewidth=1.8,
                color=FAMILY_COLOR[family], label=FAMILY_LABEL[family])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\varepsilon$ (regularization, decreasing $\rightarrow$)")
    ax.set_ylabel("Sinkhorn iterations to convergence (mean over seeds)")
    ax.set_title("Actual iteration growth as $\\varepsilon \\to 0$\n(mild, in contrast to the Birkhoff bound's near-instant saturation)")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_cost_convergence(points, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    eps_vals = [p.eps for p in points]
    gaps = [max(p.gap, 1e-16) for p in points]  # guard log(0)
    ax.plot(eps_vals, gaps, marker="o", markersize=5, linewidth=1.8, color=FAMILY_COLOR["random_points"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\varepsilon$ (regularization, decreasing $\rightarrow$)")
    ax.set_ylabel(r"Entropic bias: $\langle P_\varepsilon, C\rangle - \mathrm{OT}_{\mathrm{exact}}$")
    ax.set_title("Entropic OT cost converges to the exact (Hungarian) OT cost as $\\varepsilon \\to 0$")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_all_figures(main_records, extreme_records, cost_points, figures_dir: Path) -> None:
    figures_dir = Path(figures_dir)
    plot_rate_vs_bound_scatter(main_records, figures_dir / "rate_vs_bound_scatter.png")
    plot_tightness_by_family(main_records, figures_dir / "tightness_by_family.png")
    plot_tightness_vs_eps_matched(main_records, figures_dir / "tightness_vs_eps_matched.png")
    plot_rate_and_bound_vs_eps(main_records, figures_dir / "rate_and_bound_vs_eps.png")
    plot_extreme_iterations(extreme_records, figures_dir / "extreme_iterations.png")
    plot_cost_convergence(cost_points, figures_dir / "cost_convergence.png")
