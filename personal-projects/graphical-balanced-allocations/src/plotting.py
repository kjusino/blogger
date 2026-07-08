"""Figure generation for the graphical balanced-allocations report.

Color usage follows a fixed categorical assignment (never re-cycled per
plot) with hues drawn from the validated 8-slot palette in
`references/palette.md` of the dataviz skill. cycle/path share one hue
(orange) distinguished only by line style since they are the two extremal
"chain-like" poor expanders and are discussed as a pair in the README.
Baselines (one-choice, classical two-choice) are drawn in muted ink, not a
categorical hue, since they are reference constants rather than families
under test.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

EXPANDER_FAMILIES = [
    ("complete", "Complete graph", "#2a78d6", "-"),
    ("regular3", "Random 3-regular", "#1baf7a", "-"),
    ("regular10", "Random 10-regular", "#eda100", "-"),
    ("erdos_renyi", "Erdos-Renyi (p=2ln n/n)", "#008300", "-"),
    ("smallworld_high_rewiring", "Small-world (p=1.0)", "#4a3aa7", "-"),
]

POOR_FAMILIES = [
    ("smallworld_low_rewiring", "Small-world (p=0.01)", "#e34948", "-"),
    ("torus", "2D torus grid", "#e87ba4", "-"),
    ("cycle", "Cycle", "#eb6834", "-"),
    ("path", "Path", "#eb6834", "--"),
]

BASELINE_STYLES = {
    "one_choice": ("One choice (baseline)", INK_MUTED, ":"),
    "classical_two_choice": ("Classical two-choice (baseline)", INK_PRIMARY, "--"),
}


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.yaxis.label.set_color(INK_PRIMARY)


def _plot_family_panel(ax, summary_df, families, title):
    for key, label, color, ls in families:
        rows = summary_df[summary_df["family"] == key].sort_values("n")
        if rows.empty:
            continue
        ax.errorbar(
            rows["n"],
            rows["mean_gap"],
            yerr=rows["std_gap"],
            label=label,
            color=color,
            linestyle=ls,
            marker="o",
            markersize=4,
            linewidth=2,
            capsize=3,
        )
    for key, (label, color, ls) in BASELINE_STYLES.items():
        rows = summary_df[summary_df["family"] == key].sort_values("n")
        if rows.empty:
            continue
        ax.plot(rows["n"], rows["mean_gap"], label=label, color=color, linestyle=ls, linewidth=1.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n (bins = balls)")
    ax.set_ylabel("mean max-load gap")
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left")
    _style_axis(ax)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_SECONDARY)


def plot_max_load_gap(summary_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=SURFACE)
    _plot_family_panel(axes[0], summary_df, EXPANDER_FAMILIES, "Expander-like families")
    _plot_family_panel(axes[1], summary_df, POOR_FAMILIES, "Poor-expansion families")
    fig.suptitle(
        "Max-load gap vs n: graphical two-choice allocation, by graph family",
        color=INK_PRIMARY,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_spectral_gap(summary_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=SURFACE)
    for ax, families, title in (
        (axes[0], EXPANDER_FAMILIES, "Expander-like families"),
        (axes[1], POOR_FAMILIES, "Poor-expansion families"),
    ):
        for key, label, color, ls in families:
            rows = summary_df[summary_df["family"] == key].sort_values("n")
            if rows.empty:
                continue
            ax.plot(
                rows["n"],
                rows["mean_spectral_gap"],
                label=label,
                color=color,
                linestyle=ls,
                marker="o",
                markersize=4,
                linewidth=2,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("normalized spectral gap")
        ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left")
        _style_axis(ax)
        ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_SECONDARY)
    fig.suptitle("Spectral gap by graph family (log-log)", color=INK_PRIMARY, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


_ALL_FAMILY_MARKERS = {
    "complete": "o",
    "regular3": "s",
    "regular10": "^",
    "erdos_renyi": "D",
    "smallworld_high_rewiring": "P",
    "smallworld_low_rewiring": "v",
    "torus": "X",
    "cycle": "*",
    "path": "h",
}


def plot_gap_vs_spectral_gap(summary_df, out_path, spearman_rho, spearman_p):
    graph_rows = summary_df[summary_df["family"].isin(_ALL_FAMILY_MARKERS)].copy()
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=SURFACE)
    norm = matplotlib.colors.LogNorm(
        vmin=graph_rows["n"].min(), vmax=graph_rows["n"].max()
    )
    cmap = plt.get_cmap("Blues")
    for family, marker in _ALL_FAMILY_MARKERS.items():
        rows = graph_rows[graph_rows["family"] == family]
        if rows.empty:
            continue
        sc = ax.scatter(
            rows["mean_spectral_gap"],
            rows["mean_gap"],
            c=rows["n"],
            cmap=cmap,
            norm=norm,
            marker=marker,
            s=70,
            edgecolors=INK_PRIMARY,
            linewidths=0.5,
            label=family,
        )
    ax.set_xscale("log")
    ax.set_xlabel("normalized spectral gap (log scale)")
    ax.set_ylabel("mean max-load gap")
    ax.set_title(
        f"Max-load gap vs spectral gap across all runs\nSpearman rho={spearman_rho:.2f}, p={spearman_p:.2g}",
        color=INK_PRIMARY,
        fontsize=11,
    )
    _style_axis(ax)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("n", color=INK_PRIMARY)
    cbar.ax.yaxis.set_tick_params(color=INK_SECONDARY)
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color=INK_SECONDARY)
    ax.legend(
        fontsize=7,
        frameon=False,
        labelcolor=INK_SECONDARY,
        loc="upper right",
        ncol=1,
        markerscale=0.9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_ranking_bar(summary_df, out_path, n_value):
    rows = summary_df[summary_df["n"] == n_value].copy()
    rows = rows.sort_values("mean_gap")
    labels = rows["family"].tolist()
    values = rows["mean_gap"].tolist()
    colors = [
        INK_MUTED if f == "one_choice" else INK_PRIMARY if f == "classical_two_choice" else "#2a78d6"
        for f in labels
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, color=INK_PRIMARY)
    ax.set_xlabel("mean max-load gap")
    ax.set_title(f"Family ranking at n={n_value}", color=INK_PRIMARY, fontsize=12, loc="left")
    _style_axis(ax)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=8,
            color=INK_SECONDARY,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
