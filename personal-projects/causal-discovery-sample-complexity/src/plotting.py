"""Figure generation for the causal-discovery sample-complexity experiment."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#0f1115"
FG = "#e6e6e6"
GRID = "#2a2d34"
ACCENT = ["#5ec8f8", "#f7b267", "#8de08a", "#f76e94", "#c792ea", "#f4d35e"]


def _style_axes(ax):
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)


def plot_recovery_curves(configs, varying_key: str, fixed_label: str, out_path: str, title: str):
    """One recovery-probability-vs-n curve per config, colored by the
    varying parameter (p or d)."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    _style_axes(ax)
    for i, cfg in enumerate(configs):
        ns = [r["n"] for r in cfg["rows"]]
        probs = [r["recovery_prob"] for r in cfg["rows"]]
        color = ACCENT[i % len(ACCENT)]
        ax.plot(ns, probs, "o-", color=color, label=f"{varying_key}={cfg[varying_key]}", linewidth=2, markersize=5)
        ax.axvline(cfg["n50_interp"], color=color, linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(0.5, color=FG, linestyle="--", alpha=0.3, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("sample size n (log scale)")
    ax.set_ylabel("P(exact skeleton recovery)")
    ax.set_title(f"{title}\n({fixed_label})")
    legend = ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_scaling_fit(configs, x_key: str, out_path: str, title: str, xlabel: str, fit):
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
    _style_axes(ax)
    xs = np.array([c[x_key] for c in configs], dtype=float)
    ys = np.array([c["n50_interp"] for c in configs], dtype=float)
    ax.scatter(xs, ys, color=ACCENT[0], s=70, zorder=3, label="empirical n50")

    x_line = np.linspace(xs.min(), xs.max(), 100)
    y_line = np.exp(fit["intercept"]) * x_line ** fit["slope"]
    ax.plot(
        x_line, y_line, "--", color=ACCENT[1],
        label=f"fit: n50 ~ {x_key}^{fit['slope']:.2f}  (R^2={fit['r_squared']:.2f})",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("n50 (sample size for 50% exact recovery)")
    ax.set_title(title)
    legend = ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_rescaled_collapse(configs, varying_key: str, rescale_fn, rescale_label: str, out_path: str, title: str, fixed_label: str):
    """Universal-collapse plot: rescale n by a theory-predicted scale factor
    (e.g. log(p) or d^2) and check whether curves for different x land on
    top of each other, the standard visual test of a phase-transition
    scaling law."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    _style_axes(ax)
    for i, cfg in enumerate(configs):
        x = cfg[varying_key]
        scale = rescale_fn(x)
        ns = np.array([r["n"] for r in cfg["rows"]], dtype=float)
        probs = [r["recovery_prob"] for r in cfg["rows"]]
        color = ACCENT[i % len(ACCENT)]
        ax.plot(ns / scale, probs, "o-", color=color, label=f"{varying_key}={x}", linewidth=2, markersize=5)
    ax.axhline(0.5, color=FG, linestyle="--", alpha=0.3, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(f"n / {rescale_label}  (rescaled sample size)")
    ax.set_ylabel("P(exact skeleton recovery)")
    ax.set_title(f"{title}\n({fixed_label})")
    legend = ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
