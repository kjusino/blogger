"""Figure generation for the threshold-theorem experiment. Uses the
non-interactive Agg backend since this runs headless."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#1b6ca8", "#e08214", "#4d9221", "#c51b7d", "#762a83"]


def plot_logical_error_vs_p(summary: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    p_values = np.array(summary["p_values"])
    distances = summary["distances"]

    by_distance = {d: [] for d in distances}
    for row in summary["raw_results"]:
        by_distance[row["distance"]].append(row)

    for i, d in enumerate(distances):
        rows = sorted(by_distance[d], key=lambda r: r["p"])
        p = np.array([r["p"] for r in rows])
        rate = np.array([r["logical_error_rate"] for r in rows])
        lo = np.array([r["ci_lo"] for r in rows])
        hi = np.array([r["ci_hi"] for r in rows])
        color = COLORS[i % len(COLORS)]
        ax.plot(p, rate, "o-", color=color, label=f"d = {d}", markersize=4)
        ax.fill_between(p, lo, hi, color=color, alpha=0.15)

    if summary.get("threshold_estimate") is not None:
        ax.axvline(
            summary["threshold_estimate"],
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"estimated p_th = {summary['threshold_estimate']:.4f}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("logical error rate $P_L$")
    ax.set_title("Surface code: logical vs. physical error rate")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_threshold_crossing_zoom(summary: dict, out_path: str, zoom_frac: float = 0.4) -> None:
    p_values = np.array(summary["p_values"])
    threshold = summary.get("threshold_estimate")
    fig, ax = plt.subplots(figsize=(7, 5.5))

    by_distance = {d: [] for d in summary["distances"]}
    for row in summary["raw_results"]:
        by_distance[row["distance"]].append(row)

    if threshold is not None:
        lo_p = threshold * (1 - zoom_frac)
        hi_p = threshold * (1 + zoom_frac)
    else:
        lo_p, hi_p = p_values.min(), p_values.max()

    for i, d in enumerate(summary["distances"]):
        rows = sorted(by_distance[d], key=lambda r: r["p"])
        p = np.array([r["p"] for r in rows])
        rate = np.array([r["logical_error_rate"] for r in rows])
        mask = (p >= lo_p) & (p <= hi_p)
        color = COLORS[i % len(COLORS)]
        ax.plot(p[mask], rate[mask], "o-", color=color, label=f"d = {d}", markersize=5)

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1,
                   label=f"estimated p_th = {threshold:.4f}")

    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("logical error rate $P_L$")
    ax.set_title("Zoom near the estimated crossing point")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scaling_exponents(summary: dict, out_path: str) -> None:
    fits = summary["subthreshold_exponent_fits"]
    distances = sorted(int(d) for d in fits.keys() if "slope" in fits[d])
    fitted = [fits[str(d)]["slope"] for d in distances]
    err = [fits[str(d)]["slope_stderr"] for d in distances]
    predicted = [fits[str(d)]["predicted_slope"] for d in distances]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    x = np.arange(len(distances))
    width = 0.35
    ax.bar(x - width / 2, fitted, width, yerr=err, label="fitted slope", color="#1b6ca8", capsize=4)
    ax.bar(x + width / 2, predicted, width, label="predicted floor((d+1)/2)", color="#e08214")
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={d}" for d in distances])
    ax.set_ylabel("sub-threshold power-law exponent")
    ax.set_title("Fitted vs. predicted logical-error scaling exponent")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_all_figures(summary: dict, out_dir: str) -> None:
    plot_logical_error_vs_p(summary, f"{out_dir}/logical_error_vs_p.png")
    plot_threshold_crossing_zoom(summary, f"{out_dir}/threshold_crossing_zoom.png")
    plot_scaling_exponents(summary, f"{out_dir}/scaling_exponent_vs_distance.png")
