"""Figure generation for the ideal-cache matmul study."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .theory import fit_power_law

ALGO_COLOR = {"naive": "#d64545", "blocked": "#3a7bd5", "oblivious": "#2fa84f"}
ALGO_LABEL = {
    "naive": "naive (unblocked)",
    "blocked": "blocked (tile ~ sqrt(M/3))",
    "oblivious": "cache-oblivious",
}


def _records_for(records, experiment, algorithm):
    return [r for r in records if r["experiment"] == experiment and r["algorithm"] == algorithm]


def plot_scaling_n(records, out_path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fits = {}
    for algo in ("naive", "blocked", "oblivious"):
        rows = sorted(_records_for(records, "scaling_n", algo), key=lambda r: r["n"])
        ns = [r["n"] for r in rows]
        misses = [r["misses"] for r in rows]
        fit = fit_power_law(ns, misses)
        fits[algo] = fit
        ax.loglog(ns, misses, "o", color=ALGO_COLOR[algo], markersize=6)
        xs = np.array([min(ns), max(ns)], dtype=float)
        ys = np.exp(fit.intercept) * xs ** fit.slope
        ax.loglog(
            xs,
            ys,
            "-",
            color=ALGO_COLOR[algo],
            label=f"{ALGO_LABEL[algo]}: slope={fit.slope:.2f} (r2={fit.r_squared:.3f})",
        )
    ax.set_xlabel("matrix dimension n")
    ax.set_ylabel("cache misses (block transfers)")
    ax.set_title("Cache misses vs. n (fixed B, M) -- all predict slope 3")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fits


def plot_scaling_param(
    records, experiment, param_key, param_label, predicted_by_algo, out_path, title, algorithms=("naive", "blocked", "oblivious")
):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fits = {}
    for algo in algorithms:
        rows = sorted(_records_for(records, experiment, algo), key=lambda r: r[param_key])
        xs_raw = [r[param_key] for r in rows]
        misses = [r["misses"] for r in rows]
        fit = fit_power_law(xs_raw, misses)
        fits[algo] = fit
        ax.loglog(xs_raw, misses, "o", color=ALGO_COLOR[algo], markersize=6)
        xs = np.array([min(xs_raw), max(xs_raw)], dtype=float)
        ys = np.exp(fit.intercept) * xs ** fit.slope
        predicted = predicted_by_algo[algo]
        ax.loglog(
            xs,
            ys,
            "-",
            color=ALGO_COLOR[algo],
            label=f"{ALGO_LABEL[algo]}: fitted={fit.slope:.2f}, predicted={predicted:+.1f}",
        )
    ax.set_xlabel(param_label)
    ax.set_ylabel("cache misses (block transfers)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fits


def plot_naive_capacity_cliff(records, out_path, n, B):
    rows = sorted(records, key=lambda r: r["M"])
    Ms = [r["M"] for r in rows]
    misses = [r["misses"] for r in rows]
    threshold = n * B

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(Ms, misses, "o-", color=ALGO_COLOR["naive"])
    ax.axvline(threshold, color="gray", linestyle="--", linewidth=1.5, label=f"M = n*B = {threshold}")
    ax.axhline(n ** 3, color="#d64545", linestyle=":", linewidth=1, alpha=0.6, label=f"n^3 = {n**3}")
    ax.axhline((n ** 3) / B, color="#3a7bd5", linestyle=":", linewidth=1, alpha=0.6, label=f"n^3/B = {n**3//B}")
    ax.set_xlabel("cache size M (words)")
    ax.set_ylabel("cache misses (block transfers)")
    ax.set_title(f"Naive matmul's capacity cliff at M = n*B (n={n}, B={B})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tall_cache_boundary(records, out_path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for algo in ("naive", "blocked", "oblivious"):
        rows = sorted(
            _records_for(records, "tall_cache_boundary", algo), key=lambda r: r["tall_cache_ratio"]
        )
        ratios = [r["tall_cache_ratio"] for r in rows]
        misses = [r["misses"] for r in rows]
        ax.loglog(ratios, misses, "o-", color=ALGO_COLOR[algo], label=ALGO_LABEL[algo])
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.5, label="tall-cache boundary (M = B^2)")
    ax.set_xlabel("M / B^2  (tall-cache ratio; < 1 violates the assumption)")
    ax.set_ylabel("cache misses (block transfers)")
    ax.set_title("Cache-oblivious behavior across the tall-cache boundary")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fit_summary(all_fits, out_path):
    """Grouped bar chart: fitted vs predicted exponent for every
    (experiment, algorithm) combination that was fit."""
    labels = []
    fitted = []
    predicted = []
    errs = []
    for (experiment, algo), (fit, pred) in all_fits.items():
        labels.append(f"{experiment}\n{algo}")
        fitted.append(fit.slope)
        predicted.append(pred)
        errs.append(fit.stderr)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, predicted, width, label="theory", color="#999999")
    ax.bar(x + width / 2, fitted, width, yerr=errs, capsize=3, label="fitted", color="#3a7bd5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("power-law exponent")
    ax.set_title("Fitted vs. theoretical scaling exponents, all sweeps")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
