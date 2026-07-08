"""Run the consecutive-prime last-digit bias experiment end to end.

Sieves all primes up to N_MAX, buckets consecutive-prime pairs into
logarithmically spaced windows, measures the same-last-digit bias in each
window, fits the conjectured 1/log(X) decay across windows, writes
results/results.json, and renders the plots in plots/.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prime_bias import (  # noqa: E402
    DIGITS,
    binomial_bias_test,
    consecutive_pairs,
    fit_inverse_log_decay,
    last_digit_matrix,
    same_digit_fraction,
    sieve_primes,
    uniform_chisquare_test,
)

N_MAX = 1_000_000_000
NUM_WINDOWS = 10
LOG10_START = 4.0


def make_window_edges(n_max: int, num_windows: int) -> np.ndarray:
    edges = np.logspace(LOG10_START, math.log10(n_max), num_windows + 1)
    edges = np.unique(np.round(edges).astype(np.int64))
    edges[-1] = n_max
    return edges


def analyze_window(p_arr: np.ndarray, q_arr: np.ndarray, lo: int, hi: int) -> dict:
    mask = (p_arr > lo) & (p_arr <= hi)
    matrix = last_digit_matrix(p_arr[mask], q_arr[mask])
    frac, total = same_digit_fraction(matrix)
    chi2, chi_p = uniform_chisquare_test(matrix)
    same_count = int(np.trace(matrix))
    binom_p = binomial_bias_test(same_count, total) if total > 0 else float("nan")
    return {
        "lo": int(lo),
        "hi": int(hi),
        "scale": math.sqrt(lo * hi) if lo > 0 else float(hi),
        "total_pairs": total,
        "same_digit_fraction": frac,
        "bias": 0.25 - frac,
        "chi2": chi2,
        "chi2_p_value": chi_p,
        "binomial_p_value_less_than_quarter": binom_p,
        "matrix": matrix.tolist(),
    }


def run(n_max: int = N_MAX, num_windows: int = NUM_WINDOWS) -> dict:
    print(f"Sieving primes up to {n_max:,} ...")
    primes = sieve_primes(n_max)
    print(f"  found {len(primes):,} primes")

    p_arr, q_arr = consecutive_pairs(primes)
    edges = make_window_edges(n_max, num_windows)

    windows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        result = analyze_window(p_arr, q_arr, int(lo), int(hi))
        windows.append(result)
        print(
            f"  window ({lo:>12,}, {hi:>12,}]: n={result['total_pairs']:>10,} "
            f"same_frac={result['same_digit_fraction']:.5f} bias={result['bias']:.5f}"
        )

    fit_windows = [w for w in windows if w["total_pairs"] >= 1000]
    scales = np.array([w["scale"] for w in fit_windows])
    biases = np.array([w["bias"] for w in fit_windows])
    fit = fit_inverse_log_decay(scales, biases)

    overall_matrix = np.sum([np.array(w["matrix"]) for w in windows], axis=0)
    overall_frac, overall_total = same_digit_fraction(overall_matrix)
    overall_chi2, overall_chi_p = uniform_chisquare_test(overall_matrix)
    overall_binom_p = binomial_bias_test(int(np.trace(overall_matrix)), overall_total)

    results = {
        "n_max": n_max,
        "num_windows": num_windows,
        "digits": list(DIGITS),
        "windows": windows,
        "decay_fit": fit,
        "overall": {
            "total_pairs": overall_total,
            "same_digit_fraction": overall_frac,
            "matrix": overall_matrix.tolist(),
            "chi2": overall_chi2,
            "chi2_p_value": overall_chi_p,
            "binomial_p_value_less_than_quarter": overall_binom_p,
        },
    }
    return results


def make_plots(results: dict, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    windows = [w for w in results["windows"] if w["total_pairs"] > 0]
    scales = np.array([w["scale"] for w in windows])
    fracs = np.array([w["same_digit_fraction"] for w in windows])
    biases = np.array([w["bias"] for w in windows])
    totals = np.array([w["total_pairs"] for w in windows])
    stderr = np.sqrt(fracs * (1 - fracs) / totals)

    # Plot 1: same-digit fraction vs scale, with the naive-independence baseline.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(scales, fracs, yerr=stderr, fmt="o-", capsize=3, label="observed same-digit fraction")
    ax.axhline(0.25, color="gray", linestyle="--", label="naive independence prediction (1/4)")
    ax.set_xscale("log")
    ax.set_xlabel("scale X (geometric mean of window bounds)")
    ax.set_ylabel("P(consecutive primes share last digit)")
    ax.set_title("Consecutive primes avoid repeating their last digit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "same_digit_fraction_vs_scale.png", dpi=150)
    plt.close(fig)

    # Plot 2: bias vs 1/ln(X) with the fitted line, testing the 1/log(X) decay law.
    fit = results["decay_fit"]
    x = 1.0 / np.log(scales)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, biases, label="observed bias = 0.25 - same_digit_fraction")
    xs = np.linspace(0, x.max() * 1.05, 100)
    ax.plot(xs, fit["slope"] * xs + fit["intercept"], color="crimson",
            label=f"fit: bias = {fit['slope']:.3f}/ln(X) + {fit['intercept']:.4f}  (R^2={fit['r_squared']:.3f})")
    ax.set_xlabel("1 / ln(X)")
    ax.set_ylabel("bias")
    ax.set_title("Bias magnitude decays linearly in 1/ln(X), as conjectured")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "bias_vs_inverse_log_scale.png", dpi=150)
    plt.close(fig)

    # Plot 3: heatmap of the last-digit transition matrix at the largest window.
    largest = results["windows"][-1]
    matrix = np.array(largest["matrix"], dtype=float)
    matrix_frac = matrix / matrix.sum()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix_frac, cmap="RdBu_r", vmin=matrix_frac.min(), vmax=matrix_frac.max())
    digits = results["digits"]
    ax.set_xticks(range(len(digits)))
    ax.set_yticks(range(len(digits)))
    ax.set_xticklabels(digits)
    ax.set_yticklabels(digits)
    ax.set_xlabel("last digit of p_{n+1}")
    ax.set_ylabel("last digit of p_n")
    ax.set_title(f"Transition frequencies for primes in ({largest['lo']:,}, {largest['hi']:,}]")
    for i in range(len(digits)):
        for j in range(len(digits)):
            ax.text(j, i, f"{matrix_frac[i, j]:.4f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="fraction of pairs")
    fig.tight_layout()
    fig.savefig(plots_dir / "last_digit_transition_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "results"
    plots_dir = ROOT / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run()
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_dir / 'results.json'}")

    make_plots(results, plots_dir)
    print(f"Wrote plots to {plots_dir}")

    fit = results["decay_fit"]
    overall = results["overall"]
    print("\n=== Summary ===")
    print(f"Overall same-digit fraction: {overall['same_digit_fraction']:.5f} (naive prediction: 0.25000)")
    print(f"Overall chi-square p-value vs uniform: {overall['chi2_p_value']:.3e}")
    print(f"Overall binomial p-value (same_frac < 0.25): {overall['binomial_p_value_less_than_quarter']:.3e}")
    print(f"1/ln(X) decay fit: slope={fit['slope']:.4f}, R^2={fit['r_squared']:.4f}, p={fit['p_value']:.3e}")


if __name__ == "__main__":
    main()
