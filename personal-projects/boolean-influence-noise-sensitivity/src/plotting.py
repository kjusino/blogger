"""Figure generation for the influence / noise-sensitivity experiment."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_majority_scaling(df: pd.DataFrame, fit_exponent: float, fit_intercept: float, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(df["n"], df["total_influence"], "o", markersize=3, alpha=0.6, label="I(Maj_n) (exact)")
    xs = np.array([df["n"].min(), df["n"].max()], dtype=np.float64)
    ax.loglog(xs, np.exp(fit_intercept) * xs**fit_exponent, "r--",
               label=f"fit: n^{fit_exponent:.3f}")
    ax.loglog(xs, np.sqrt(2 / np.pi) * np.sqrt(xs), "g:", label="theory: sqrt(2n/pi)")
    ax.set_xlabel("n")
    ax.set_ylabel("Total influence I(Maj_n)")
    ax.set_title("Majority: total influence vs. n")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tribes_scaling(df: pd.DataFrame, fit_exponent: float, fit_intercept: float, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(df["n"], df["max_influence"], "o", markersize=3, alpha=0.6, label="Inf_i(Tribes) (exact)")
    xs = np.array(sorted(df["n"].unique()), dtype=np.float64)
    ax.loglog(xs, np.exp(fit_intercept) * xs**fit_exponent, "r--",
               label=f"fit: n^{fit_exponent:.3f}")
    ax.loglog(xs, np.log2(xs) / xs, "g:", label="theory shape: log2(n)/n")
    ax.set_xlabel("n = w * s")
    ax.set_ylabel("Max (per-coordinate) influence")
    ax.set_title("Tribes: KKL-tight max influence vs. n")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_noise_sensitivity_curves(df: pd.DataFrame, fits: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for family, group in df.groupby("family"):
        group = group.sort_values("delta")
        exponent = fits.get(family, {}).get("exponent")
        label = family if exponent is None else f"{family} (fit slope={exponent:.2f})"
        ax.errorbar(group["delta"], group["estimate"], yerr=group["stderr"], marker="o",
                    markersize=3, capsize=2, label=label)
    ax.set_xlabel("delta (noise rate)")
    ax.set_ylabel("NS_delta(f)")
    ax.set_title("Noise sensitivity vs. delta, by function family")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_random_dnf_influence_trend(df: pd.DataFrame, majority_ref: float, tribes_ref: float, n: int, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["k"], df["total_influence"], "o-", label="Random k-DNF (Monte Carlo)")
    ax.axhline(majority_ref, color="tab:blue", linestyle=":", label=f"Majority_{n}: I={majority_ref:.2f}")
    ax.axhline(tribes_ref, color="tab:green", linestyle="--", label=f"Tribes (matched n): I={tribes_ref:.2f}")
    ax.set_xlabel("term width k")
    ax.set_ylabel("Total influence I(f)  (all at n=%d)" % n)
    ax.set_title("Random k-DNF: total influence vs. term width")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_noise_sensitivity_vs_n(df: pd.DataFrame, sheppard_limit: float, delta: float, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for family, group in df.groupby("family"):
        group = group.sort_values("n")
        ax.errorbar(group["n"], group["estimate"], yerr=group["stderr"], marker="o",
                    markersize=3, capsize=2, label=family)
    ax.axhline(sheppard_limit, color="gray", linestyle=":",
               label=f"Sheppard/CLT limit for Majority (delta={delta}): {sheppard_limit:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel(f"NS_{delta}(f)")
    ax.set_title(f"Noise sensitivity at fixed delta={delta} as n grows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_kkl_bound_check(df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for family, group in df.groupby("family"):
        ax.scatter(group["n"], group["kkl_ratio"], s=14, alpha=0.7, label=family)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("max_influence * n / (Var(f) * log2(n))")
    ax.set_title("KKL isoperimetric ratio across function families (must stay > 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
