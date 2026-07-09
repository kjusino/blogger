"""Turn results/threshold_sweep.json and results/scaling_sweep.json into
figures. Pure matplotlib, no seaborn dependency; safe to re-run."""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"

COLORS = ["#1b6ca8", "#d1495b", "#2a9d8f", "#e9a441", "#7a5195", "#4b4b4b"]


def plot_threshold(decoder_name: str, out_path: pathlib.Path, title: str):
    with open(RESULTS_DIR / "threshold_sweep.json") as f:
        records = json.load(f)

    by_L = defaultdict(list)
    for rec in records:
        if rec["decoder"] != decoder_name:
            continue
        by_L[rec["L"]].append(rec)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, L in enumerate(sorted(by_L)):
        recs = sorted(by_L[L], key=lambda r: r["p"])
        ps = [r["p"] for r in recs]
        rates = [r["logical_error_rate"] for r in recs]
        # binomial standard error, for visual error bars
        errs = [
            (r["logical_error_rate"] * (1 - r["logical_error_rate"]) / r["shots"]) ** 0.5
            for r in recs
        ]
        ax.errorbar(
            ps,
            rates,
            yerr=errs,
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=2,
            label=f"L = {L}",
            color=COLORS[i % len(COLORS)],
        )

    ax.set_xlabel("physical bit-flip rate p")
    ax.set_ylabel("logical error rate")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_decoder_comparison(out_path: pathlib.Path):
    with open(RESULTS_DIR / "threshold_sweep.json") as f:
        records = json.load(f)

    by_decoder_L = defaultdict(lambda: defaultdict(list))
    for rec in records:
        by_decoder_L[rec["decoder"]][rec["L"]].append(rec)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    titles = {"mwpm": "MWPM decoder", "union_find": "Union-Find-style decoder"}
    for ax, decoder in zip(axes, ["mwpm", "union_find"]):
        by_L = by_decoder_L[decoder]
        for i, L in enumerate(sorted(by_L)):
            recs = sorted(by_L[L], key=lambda r: r["p"])
            ps = [r["p"] for r in recs]
            rates = [r["logical_error_rate"] for r in recs]
            ax.plot(ps, rates, marker="o", markersize=4, label=f"L={L}", color=COLORS[i % len(COLORS)])
        ax.set_xlabel("physical bit-flip rate p")
        ax.set_title(titles[decoder])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("logical error rate")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_scaling(out_path: pathlib.Path):
    with open(RESULTS_DIR / "scaling_sweep.json") as f:
        records = json.load(f)

    by_decoder = defaultdict(list)
    for rec in records:
        by_decoder[rec["decoder"]].append(rec)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    labels = {"mwpm": "MWPM (global blossom matching)", "union_find": "Union-Find-style clustering"}
    for i, decoder in enumerate(["mwpm", "union_find"]):
        recs = sorted(by_decoder[decoder], key=lambda r: r["L"])
        Ls = [r["L"] for r in recs]
        times = [r["mean_decode_seconds"] for r in recs]
        ax.plot(Ls, times, marker="o", markersize=5, label=labels[decoder], color=COLORS[i])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("code distance L (log scale)")
    ax.set_ylabel("mean decode time per shot, seconds (log scale)")
    ax.set_title("Decoder runtime scaling (below threshold, p = 0.03)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    plot_threshold("mwpm", RESULTS_DIR / "threshold_mwpm.png", "MWPM decoder: logical error rate vs p")
    plot_threshold(
        "union_find",
        RESULTS_DIR / "threshold_union_find.png",
        "Union-Find-style decoder: logical error rate vs p",
    )
    plot_decoder_comparison(RESULTS_DIR / "decoder_comparison.png")
    plot_scaling(RESULTS_DIR / "runtime_scaling.png")
    print("Wrote plots to", RESULTS_DIR)


if __name__ == "__main__":
    main()
