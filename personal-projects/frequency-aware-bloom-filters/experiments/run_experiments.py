"""Runs the full experiment suite for Frequency-Aware Bloom Filters (FAB) and
writes results.json + three PNG plots into ../results and ../plots.

Three schemes are compared at EQUAL total memory (same bit array size m,
same target average hash-function count k_base -> same expected total
(item, hash) insertions T):

  * Uniform    -- classic Bloom filter, k_base hash functions for every key.
  * Oracle     -- two-tier allocation using the TRUE Zipf popularity weight
                  of every key (upper bound on achievable benefit).
  * CMS(width) -- two-tier allocation using popularity estimated from a
                  Count-Min Sketch trained on a sampled access log (what a
                  real system could actually deploy, no oracle needed).

Experiment 1 (fpr_vs_skew): empirically measured, query-weighted false
positive rate for Uniform / Oracle / CMS(width=500) across Zipf skew
parameters, summarized by median + 10th/90th percentile band over
independent trials (median rather than mean because at high skew the
per-trial statistic is heavy-tailed -- see `summarize()`).

Experiment 2 (gap_closed_vs_width): at a fixed, strongly-skewed workload,
how much of the Uniform-to-Oracle FPR gap does the CMS-based estimator
recover as its memory (width) grows?

Experiment 3 (theory_vs_empirical): validates the closed-form independence
approximation (fab.theory) against the actual empirical simulation for the
Oracle scheme across the same skew sweep.

Everything is seeded; re-running reproduces bit-identical results.json.
"""
from __future__ import annotations

import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fab.bloom_filter import BloomFilter
from fab.count_min_sketch import CountMinSketch
from fab.theory import expected_weighted_fpr, item_fpr, load_factor
from fab.tiering import assign_k, uniform_k
from fab.zipf import popularity_weights, sample_stream

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(HERE, "..", "plots")
RESULTS_DIR = os.path.join(HERE, "..", "results")

# -- Shared workload parameters --
NUM_KEYS = 6000
N_MEMBERS = 1200  # 20% of universe is "in the set"
K_BASE = 6.0
HOT_FRACTION = 0.1
K_HOT = 10
BITS_PER_MEMBER = 10
NUM_BITS = N_MEMBERS * BITS_PER_MEMBER
TRAINING_STREAM_LEN = 20_000
CMS_DEPTH = 4
N_TRIALS = 20


def summarize(values: list[float]) -> dict:
    """Median + 10th/90th percentile band, not mean +/- SEM.

    At high Zipf skew a single "hottest" key can carry most of the query
    weight, so whether *that one key* happens to be a false positive
    dominates the whole weighted-FPR statistic for a trial -- the
    per-trial distribution is heavy-tailed / bimodal, not approximately
    Gaussian. A few unlucky trials can otherwise drag the mean an order of
    magnitude above the typical value. The median and a percentile band
    describe "what usually happens" without being dominated by rare
    high-impact draws.
    """
    arr = np.asarray(values)
    return {
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(arr.mean()),
    }


def build_and_measure(keys, scores, members, hot_fraction, k_hot, k_base, num_bits, seed, rng, query_weights):
    """Build one Bloom filter and return its EXACT query-weighted FPR (a
    deterministic weighted average of bf.query(...) over every negative key,
    not a Monte Carlo sample). At the FPRs involved here (as low as 1e-4),
    sampling a few thousand queries is too noisy to resolve real differences
    between schemes; summing over the full negative-key population removes
    that noise entirely, leaving only genuine trial-to-trial variance from
    which keys land in the random member set and which hash seed is used."""
    if scores is None:
        ks = uniform_k(keys, k_base=int(round(k_base)))
    else:
        ks = assign_k(keys, scores, hot_fraction, k_hot, k_base, rng)

    bf = BloomFilter(num_bits=num_bits, seed=seed)
    for m in members:
        bf.insert(m, ks[m])

    negative_keys = [k for k in keys if k not in members]
    total_w = sum(query_weights[k] for k in negative_keys)
    empirical_fpr = sum(
        query_weights[k] * bf.query(k, ks[k]) for k in negative_keys
    ) / total_w

    load = load_factor(num_bits, bf.total_insertions)
    exact_negatives = {k: query_weights[k] for k in negative_keys}
    exact_ks = {k: ks[k] for k in negative_keys}
    theoretical_fpr = expected_weighted_fpr(exact_negatives, exact_ks, load)

    return empirical_fpr, theoretical_fpr, bf.total_insertions


def run_skew_sweep(skews: list[float]) -> dict:
    results = {"uniform": {}, "oracle": {}, "cms": {}}
    theory_oracle = {}

    for skew in skews:
        emp = {"uniform": [], "oracle": [], "cms": []}
        theory_pred = []

        for trial in range(N_TRIALS):
            rng = np.random.default_rng(hash((round(skew, 3), trial)) % (2**32))
            keys = [f"k{i}" for i in range(NUM_KEYS)]
            weight_arr = popularity_weights(NUM_KEYS, skew, rng=rng)
            scores = dict(zip(keys, weight_arr))

            member_idx = rng.choice(NUM_KEYS, size=N_MEMBERS, replace=False)
            members = set(keys[i] for i in member_idx)

            training_stream = sample_stream(weight_arr, length=TRAINING_STREAM_LEN, rng=rng)
            cms = CountMinSketch.from_stream(training_stream, width=500, depth=CMS_DEPTH, seed=trial)
            cms_scores = {key: cms.estimate(key) for key in keys}

            e_uniform, t_uniform, _ = build_and_measure(
                keys, None, members, HOT_FRACTION, K_HOT, K_BASE, NUM_BITS, trial, rng, scores
            )
            e_oracle, t_oracle, _ = build_and_measure(
                keys, scores, members, HOT_FRACTION, K_HOT, K_BASE, NUM_BITS, trial, rng, scores
            )
            e_cms, t_cms, _ = build_and_measure(
                keys, cms_scores, members, HOT_FRACTION, K_HOT, K_BASE, NUM_BITS, trial, rng, scores
            )

            emp["uniform"].append(e_uniform)
            emp["oracle"].append(e_oracle)
            emp["cms"].append(e_cms)
            theory_pred.append(t_oracle)

        for scheme in emp:
            summary = summarize(emp[scheme])
            summary["raw"] = emp[scheme]
            results[scheme][skew] = summary
        theory_oracle[skew] = float(np.median(theory_pred))

    return {"empirical": results, "theory_oracle": theory_oracle}


def run_width_sweep(skew: float, widths: list[int]) -> dict:
    results = {}
    for width in widths:
        emp = {"cms": []}
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(hash((width, trial)) % (2**32))
            keys = [f"k{i}" for i in range(NUM_KEYS)]
            weight_arr = popularity_weights(NUM_KEYS, skew, rng=rng)
            scores = dict(zip(keys, weight_arr))
            member_idx = rng.choice(NUM_KEYS, size=N_MEMBERS, replace=False)
            members = set(keys[i] for i in member_idx)

            training_stream = sample_stream(weight_arr, length=TRAINING_STREAM_LEN, rng=rng)
            cms = CountMinSketch.from_stream(training_stream, width=width, depth=CMS_DEPTH, seed=trial)
            cms_scores = {key: cms.estimate(key) for key in keys}

            e_cms, _, _ = build_and_measure(
                keys, cms_scores, members, HOT_FRACTION, K_HOT, K_BASE, NUM_BITS, trial, rng, scores
            )
            emp["cms"].append(e_cms)

        summary = summarize(emp["cms"])
        summary["raw"] = emp["cms"]
        results[width] = summary
    return results


def plot_fpr_vs_skew(skew_results: dict, path: str) -> None:
    skews = sorted(skew_results["empirical"]["uniform"].keys())
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"uniform": "#888888", "oracle": "#1f77b4", "cms": "#d62728"}
    labels = {"uniform": "Uniform (baseline)", "oracle": "Oracle-weighted (upper bound)", "cms": "CMS-weighted (width=500)"}
    for scheme in ["uniform", "oracle", "cms"]:
        medians = np.array([skew_results["empirical"][scheme][s]["median"] for s in skews])
        p10 = np.array([skew_results["empirical"][scheme][s]["p10"] for s in skews])
        p90 = np.array([skew_results["empirical"][scheme][s]["p90"] for s in skews])
        yerr = np.vstack([medians - p10, p90 - medians])
        ax.errorbar(skews, medians, yerr=yerr, marker="o", capsize=3, label=labels[scheme], color=colors[scheme])
    ax.set_xlabel("Zipf skew parameter (s)")
    ax.set_ylabel("Query-weighted false positive rate (median, 10th-90th pct.)")
    ax.set_title("FAB-filter vs. uniform Bloom filter at equal memory")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gap_closed_vs_width(width_results: dict, uniform_fpr: float, oracle_fpr: float, path: str) -> None:
    widths = sorted(width_results.keys())
    gap_closed = []
    for w in widths:
        cms_fpr = width_results[w]["median"]
        gap_closed.append((uniform_fpr - cms_fpr) / (uniform_fpr - oracle_fpr))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(widths, gap_closed, marker="o", color="#2ca02c")
    ax.axhline(1.0, color="#1f77b4", linestyle="--", label="Oracle (100% of gap closed)")
    ax.axhline(0.0, color="#888888", linestyle="--", label="Uniform (0% of gap closed)")
    ax.set_xscale("log")
    ax.set_xlabel("Count-Min Sketch width (log scale)")
    ax.set_ylabel("Fraction of Uniform->Oracle FPR gap closed")
    ax.set_title("Online frequency estimation recovers the oracle's advantage")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_theory_vs_empirical(skew_results: dict, path: str) -> None:
    skews = sorted(skew_results["empirical"]["oracle"].keys())
    empirical = [skew_results["empirical"]["oracle"][s]["median"] for s in skews]
    theoretical = [skew_results["theory_oracle"][s] for s in skews]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(skews, theoretical, marker="s", label="Theory (independence approximation)", color="#9467bd")
    ax.plot(skews, empirical, marker="o", label="Empirical simulation", color="#1f77b4")
    ax.set_xlabel("Zipf skew parameter (s)")
    ax.set_ylabel("Expected false positive rate (Oracle scheme)")
    ax.set_title("Closed-form prediction vs. simulated Bloom filter")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t0 = time.time()
    skews = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2]
    print(f"Running skew sweep over {skews} ({N_TRIALS} trials each)...")
    skew_results = run_skew_sweep(skews)
    print(f"  done in {time.time() - t0:.1f}s")

    fixed_skew = 1.8
    widths = [50, 100, 250, 500, 1000, 2500, 6000]
    print(f"Running CMS width sweep at skew={fixed_skew} over widths {widths}...")
    t1 = time.time()
    width_results = run_width_sweep(fixed_skew, widths)
    print(f"  done in {time.time() - t1:.1f}s")

    assert fixed_skew in skews, "fixed_skew must be part of the skew sweep to reuse its results"
    uniform_fpr_at_fixed_skew = skew_results["empirical"]["uniform"][fixed_skew]["median"]
    oracle_fpr_at_fixed_skew = skew_results["empirical"]["oracle"][fixed_skew]["median"]

    plot_fpr_vs_skew(skew_results, os.path.join(PLOTS_DIR, "fpr_vs_skew.png"))
    plot_gap_closed_vs_width(
        width_results, uniform_fpr_at_fixed_skew, oracle_fpr_at_fixed_skew,
        os.path.join(PLOTS_DIR, "gap_closed_vs_width.png"),
    )
    plot_theory_vs_empirical(skew_results, os.path.join(PLOTS_DIR, "theory_vs_empirical.png"))

    output = {
        "config": {
            "num_keys": NUM_KEYS,
            "n_members": N_MEMBERS,
            "k_base": K_BASE,
            "hot_fraction": HOT_FRACTION,
            "k_hot": K_HOT,
            "num_bits": NUM_BITS,
            "training_stream_len": TRAINING_STREAM_LEN,
            "cms_depth": CMS_DEPTH,
            "n_trials": N_TRIALS,
            "fixed_skew_for_width_sweep": fixed_skew,
        },
        "fpr_vs_skew": {
            scheme: {str(s): v for s, v in skew_results["empirical"][scheme].items()}
            for scheme in skew_results["empirical"]
        },
        "theory_oracle_vs_skew": {str(s): v for s, v in skew_results["theory_oracle"].items()},
        "gap_closed_vs_width": {str(w): v for w, v in width_results.items()},
        "fixed_skew_reference": {
            "uniform_fpr": uniform_fpr_at_fixed_skew,
            "oracle_fpr": oracle_fpr_at_fixed_skew,
        },
        "elapsed_seconds": time.time() - t0,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"Total elapsed: {time.time() - t0:.1f}s")
    print("Wrote plots/fpr_vs_skew.png, plots/gap_closed_vs_width.png, plots/theory_vs_empirical.png")
    print("Wrote results/results.json")


if __name__ == "__main__":
    main()
