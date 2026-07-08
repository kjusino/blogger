"""End-to-end experiment runner for the LPN/BKW scaling study.

Produces three things under `../results/` and `../plots/`:

  1. results.json      -- every raw measurement from all three studies.
  2. required_confidence_vs_depth.png
  3. scaling_vs_n.png
  4. window_size_optimum.png

Run with: python3 experiments/run_experiments.py
(from the `lpn-bkw-scaling` project root, with the venv active).
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lpn_bkw import theory
from lpn_bkw.experiment import run_config

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLOTS_DIR = os.path.join(ROOT, "plots")
RESULTS_DIR = os.path.join(ROOT, "results")

TAU = 0.1
TRIALS_A = 25   # confidence-constant-vs-depth study
TRIALS_B = 30   # scaling-vs-n study
TRIALS_C = 25   # window-size-optimum study
SEED = 2026


def divisors_up_to_half(n):
    return [d for d in range(2, n // 2 + 1) if n % d == 0]


def study_a_confidence_vs_depth():
    """Fixed window size b=4; vary elimination depth a via n = a*b.

    Rather than searching for "the C that reaches 90%" (which, at deeper
    elimination depths, may not exist within any practical compute budget:
    pivot sharing correlates the surviving samples, and pumping up the
    naive independence-assumption confidence constant C stops helping
    well before 90% is reached), this sweeps a small, bounded grid of C
    values at each depth and records the resulting success rate directly.
    The interesting empirical signature is a *plateau*: shallow depths
    climb to ~100% success as C grows, while deep depths saturate well
    below it.
    """
    print("\n=== Study A: success rate vs confidence constant, by elimination depth ===")
    b = 4
    depths = [2, 3, 4]
    candidate_consts = [20, 80, 300]
    rows = []
    for a in depths:
        n = a * b
        for c in candidate_consts:
            summary = run_config(n, b, TAU, trials=TRIALS_A, confidence_const=c, seed=SEED + a * 100 + c)
            rows.append({
                "a_levels": a, "n": n, "b": b, "tau": TAU,
                "confidence_const": c, "success_rate": summary["success_rate"],
            })
            print(f"  a={a} (n={n}) C={c:4d}: success_rate={summary['success_rate']:.2f}")
    return rows


def study_b_scaling_vs_n():
    """At the theory-optimal window size, does cost grow like n/log2(n)
    rather than like n (a naive exponential-in-n baseline)?
    """
    print("\n=== Study B: attack cost scaling vs n at the theory-optimal window ===")
    C = 150
    n_values = [12, 16, 20, 24, 28, 32]
    rows = []
    for n in n_values:
        candidates = divisors_up_to_half(n)
        b = theory.optimal_b(n, TAU, candidates, confidence_const=C)
        summary = run_config(n, b, TAU, trials=TRIALS_B, confidence_const=C, seed=SEED + n)
        summary["n_over_log2_n"] = n / math.log2(n)
        rows.append(summary)
        print(f"  n={n:2d} b={b:2d} a={n//b}: success={summary['success_rate']:.2f} "
              f"meanQ={summary['mean_total_queries']:.0f} "
              f"meanTime={summary['mean_wall_time_sec']*1000:.1f}ms")
    return rows


def study_c_window_size_optimum(n=24, feasible_query_cap=2_000_000):
    """For one n with a rich divisor set, sweep every candidate window
    size and see whether the empirically fastest b matches the b that
    minimizes the textbook total_queries formula.
    """
    print(f"\n=== Study C: window-size optimum at n={n} ===")
    C = 150
    all_divisors = divisors_up_to_half(n)
    theory_curve = []
    feasible = []
    for b in all_divisors:
        tq = theory.total_queries(n, b, TAU, confidence_const=C)
        theory_curve.append({"b": b, "a_levels": n // b, "theoretical_total_queries": tq})
        if tq <= feasible_query_cap:
            feasible.append(b)
        else:
            print(f"  b={b} (a={n//b}): SKIPPED, theoretical_total_queries={tq:.3g} "
                  f"exceeds feasibility cap {feasible_query_cap}")

    rows = []
    for b in feasible:
        summary = run_config(n, b, TAU, trials=TRIALS_C, confidence_const=C, seed=SEED + b)
        rows.append(summary)
        print(f"  b={b:2d} a={n//b}: success={summary['success_rate']:.2f} "
              f"meanQ={summary['mean_total_queries']:.0f} "
              f"meanTime={summary['mean_wall_time_sec']*1000:.1f}ms "
              f"theoryQ={summary['theoretical_total_queries']:.0f}")

    theory_best = min(theory_curve, key=lambda r: r["theoretical_total_queries"])
    empirical_best = min(rows, key=lambda r: r["mean_wall_time_sec"]) if rows else None
    return {
        "n": n,
        "theory_curve": theory_curve,
        "empirical_rows": rows,
        "theory_optimal_b": theory_best["b"],
        "empirical_fastest_b": empirical_best["b"] if empirical_best else None,
    }


def plot_study_a(rows):
    depths = sorted(set(r["a_levels"] for r in rows))
    consts = sorted(set(r["confidence_const"] for r in rows))
    colors = {2: "#27ae60", 3: "#2980b9", 4: "#c0392b"}

    fig, ax = plt.subplots(figsize=(7, 5))
    for a in depths:
        pts = sorted((r["confidence_const"], r["success_rate"]) for r in rows if r["a_levels"] == a)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color=colors.get(a, "black"), label=f"a={a} elimination levels")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("confidence constant C (log scale)")
    ax.set_ylabel("empirical success rate")
    ax.set_title("Success rate vs safety margin C, by elimination depth (b=4 fixed)\n"
                  "shallow depths climb to ~1.0; deep depths plateau below it")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "required_confidence_vs_depth.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def _linfit(xs, ys):
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    fitted = [slope * x + intercept for x in xs]
    ss_res = sum((y - f) ** 2 for y, f in zip(ys, fitted))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


def plot_study_b(rows):
    n_values = [r["n"] for r in rows]
    log2_q = [math.log2(r["mean_total_queries"]) for r in rows]
    n_over_logn = [r["n_over_log2_n"] for r in rows]

    slope_n, intercept_n, r2_n = _linfit(n_values, log2_q)
    slope_s, intercept_s, r2_s = _linfit(n_over_logn, log2_q)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(n_values, log2_q, color="#2980b9", zorder=3)
    xs = sorted(n_values)
    ax.plot(xs, [slope_n * x + intercept_n for x in xs], "--", color="#2980b9")
    ax.set_xlabel("n")
    ax.set_ylabel("log2(mean total queries)")
    ax.set_title(f"vs. n directly (pure-exponential model)\nR^2={r2_n:.3f}")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(n_over_logn, log2_q, color="#27ae60", zorder=3)
    xs = sorted(n_over_logn)
    ax.plot(xs, [slope_s * x + intercept_s for x in xs], "--", color="#27ae60")
    ax.set_xlabel("n / log2(n)")
    ax.set_ylabel("log2(mean total queries)")
    ax.set_title(f"vs. n/log2(n) (BKW's predicted model)\nR^2={r2_s:.3f}")
    ax.grid(alpha=0.3)

    fig.suptitle("Does measured BKW cost scale like n or like n/log2(n)?  (window size = theory-optimal b*)")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "scaling_vs_n.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}  (R^2 vs n = {r2_n:.4f}, R^2 vs n/log2n = {r2_s:.4f})")
    return {"r2_vs_n": r2_n, "r2_vs_n_over_logn": r2_s, "slope_vs_n": slope_n, "slope_vs_n_over_logn": slope_s}


def plot_study_c(result):
    n = result["n"]
    curve = sorted(result["theory_curve"], key=lambda r: r["b"])
    emp = {r["b"]: r for r in result["empirical_rows"]}

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    bs = [r["b"] for r in curve]
    theory_q = [r["theoretical_total_queries"] for r in curve]
    ax1.plot(bs, theory_q, "o-", color="#8e44ad", label="theoretical total_queries(b)")
    ax1.set_yscale("log")
    ax1.set_xlabel(f"window size b  (n={n} fixed)")
    ax1.set_ylabel("theoretical total queries (log scale)", color="#8e44ad")
    ax1.tick_params(axis="y", labelcolor="#8e44ad")
    for r in curve:
        if r["b"] not in emp:
            ax1.annotate("infeasible\n(not run)", (r["b"], r["theoretical_total_queries"]),
                         textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color="#8e44ad")

    ax2 = ax1.twinx()
    emp_bs = sorted(emp.keys())
    emp_times = [emp[b]["mean_wall_time_sec"] * 1000 for b in emp_bs]
    ax2.plot(emp_bs, emp_times, "s--", color="#d35400", label="measured mean wall time")
    ax2.set_ylabel("measured mean wall time (ms, log scale)", color="#d35400")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="#d35400")

    ax1.axvline(result["theory_optimal_b"], color="#8e44ad", linestyle=":", alpha=0.6)
    if result["empirical_fastest_b"] is not None:
        ax1.axvline(result["empirical_fastest_b"], color="#d35400", linestyle=":", alpha=0.6)

    ax1.set_title(f"Window-size optimum at n={n}: theory-optimal b*={result['theory_optimal_b']} "
                  f"vs empirically-fastest b={result['empirical_fastest_b']}")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "window_size_optimum.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t0 = time.time()
    a_rows = study_a_confidence_vs_depth()
    b_rows = study_b_scaling_vs_n()
    c_result = study_c_window_size_optimum(n=24)
    elapsed = time.time() - t0

    plot_study_a(a_rows)
    fit_stats = plot_study_b(b_rows)
    plot_study_c(c_result)

    results = {
        "config": {"tau": TAU, "trials_a": TRIALS_A, "trials_b": TRIALS_B,
                   "trials_c": TRIALS_C, "seed": SEED},
        "study_a_confidence_vs_depth": a_rows,
        "study_b_scaling_vs_n": b_rows,
        "study_b_fit_stats": fit_stats,
        "study_c_window_size_optimum": c_result,
        "total_wall_time_sec": elapsed,
    }
    def sanitize_infinities(obj):
        if isinstance(obj, float) and math.isinf(obj):
            return None  # keep the JSON strictly standard-compliant
        if isinstance(obj, dict):
            return {k: sanitize_infinities(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize_infinities(v) for v in obj]
        return obj

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(sanitize_infinities(results), f, indent=2)
    print(f"\nAll studies complete in {elapsed:.1f}s. Wrote {out_path}")


if __name__ == "__main__":
    main()
