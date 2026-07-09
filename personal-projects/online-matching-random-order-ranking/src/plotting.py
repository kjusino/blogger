"""Matplotlib figures for the experiment. All functions take already-computed
data (rows / histories / raw samples) and a destination path -- no
computation happens here, only rendering."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .search import ONE_MINUS_INV_E, exact_finite_floor

RANKING_COLOR = "#1f77b4"
ROM_COLOR = "#2ca02c"
GREEDY_COLOR = "#d62728"
FLOOR_COLOR = "#7f7f7f"


def plot_ratio_vs_n(rows, path):
    ns = [r["n"] for r in rows]
    rk_mean = [r["ranking_adversarial_mean"] for r in rows]
    rk_lo = [r["ranking_adversarial_lo"] for r in rows]
    rk_hi = [r["ranking_adversarial_hi"] for r in rows]
    gr = [r["greedy_adversarial"] for r in rows]
    floor_exact = [exact_finite_floor(n) for n in ns]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(ns, rk_lo, rk_hi, color=RANKING_COLOR, alpha=0.2)
    ax.plot(ns, rk_mean, "o-", color=RANKING_COLOR, label="RANKING (adversarial order, staircase construction)")
    ax.plot(ns, gr, "s--", color=GREEDY_COLOR, label="Greedy (adversarial order, same instance)")
    ax.plot(ns, floor_exact, ":", color=FLOOR_COLOR, label="Exact finite-n floor  1-(1-1/n)^n")
    ax.axhline(ONE_MINUS_INV_E, color=FLOOR_COLOR, linestyle="-", linewidth=1,
               label=f"Asymptotic floor  1-1/e = {ONE_MINUS_INV_E:.4f}")
    ax.axhline(0.5, color="#bbbbbb", linestyle="--", linewidth=1, label="1/2 (naive greedy worst case)")
    ax.set_xscale("log")
    ax.set_xlabel("n (offline = online vertices, log scale)")
    ax.set_ylabel("competitive ratio (achieved / optimal)")
    ax.set_title("RANKING vs greedy on the derived hard instance")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_adversarial_vs_rom(rows, path):
    ns = [r["n"] for r in rows]
    adv_mean = [r["ranking_adversarial_mean"] for r in rows]
    adv_lo = [r["ranking_adversarial_lo"] for r in rows]
    adv_hi = [r["ranking_adversarial_hi"] for r in rows]
    rom_mean = [r["ranking_rom_mean"] for r in rows]
    rom_lo = [r["ranking_rom_lo"] for r in rows]
    rom_hi = [r["ranking_rom_hi"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(ns, adv_lo, adv_hi, color=RANKING_COLOR, alpha=0.2)
    ax.plot(ns, adv_mean, "o-", color=RANKING_COLOR, label="RANKING, adversarial arrival order")
    ax.fill_between(ns, rom_lo, rom_hi, color=ROM_COLOR, alpha=0.2)
    ax.plot(ns, rom_mean, "^-", color=ROM_COLOR, label="RANKING, random arrival order (ROM)")
    ax.axhline(ONE_MINUS_INV_E, color=FLOOR_COLOR, linestyle="-", linewidth=1,
               label=f"1-1/e = {ONE_MINUS_INV_E:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("n (log scale)")
    ax.set_ylabel("mean competitive ratio (95% bootstrap CI band)")
    ax.set_title("Same hard instance, adversarial vs. random arrival order")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ratio_distribution(adv_samples, rom_samples, n, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0.0, 1.0, 41)
    ax.hist(adv_samples, bins=bins, alpha=0.55, color=RANKING_COLOR,
            label=f"adversarial order (mean={np.mean(adv_samples):.3f})", density=True)
    ax.hist(rom_samples, bins=bins, alpha=0.55, color=ROM_COLOR,
            label=f"random order / ROM (mean={np.mean(rom_samples):.3f})", density=True)
    ax.axvline(ONE_MINUS_INV_E, color=FLOOR_COLOR, linestyle="-", linewidth=1.5,
               label=f"1-1/e = {ONE_MINUS_INV_E:.4f}")
    ax.axvline(exact_finite_floor(n), color=FLOOR_COLOR, linestyle=":", linewidth=1.5,
               label=f"exact floor at n={n} = {exact_finite_floor(n):.4f}")
    ax.set_xlabel("competitive ratio (single trial)")
    ax.set_ylabel("density")
    ax.set_title(f"RANKING's per-trial ratio distribution at n={n}")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_search_convergence(history, n, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(len(history)), history, color=RANKING_COLOR)
    ax.axhline(ONE_MINUS_INV_E, color=FLOOR_COLOR, linestyle="-", linewidth=1,
               label=f"1-1/e = {ONE_MINUS_INV_E:.4f}")
    ax.axhline(exact_finite_floor(n), color=FLOOR_COLOR, linestyle=":", linewidth=1,
               label=f"exact floor at n={n} = {exact_finite_floor(n):.4f}")
    ax.set_xlabel("search iteration")
    ax.set_ylabel("best mean ratio found so far")
    ax.set_title(f"Search initialized at the derived construction (n={n})")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_control_vs_adversarial(control_rows, adversarial_rows, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    cb = [r for r in control_rows if r["family"] == "complete_bipartite"]
    ax.plot([r["n"] for r in cb], [r["ranking_mean"] for r in cb], "o-",
            color="#9467bd", label="complete bipartite (control)")

    by_p = {}
    for r in control_rows:
        if r["family"] != "random_bipartite":
            continue
        by_p.setdefault(r["param"], []).append(r)
    cmap = plt.get_cmap("YlOrBr")
    p_values = sorted(by_p)
    for idx, p in enumerate(p_values):
        rs = sorted(by_p[p], key=lambda r: r["n"])
        color = cmap(0.4 + 0.5 * idx / max(1, len(p_values) - 1))
        ax.plot([r["n"] for r in rs], [r["ranking_mean"] for r in rs], "d-",
                color=color, label=f"random G(n,n,p={p}) (control)")

    ax.plot([r["n"] for r in adversarial_rows], [r["ranking_adversarial_mean"] for r in adversarial_rows],
            "o-", color=RANKING_COLOR, label="derived hard instance")
    ax.axhline(ONE_MINUS_INV_E, color=FLOOR_COLOR, linestyle="-", linewidth=1,
               label=f"1-1/e = {ONE_MINUS_INV_E:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("n (log scale)")
    ax.set_ylabel("RANKING mean competitive ratio")
    ax.set_title("The derived hard instance vs. easy control graphs")
    ax.set_ylim(0.55, 1.02)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
