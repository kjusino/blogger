"""Figure generation from a results CSV produced by experiment.run_sweep."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analysis import (
    CLASSICAL_THRESHOLD,
    aggregate_by_alpha_mu,
    group_by,
    nearest_alpha,
    pearson_correlation,
    sorted_unique,
)

MU_COLORS = {
    1.0: "#7f7f7f",
    0.66: "#4c72b0",
    0.33: "#dd8452",
    0.0: "#c44e52",
}
MU_LABELS = {
    1.0: "mu=1.0 (uniform random)",
    0.66: "mu=0.66",
    0.33: "mu=0.33",
    0.0: "mu=0.0 (fully community-local)",
}


def _color_for(mu):
    return MU_COLORS.get(mu, None)


def _label_for(mu):
    return MU_LABELS.get(mu, f"mu={mu}")


def plot_satisfiability_transition(rows, out_path):
    agg = aggregate_by_alpha_mu(rows)
    alphas = sorted_unique(rows, "alpha")
    mus = sorted_unique(rows, "mu")

    fig, ax = plt.subplots(figsize=(7, 5))
    for mu in mus:
        ys = [agg[(a, mu)]["p_sat"] for a in alphas]
        ax.plot(alphas, ys, marker="o", markersize=3, label=_label_for(mu), color=_color_for(mu))

    ax.axvline(CLASSICAL_THRESHOLD, color="black", linestyle="--", linewidth=1,
               label=f"classical threshold ({CLASSICAL_THRESHOLD})")
    ax.set_xlabel("clause/variable ratio (alpha = m/n)")
    ax.set_ylabel("P(satisfiable)")
    ax.set_title("Satisfiability transition vs. community mixing parameter")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hardness_peak(rows, out_path):
    agg = aggregate_by_alpha_mu(rows)
    alphas = sorted_unique(rows, "alpha")
    mus = sorted_unique(rows, "mu")

    fig, ax = plt.subplots(figsize=(7, 5))
    for mu in mus:
        ys = [agg[(a, mu)]["median_decisions"] for a in alphas]
        ax.plot(alphas, ys, marker="o", markersize=3, label=_label_for(mu), color=_color_for(mu))

    ax.axvline(CLASSICAL_THRESHOLD, color="black", linestyle="--", linewidth=1,
               label=f"classical threshold ({CLASSICAL_THRESHOLD})")
    ax.set_yscale("log")
    ax.set_xlabel("clause/variable ratio (alpha = m/n)")
    ax.set_ylabel("median DPLL branching decisions (log scale)")
    ax.set_title("Search-effort peak vs. community mixing parameter")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_modularity_vs_mu(rows, out_path):
    groups = group_by(rows, "mu")
    mus = sorted(groups.keys())
    means, stds = [], []
    for mu in mus:
        qs = [r["modularity_q"] for r in groups[mu] if r["modularity_q"] is not None]
        mean = sum(qs) / len(qs)
        variance = sum((q - mean) ** 2 for q in qs) / len(qs)
        means.append(mean)
        stds.append(variance ** 0.5)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(mus, means, yerr=stds, marker="o", capsize=4, color="#4c72b0")
    ax.set_xlabel("mixing parameter mu")
    ax.set_ylabel("realized modularity Q of planted partition")
    ax.set_title("Generator sanity check: mu controls realized modularity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hardness_vs_modularity_scatter(rows, out_path, alpha_target=CLASSICAL_THRESHOLD):
    alphas = sorted_unique(rows, "alpha")
    alpha_used = nearest_alpha(alphas, alpha_target)
    subset = [r for r in rows if r["alpha"] == alpha_used]

    qs = [r["modularity_q"] for r in subset]
    decisions = [r["decisions"] for r in subset]
    r_value, p_value = pearson_correlation(qs, decisions)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = [_color_for(r["mu"]) for r in subset]
    ax.scatter(qs, decisions, c=colors, alpha=0.7, edgecolors="none")
    ax.set_yscale("log")
    ax.set_xlabel("realized modularity Q")
    ax.set_ylabel("DPLL branching decisions (log scale)")
    ax.set_title(f"Hardness vs. modularity at alpha={alpha_used:.3f} "
                 f"(Pearson r={r_value:.2f})")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {"alpha_used": alpha_used, "pearson_r": r_value, "pearson_p": p_value, "n": len(subset)}


def generate_all_figures(rows, figures_dir):
    plot_satisfiability_transition(rows, f"{figures_dir}/fig1_satisfiability_transition.png")
    plot_hardness_peak(rows, f"{figures_dir}/fig2_hardness_peak.png")
    plot_modularity_vs_mu(rows, f"{figures_dir}/fig3_modularity_vs_mu.png")
    scatter_stats = plot_hardness_vs_modularity_scatter(
        rows, f"{figures_dir}/fig4_hardness_vs_modularity_scatter.png"
    )
    return scatter_stats
