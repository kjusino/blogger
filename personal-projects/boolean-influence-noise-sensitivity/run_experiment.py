"""Full experiment pipeline.

Research question: do the classical Boolean-function-analysis theorems on
influence and noise sensitivity -- the Kahn-Kalai-Linial (KKL) isoperimetric
influence bound, Majority's Theta(sqrt(n)) total influence, and the
Benjamini-Kalai-Schramm (BKS) Theta(delta) noise-sensitivity of Tribes --
hold up numerically across a wide range of n, and how does an "unsolved"
family (random k-DNF, for which no closed-form theorem is known) interpolate
between the Parity/Majority and Tribes regimes as term width k grows?

Run with `python run_experiment.py` (full) or `python run_experiment.py --quick`
(small grid, used by the integration test).
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from src.fitting import kkl_scaling_ratio, power_law_fit
from src.influence import monte_carlo_influence
from src.functions import (
    MajorityFunction,
    ParityFunction,
    RandomDNFFunction,
    TribesFunction,
    majority_influence_exact,
    tribes_influence_exact,
)
from src.noise_sensitivity import majority_sheppard_limit, monte_carlo_noise_sensitivity
from src.plotting import (
    plot_kkl_bound_check,
    plot_majority_scaling,
    plot_noise_sensitivity_curves,
    plot_noise_sensitivity_vs_n,
    plot_random_dnf_influence_trend,
    plot_tribes_scaling,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")

SMALL_DELTA_CUTOFF = 0.05  # deltas <= this are used for the "small-delta" exponent fit


def majority_scaling(n_values: list[int]) -> pd.DataFrame:
    rows = []
    for n in n_values:
        inf_i = majority_influence_exact(n)
        rows.append({"n": n, "influence_per_coord": inf_i, "total_influence": n * inf_i})
    return pd.DataFrame(rows)


def tribes_scaling(w_values: list[int]) -> pd.DataFrame:
    rows = []
    for w in w_values:
        balanced_s = max(2, round(math.log(2) * (2**w)))
        for multiplier in (0.5, 1.0, 2.0):
            s = max(2, round(balanced_s * multiplier))
            n = w * s
            inf_i = tribes_influence_exact(w, s)
            p_true = 2.0 ** (-w)
            pr_f_true = 1 - (1 - p_true) ** s
            variance = 1 - (2 * pr_f_true - 1) ** 2
            rows.append(
                {
                    "w": w,
                    "s": s,
                    "n": n,
                    "max_influence": inf_i,
                    "variance": variance,
                    "kkl_ratio": kkl_scaling_ratio(inf_i, variance, n) if n >= 2 else None,
                }
            )
    df = pd.DataFrame(rows).drop_duplicates(subset=["n"]).sort_values("n").reset_index(drop=True)
    return df


def noise_sensitivity_sweep(
    families: dict[str, "object"],
    deltas: list[float],
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for name, f in families.items():
        for delta in deltas:
            result = monte_carlo_noise_sensitivity(f, delta, n_samples, rng)
            rows.append(
                {
                    "family": name,
                    "n": f.n,
                    "delta": result.delta,
                    "estimate": result.estimate,
                    "stderr": result.stderr,
                }
            )
    return pd.DataFrame(rows)


def fit_small_delta_exponents(ns_df: pd.DataFrame) -> dict:
    fits = {}
    for family, group in ns_df.groupby("family"):
        small = group[(group["delta"] <= SMALL_DELTA_CUTOFF) & (group["estimate"] > 0)]
        if len(small) < 3:
            continue
        fit = power_law_fit(small["delta"].to_numpy(), small["estimate"].to_numpy())
        fits[family] = {
            "exponent": fit.exponent,
            "exponent_stderr": fit.exponent_stderr,
            "r_squared": fit.r_squared,
        }
    return fits


def random_dnf_k_sweep(n: int, k_values: list[int], deltas: list[float], n_samples: int, seed0: int) -> pd.DataFrame:
    """How does term width k reshape a balanced random k-DNF's influence
    profile? No closed-form theorem is known for this family, so this is the
    purely exploratory arm of the experiment.

    We report total influence I(f) (Monte Carlo, all n coordinates) rather
    than a small-delta noise-sensitivity exponent: NS_delta(f) ~ delta*I(f)/2
    as delta -> 0 at ANY fixed n for ANY Boolean function (see
    `noise_sensitivity_vs_n` docstring), so a small-delta exponent fit would
    converge to ~1 regardless of k and carry no information about how the
    family actually changes with clause width. Total influence is the
    quantity that genuinely varies with k, and (via the small-delta relation)
    it *is* the coefficient of the linear noise-sensitivity regime.
    """
    rows = []
    for k in k_values:
        m = max(1, round(math.log(2) * (2**k)))
        f = RandomDNFFunction(n=n, k=k, m=m, seed=seed0 + k)
        rng = np.random.default_rng(seed0 + k)
        influence = monte_carlo_influence(f, n_samples=n_samples, rng=rng, coordinates=np.arange(n))

        estimates = []
        for delta in deltas:
            if delta > SMALL_DELTA_CUTOFF:
                continue
            result = monte_carlo_noise_sensitivity(f, delta, n_samples, rng)
            estimates.append((delta, result.estimate))
        deltas_arr = np.array([d for d, e in estimates if e > 0])
        ns_arr = np.array([e for d, e in estimates if e > 0])
        row = {"k": k, "m": m, "total_influence": influence.total_influence, "variance": influence.variance}
        if len(deltas_arr) >= 3:
            fit = power_law_fit(deltas_arr, ns_arr)
            row["ns_exponent"] = fit.exponent
            row["ns_r_squared"] = fit.r_squared
        rows.append(row)
    return pd.DataFrame(rows)


def noise_sensitivity_vs_n(delta: float, n_values: list[int], n_samples: int, seed: int) -> pd.DataFrame:
    """Fix the noise level delta and grow n: this is the regime where the
    Majority-vs-Tribes distinction (Sheppard/CLT limit vs. BKS's O(delta)
    bound) is actually visible -- unlike the fixed-n, small-delta regime,
    where NS_delta(f) ~ delta*I(f)/2 for every family (see module docstring
    below `main`).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_values:
        n_odd = n if n % 2 == 1 else n + 1
        maj = MajorityFunction(n=n_odd)
        result = monte_carlo_noise_sensitivity(maj, delta, n_samples, rng)
        rows.append({"family": "majority", "n": n_odd, "delta": delta,
                     "estimate": result.estimate, "stderr": result.stderr})

        w = max(2, round(math.log2(n)))
        s = max(2, round(n / w))
        tribes = TribesFunction(w=w, s=s)
        result = monte_carlo_noise_sensitivity(tribes, delta, n_samples, rng)
        rows.append({"family": "tribes", "n": tribes.n, "delta": delta,
                     "estimate": result.estimate, "stderr": result.stderr})

        parity = ParityFunction(n=n)
        result = monte_carlo_noise_sensitivity(parity, delta, n_samples, rng)
        rows.append({"family": "parity", "n": n, "delta": delta,
                     "estimate": result.estimate, "stderr": result.stderr})
    return pd.DataFrame(rows)


def kkl_check_all(instances: list[dict]) -> pd.DataFrame:
    rows = []
    for inst in instances:
        ratio = kkl_scaling_ratio(inst["max_influence"], inst["variance"], inst["n"])
        rows.append({**inst, "kkl_ratio": ratio})
    return pd.DataFrame(rows)


def main(quick: bool = False) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if quick:
        n_maj = sorted({n | 1 for n in np.geomspace(3, 501, 12).astype(int)})
        w_tribes = list(range(2, 7))
        ns_n = 21
        deltas = [0.01, 0.02, 0.05, 0.1, 0.2]
        ns_samples = 2000
        k_values = [1, 2, 3, 4]
        fixed_n_values = [11, 51, 201]
    else:
        n_maj = sorted({n | 1 for n in np.geomspace(3, 200001, 60).astype(int)})
        w_tribes = list(range(2, 17))
        ns_n = 45
        deltas = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
        ns_samples = 30000
        k_values = [1, 2, 3, 4, 5, 6]
        fixed_n_values = [11, 51, 201, 801, 3201, 12801]

    # 1. Majority total-influence scaling
    maj_df = majority_scaling(n_maj)
    maj_df.to_csv(os.path.join(RESULTS_DIR, "majority_scaling.csv"), index=False)
    maj_fit = power_law_fit(maj_df["n"].to_numpy(), maj_df["total_influence"].to_numpy())
    plot_majority_scaling(maj_df, maj_fit.exponent, maj_fit.intercept,
                           os.path.join(FIGURES_DIR, "majority_scaling.png"))

    # 2. Tribes max-influence (KKL-tight) scaling
    tribes_df = tribes_scaling(w_tribes)
    tribes_df.to_csv(os.path.join(RESULTS_DIR, "tribes_scaling.csv"), index=False)
    tribes_fit = power_law_fit(tribes_df["n"].to_numpy(), tribes_df["max_influence"].to_numpy())
    # Stability of Inf * n / log2(n): linear-regression slope of the ratio against log(n)
    # should be close to 0 if the Theta(log n / n) prediction holds.
    ratio_series = tribes_df["max_influence"] * tribes_df["n"] / np.log2(tribes_df["n"])
    ratio_slope_fit = power_law_fit(tribes_df["n"].to_numpy(), ratio_series.to_numpy())
    plot_tribes_scaling(tribes_df, tribes_fit.exponent, tribes_fit.intercept,
                         os.path.join(FIGURES_DIR, "tribes_scaling.png"))

    # 3. Noise sensitivity across families at matched n
    balanced_w = 4 if ns_n >= 40 else 3
    balanced_s = max(2, round(ns_n / balanced_w))
    families = {
        "parity": ParityFunction(n=ns_n if ns_n % 2 == 1 else ns_n + 1),
        "majority": MajorityFunction(n=ns_n if ns_n % 2 == 1 else ns_n + 1),
        "tribes": TribesFunction(w=balanced_w, s=balanced_s),
        "random_dnf": RandomDNFFunction(n=ns_n, k=3, m=max(1, round(math.log(2) * 8)), seed=7),
    }
    ns_df = noise_sensitivity_sweep(families, deltas, ns_samples, seed=42)
    ns_df.to_csv(os.path.join(RESULTS_DIR, "noise_sensitivity.csv"), index=False)
    ns_fits = fit_small_delta_exponents(ns_df)
    plot_noise_sensitivity_curves(ns_df, ns_fits, os.path.join(FIGURES_DIR, "noise_sensitivity.png"))

    # 4. Random k-DNF total-influence-vs-k sweep (exploratory arm, no closed-form theorem)
    dnf_df = random_dnf_k_sweep(ns_n, k_values, deltas, ns_samples, seed0=1000)
    dnf_df.to_csv(os.path.join(RESULTS_DIR, "random_dnf_k_sweep.csv"), index=False)
    if len(dnf_df) >= 2:
        majority_ref_n = ns_n if ns_n % 2 == 1 else ns_n + 1
        majority_ref_influence = majority_ref_n * majority_influence_exact(majority_ref_n)
        tribes_ref_influence = balanced_w * balanced_s * tribes_influence_exact(balanced_w, balanced_s)
        plot_random_dnf_influence_trend(
            dnf_df, majority_ref_influence, tribes_ref_influence, ns_n,
            os.path.join(FIGURES_DIR, "random_dnf_influence_trend.png"),
        )

    # 5. Fixed-delta, growing-n regime: this is where Majority (-> Sheppard/CLT
    # limit) and Tribes (BKS: stays well below it) actually separate, unlike the
    # fixed-n small-delta exponents above, which converge to slope 1 for every
    # family since NS_delta(f) ~ delta*I(f)/2 as delta -> 0 at any fixed n.
    fixed_delta = 0.2
    vs_n_values = fixed_n_values
    vsn_df = noise_sensitivity_vs_n(fixed_delta, vs_n_values, ns_samples, seed=99)
    vsn_df.to_csv(os.path.join(RESULTS_DIR, "noise_sensitivity_vs_n.csv"), index=False)
    sheppard_limit = majority_sheppard_limit(fixed_delta)
    plot_noise_sensitivity_vs_n(vsn_df, sheppard_limit, fixed_delta,
                                 os.path.join(FIGURES_DIR, "noise_sensitivity_vs_n.png"))
    maj_vsn = vsn_df[vsn_df["family"] == "majority"].sort_values("n")
    tribes_vsn = vsn_df[vsn_df["family"] == "tribes"].sort_values("n")
    majority_gap_shrinks = bool(
        abs(maj_vsn["estimate"].iloc[-1] - sheppard_limit) < abs(maj_vsn["estimate"].iloc[0] - sheppard_limit)
    )
    tribes_stays_below_sheppard = bool((tribes_vsn["estimate"] < sheppard_limit - 3 * tribes_vsn["stderr"]).all())

    # 6. Universal KKL isoperimetric-ratio check across every instance computed above
    kkl_instances = []
    for _, row in maj_df.iterrows():
        kkl_instances.append({"family": "majority", "n": int(row["n"]), "max_influence": row["influence_per_coord"], "variance": 1.0})
    for _, row in tribes_df.iterrows():
        kkl_instances.append({"family": "tribes", "n": int(row["n"]), "max_influence": row["max_influence"], "variance": row["variance"]})
    kkl_df = kkl_check_all(kkl_instances)
    kkl_df.to_csv(os.path.join(RESULTS_DIR, "kkl_bound_check.csv"), index=False)
    plot_kkl_bound_check(kkl_df, os.path.join(FIGURES_DIR, "kkl_bound_check.png"))

    summary = {
        "majority_total_influence_exponent": maj_fit.exponent,
        "majority_total_influence_exponent_ci95": maj_fit.exponent_ci95(),
        "majority_theory_exponent": 0.5,
        "majority_fit_r_squared": maj_fit.r_squared,
        "tribes_max_influence_loglog_exponent": tribes_fit.exponent,
        "tribes_ratio_vs_n_slope": ratio_slope_fit.exponent,  # ~0 => ratio stabilizes, matching Theta(log n / n)
        "noise_sensitivity_small_delta_exponents": ns_fits,
        "noise_sensitivity_small_delta_caveat": (
            "At fixed n, NS_delta(f) ~ delta * I(f) / 2 as delta -> 0 for ANY "
            "Boolean function (first-order Taylor expansion of noise "
            "stability), so this small-delta exponent fit converges to ~1 for "
            "every family and does NOT distinguish Majority's asymptotic "
            "sqrt(delta) law from Tribes' Theta(delta) law -- see "
            "noise_sensitivity_vs_n_fixed_delta below for the regime that does."
        ),
        "noise_sensitivity_vs_n_fixed_delta": fixed_delta,
        "majority_converges_toward_sheppard_limit": majority_gap_shrinks,
        "majority_sheppard_limit_value": sheppard_limit,
        "tribes_stays_below_sheppard_limit": tribes_stays_below_sheppard,
        "kkl_min_ratio_observed": float(kkl_df["kkl_ratio"].min()),
        "kkl_bound_ever_violated": bool((kkl_df["kkl_ratio"] < 0).any()),
        "random_dnf_total_influence_by_k": dnf_df.to_dict(orient="records"),
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = main(quick=args.quick)
    print(json.dumps(result, indent=2))
