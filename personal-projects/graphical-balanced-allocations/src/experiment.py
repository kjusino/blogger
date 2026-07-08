"""Sweep orchestration: run every graph family across every n, several
trials each, and collect max-load-gap + spectral-gap statistics into
tidy pandas DataFrames.
"""

import numpy as np
import pandas as pd

from . import allocation, graphs

DEFAULT_NS = (128, 256, 512, 1024, 2048)
DEFAULT_TRIALS = 30

# these families are entirely determined by n (generate_graph ignores the
# seed for them), so their spectral gap is identical across all "trials" --
# computing it once per (family, n) instead of once per trial avoids 29/30
# redundant eigendecompositions of the same graph.
DETERMINISTIC_FAMILIES = frozenset({"complete", "cycle", "path", "torus"})


def run_family_trials(family: str, n: int, trials: int, seed: int):
    """Run `trials` independent instances of `family` at size `n`.

    Returns (actual_n, spectral_gap, list_of_max_load_gaps). A fresh graph
    is drawn per trial so results reflect the family, not one lucky/unlucky
    instance.
    """
    gaps = []
    spectral_gaps = []
    actual_n = n
    is_deterministic = family in DETERMINISTIC_FAMILIES
    cached_graph = None
    for t in range(trials):
        trial_seed = seed + t
        if is_deterministic and cached_graph is not None:
            G, edges = cached_graph
        else:
            G, edges = graphs.generate_graph(family, n, seed=trial_seed)
            if is_deterministic:
                cached_graph = (G, edges)
        actual_n = G.number_of_nodes()
        rng = np.random.default_rng(trial_seed)
        loads = allocation.simulate_graphical_two_choice(edges, actual_n, actual_n, rng)
        gaps.append(allocation.max_load_gap(loads, actual_n, actual_n))
        if is_deterministic and spectral_gaps:
            spectral_gaps.append(spectral_gaps[0])
        else:
            spectral_gaps.append(graphs.spectral_gap(G))
    return actual_n, float(np.mean(spectral_gaps)), gaps


def run_baseline_trials(kind: str, n: int, trials: int, seed: int):
    gaps = []
    for t in range(trials):
        rng = np.random.default_rng(seed + t)
        if kind == "one_choice":
            loads = allocation.simulate_one_choice(n, n, rng)
        elif kind == "classical_two_choice":
            loads = allocation.simulate_classical_two_choice(n, n, rng)
        else:
            raise ValueError(f"unknown baseline kind: {kind!r}")
        gaps.append(allocation.max_load_gap(loads, n, n))
    return gaps


def run_sweep(
    families=graphs.FAMILIES,
    ns=DEFAULT_NS,
    trials=DEFAULT_TRIALS,
    seed=0,
    progress_cb=None,
):
    """Run the full sweep. Returns (raw_df, summary_df).

    raw_df has one row per (family, n, trial). summary_df aggregates to
    one row per (family, n) with mean/std gap and mean spectral gap.
    """
    raw_rows = []
    for family in families:
        for n in ns:
            actual_n, mean_spectral_gap, trial_gaps = run_family_trials(
                family, n, trials, seed=hash((family, n)) % (2**31)
            )
            for trial_idx, gap in enumerate(trial_gaps):
                raw_rows.append(
                    {
                        "family": family,
                        "requested_n": n,
                        "n": actual_n,
                        "trial": trial_idx,
                        "max_load_gap": gap,
                        "spectral_gap": mean_spectral_gap,
                    }
                )
            if progress_cb:
                progress_cb(family, n)

    for kind in ("one_choice", "classical_two_choice"):
        for n in ns:
            trial_gaps = run_baseline_trials(kind, n, trials, seed=hash((kind, n)) % (2**31))
            for trial_idx, gap in enumerate(trial_gaps):
                raw_rows.append(
                    {
                        "family": kind,
                        "requested_n": n,
                        "n": n,
                        "trial": trial_idx,
                        "max_load_gap": gap,
                        "spectral_gap": np.nan,
                    }
                )
            if progress_cb:
                progress_cb(kind, n)

    raw_df = pd.DataFrame(raw_rows)
    summary_df = (
        raw_df.groupby(["family", "requested_n"])
        .agg(
            n=("n", "first"),
            mean_gap=("max_load_gap", "mean"),
            std_gap=("max_load_gap", "std"),
            mean_spectral_gap=("spectral_gap", "mean"),
        )
        .reset_index()
        .sort_values(["family", "n"])
        .reset_index(drop=True)
    )
    return raw_df, summary_df
