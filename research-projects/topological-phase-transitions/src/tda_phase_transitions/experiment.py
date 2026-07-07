"""Experiment orchestrator: sweep n and trials across 3 random-graph
families, compare topological detectors against theoretical percolation
thresholds, save a results table and diagnostic plots.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from . import cycle_onset, graph_models, percolation, persistence, theory

MODEL_NAMES = ("er", "rgg", "chung_lu")


@dataclass
class TrialResult:
    model: str
    n: int
    seed: int
    theory_threshold: float
    percolation_threshold: float
    cycle_onset_threshold: Optional[float]
    t_max: float

    def as_row(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "n": self.n,
            "seed": self.seed,
            "theory_threshold": self.theory_threshold,
            "percolation_threshold": self.percolation_threshold,
            "cycle_onset_threshold": self.cycle_onset_threshold,
            "t_max": self.t_max,
            "percolation_ratio": self.percolation_threshold / self.theory_threshold,
            "cycle_onset_ratio": (
                self.cycle_onset_threshold / self.theory_threshold
                if self.cycle_onset_threshold is not None
                else None
            ),
        }


def _build_model(model: str, n: int, seed: int):
    """Return (dist, t_max, theory_threshold) for the given model."""
    if model == "er":
        m = graph_models.er_distance_matrix(n, seed=seed)
        t_max = min(1.0, 8.0 / n)
        thr = theory.er_giant_component_threshold(n)
        return m.dist, t_max, thr
    if model == "rgg":
        m = graph_models.rgg_distance_matrix(n, seed=seed)
        t_max = min(0.5, float(np.sqrt(10.0 / (np.pi * n))))
        thr = theory.rgg_giant_component_threshold(n)
        return m.dist, t_max, thr
    if model == "chung_lu":
        m = graph_models.chung_lu_distance_matrix(n, gamma=2.5, seed=seed)
        thr = theory.chung_lu_giant_component_threshold(m.weights)
        t_max = 4.0 * thr
        return m.dist, t_max, thr
    raise ValueError(f"unknown model {model!r}")


def run_trial(
    model: str,
    n: int,
    seed: int,
    grid_points: int = 200,
    birth_quantile: float = 0.1,
    return_curves: bool = False,
) -> TrialResult:
    dist, t_max, thr = _build_model(model, n, seed)
    thresholds = np.linspace(0.0, t_max, grid_points)

    dgms = persistence.compute_persistence(dist, maxdim=1, thresh=t_max)
    giant_frac, chi = percolation.percolation_curve(dist, thresholds)
    perc_threshold = float(thresholds[np.argmax(chi)])

    min_persistence = 0.01 * t_max
    onset = cycle_onset.cycle_onset_threshold(
        dgms[1], birth_quantile=birth_quantile, min_persistence=min_persistence
    )

    result = TrialResult(
        model=model,
        n=n,
        seed=seed,
        theory_threshold=thr,
        percolation_threshold=perc_threshold,
        cycle_onset_threshold=onset,
        t_max=t_max,
    )
    if return_curves:
        return result, {
            "thresholds": thresholds,
            "beta0": persistence.betti_curve(dgms[0], thresholds),
            "beta1": persistence.betti_curve(dgms[1], thresholds),
            "giant_frac": giant_frac,
            "susceptibility": chi,
            "dgms": dgms,
        }
    return result


def run_sweep(
    models: List[str] = list(MODEL_NAMES),
    n_values: List[int] = (50, 100, 200),
    trials: int = 20,
    base_seed: int = 0,
    progress: Optional[Callable[[str], None]] = None,
) -> List[TrialResult]:
    results = []
    seed_counter = base_seed
    for model in models:
        for n in n_values:
            for trial in range(trials):
                seed_counter += 1
                res = run_trial(model, n, seed=seed_counter)
                results.append(res)
                if progress is not None:
                    progress(
                        f"{model} n={n} trial={trial + 1}/{trials} "
                        f"perc_ratio={res.percolation_threshold / res.theory_threshold:.2f}"
                    )
    return results


def save_results_csv(results: List[TrialResult], path: str) -> None:
    rows = [r.as_row() for r in results]
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(results: List[TrialResult], path: str) -> Dict:
    from scipy import stats

    summary = {}
    for model in MODEL_NAMES:
        model_results = [r for r in results if r.model == model]
        if not model_results:
            continue
        perc_ratios = np.array([r.percolation_threshold / r.theory_threshold for r in model_results])
        onset_ratios = np.array(
            [
                r.cycle_onset_threshold / r.theory_threshold
                for r in model_results
                if r.cycle_onset_threshold is not None
            ]
        )
        entry = {
            "n_trials": len(model_results),
            "percolation_ratio_mean": float(perc_ratios.mean()),
            "percolation_ratio_std": float(perc_ratios.std()),
            "cycle_onset_ratio_mean": float(onset_ratios.mean()) if onset_ratios.size else None,
            "cycle_onset_ratio_std": float(onset_ratios.std()) if onset_ratios.size else None,
            "n_missing_cycle_onset": len(model_results) - onset_ratios.size,
        }
        if onset_ratios.size == perc_ratios.size and onset_ratios.size > 1:
            stat, p = stats.wilcoxon(perc_ratios, onset_ratios)
            entry["wilcoxon_statistic"] = float(stat)
            entry["wilcoxon_p_value"] = float(p)
        summary[model] = entry

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary
