"""Orchestrates the Monte Carlo sweep over (distance, physical error rate)
and assembles the summary statistics used to test the threshold theorem."""

from dataclasses import asdict

import numpy as np

from .circuits import build_surface_code_circuit
from .decode import DecodeResult, sample_and_decode
from .theory import estimate_threshold, fit_subthreshold_exponent


def seed_for(base_seed: int, distance: int, p_index: int) -> int:
    """Deterministic, collision-free per-(distance, p) seed."""
    return (base_seed * 1_000_003 + distance * 10_007 + p_index) % (2 ** 31 - 1)


def run_sweep(
    distances: list[int],
    p_values: np.ndarray,
    shots: int,
    base_seed: int = 0,
) -> dict[int, list[DecodeResult]]:
    """Run the full (distance x physical error rate) Monte Carlo sweep.

    Each circuit uses `rounds = distance` syndrome-extraction rounds, the
    standard choice for a memory experiment (enough rounds that boundary
    effects from the initial/final transversal round are a small
    correction relative to the bulk behaviour).
    """
    results: dict[int, list[DecodeResult]] = {}
    for d in distances:
        per_distance = []
        for i, p in enumerate(p_values):
            circuit = build_surface_code_circuit(distance=d, rounds=d, p=float(p))
            r = sample_and_decode(
                circuit,
                shots=shots,
                seed=seed_for(base_seed, d, i),
                distance=d,
                p=float(p),
            )
            per_distance.append(r)
        results[d] = per_distance
    return results


def summarize(
    p_values: np.ndarray,
    results: dict[int, list[DecodeResult]],
    subthreshold_fraction: float = 0.5,
) -> dict:
    """Build the JSON-serializable summary: raw rates + CIs, per-distance
    sub-threshold power-law fits, and the pairwise threshold estimate.

    `subthreshold_fraction`: fit the power law using only the lower this
    fraction of the swept p-range, since the ~p^k asymptotic only holds
    well below threshold.
    """
    p_values = np.asarray(p_values, dtype=float)
    rates_by_distance = {
        d: np.array([r.logical_error_rate for r in per_d])
        for d, per_d in results.items()
    }

    n_sub = max(2, int(len(p_values) * subthreshold_fraction))
    exponent_fits = {}
    for d, rates in rates_by_distance.items():
        try:
            fit = fit_subthreshold_exponent(p_values[:n_sub], rates[:n_sub], distance=d)
            exponent_fits[d] = asdict(fit)
        except ValueError as e:
            # Too few nonzero-rate points in the sub-threshold window to fit
            # a slope (can happen with few shots deep below threshold).
            exponent_fits[d] = {"error": str(e), "predicted_slope": (d + 1) // 2}

    threshold_estimate, pairwise_crossings = estimate_threshold(p_values, rates_by_distance)

    raw = []
    for d, per_d in results.items():
        for r in per_d:
            lo, hi = r.confidence_interval
            raw.append(
                {
                    "distance": d,
                    "p": r.p,
                    "shots": r.shots,
                    "logical_errors": r.logical_errors,
                    "logical_error_rate": r.logical_error_rate,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )

    return {
        "p_values": p_values.tolist(),
        "distances": sorted(results.keys()),
        "raw_results": raw,
        "subthreshold_exponent_fits": {str(k): v for k, v in exponent_fits.items()},
        "threshold_estimate": threshold_estimate,
        "pairwise_crossings": {
            f"{lo}-{hi}": v for (lo, hi), v in pairwise_crossings.items()
        },
    }
