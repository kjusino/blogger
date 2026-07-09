"""Run the full experiment: (1) logical-error-rate threshold sweep for both
decoders across several code distances, and (2) a decode-time scaling sweep
comparing MWPM against the Union-Find-style clustering decoder.

Writes raw data to results/*.json and (via scripts/make_plots.py) figures to
results/*.png. Fully deterministic given the fixed seeds below, and safe to
re-run (overwrites its own output files).
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from toriccode.simulate import run_trials
from toriccode.stabilizer import ToricLattice

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"

THRESHOLD_L_VALUES = [5, 7, 9, 11]
THRESHOLD_P_VALUES = [round(p, 4) for p in np.linspace(0.03, 0.18, 13)]
THRESHOLD_SHOTS = 2000
THRESHOLD_DECODERS = ["mwpm", "union_find"]

SCALING_L_VALUES = [5, 9, 13, 17, 21, 25, 29, 33]
SCALING_P = 0.03
SCALING_SHOTS = 150


def run_threshold_sweep():
    records = []
    total = len(THRESHOLD_L_VALUES) * len(THRESHOLD_P_VALUES) * len(THRESHOLD_DECODERS)
    done = 0
    t_start = time.time()
    for L in THRESHOLD_L_VALUES:
        lattice = ToricLattice(L)
        for decoder in THRESHOLD_DECODERS:
            for p in THRESHOLD_P_VALUES:
                rng = np.random.default_rng(hash((L, decoder, p)) % (2**32))
                result = run_trials(lattice, p, decoder, THRESHOLD_SHOTS, rng)
                records.append(
                    {
                        "L": L,
                        "p": p,
                        "decoder": decoder,
                        "shots": result.shots,
                        "logical_errors": result.logical_errors,
                        "logical_error_rate": result.logical_error_rate,
                    }
                )
                done += 1
                elapsed = time.time() - t_start
                print(
                    f"[threshold {done}/{total}] L={L} decoder={decoder} p={p:.4f} "
                    f"-> p_L={result.logical_error_rate:.4f}  ({elapsed:.1f}s elapsed)",
                    flush=True,
                )
    return records


def run_scaling_sweep():
    records = []
    for L in SCALING_L_VALUES:
        lattice = ToricLattice(L)
        for decoder in ["mwpm", "union_find"]:
            rng = np.random.default_rng(hash((L, decoder, "scaling")) % (2**32))
            result = run_trials(lattice, SCALING_P, decoder, SCALING_SHOTS, rng)
            records.append(
                {
                    "L": L,
                    "decoder": decoder,
                    "shots": result.shots,
                    "mean_decode_seconds": result.mean_decode_seconds,
                }
            )
            print(
                f"[scaling] L={L} decoder={decoder} "
                f"mean_decode_seconds={result.mean_decode_seconds:.6f}",
                flush=True,
            )
    return records


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=== Running threshold sweep ===", flush=True)
    threshold_records = run_threshold_sweep()
    with open(RESULTS_DIR / "threshold_sweep.json", "w") as f:
        json.dump(threshold_records, f, indent=2)

    print("\n=== Running decode-time scaling sweep ===", flush=True)
    scaling_records = run_scaling_sweep()
    with open(RESULTS_DIR / "scaling_sweep.json", "w") as f:
        json.dump(scaling_records, f, indent=2)

    print("\nDone. Wrote results/threshold_sweep.json and results/scaling_sweep.json")


if __name__ == "__main__":
    main()
