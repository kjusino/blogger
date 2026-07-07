"""Experiment grid: sigma x T, comparing the RDP accountant's theoretical
epsilon against the Monte Carlo membership-inference audit's empirical
eps_lower.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .accountant import epsilon_from_rdp
from .audit import run_membership_audit
from .data import make_dataset

DEFAULT_SIGMAS = (0.5, 1.0, 2.0)
DEFAULT_TS = (1, 2, 4, 8)
DEFAULT_C = 1.0
DEFAULT_LR = 0.1
DEFAULT_DELTA = 1e-5
DEFAULT_N = 100
DEFAULT_D = 5


def run_grid(
    N: int,
    seed: int = 0,
    sigmas=DEFAULT_SIGMAS,
    Ts=DEFAULT_TS,
    C: float = DEFAULT_C,
    lr: float = DEFAULT_LR,
    delta: float = DEFAULT_DELTA,
    n: int = DEFAULT_N,
    d: int = DEFAULT_D,
):
    """Run the full sigma x T grid. Returns (rows, roc_records).

    rows: list of dicts, one per (sigma, T) config, with theory/audit results.
    roc_records: dict keyed by (sigma, T) -> the audit's ROC list, for
        plotting the empirical ROC curves.
    """
    master_rng = np.random.default_rng(seed)
    data_rng = np.random.default_rng(seed + 1)
    X, y = make_dataset(n=n, d=d, rng=data_rng)
    theta0 = np.zeros(d)

    rows = []
    roc_records = {}

    for sigma in sigmas:
        for T in Ts:
            t_start = time.time()
            eps_theory, alpha_star = epsilon_from_rdp(sigma, T, delta)

            trial_seed = int(master_rng.integers(0, 2 ** 32 - 1))
            trial_rng = np.random.default_rng(trial_seed)
            audit_result = run_membership_audit(
                X, y, theta0, T, C, sigma, lr, N, delta, trial_rng
            )
            elapsed = time.time() - t_start

            eps_lower = audit_result["eps_lower"]
            ratio = eps_lower / eps_theory if eps_theory not in (0, float("inf")) else float("nan")

            rows.append(
                {
                    "sigma": sigma,
                    "T": T,
                    "C": C,
                    "lr": lr,
                    "delta": delta,
                    "N": N,
                    "epsilon_theory": eps_theory,
                    "alpha_star": alpha_star,
                    "eps_lower": eps_lower,
                    "ratio_audit_over_theory": ratio,
                    "tp": audit_result["tp"],
                    "fp": audit_result["fp"],
                    "mean_in": audit_result["mean_in"],
                    "mean_out": audit_result["mean_out"],
                    "mean_shift_is_negative": audit_result["mean_shift_is_negative"],
                    "elapsed_sec": elapsed,
                }
            )
            roc_records[(sigma, T)] = audit_result["roc"]

    return rows, roc_records


def save_results(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "grid_results.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "n_configs": len(rows),
        "sigmas": sorted({r["sigma"] for r in rows}),
        "Ts": sorted({r["T"] for r in rows}),
        "N_per_world": rows[0]["N"],
        "results": rows,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
