"""Orchestrates the grid sweep over {train_size} x {noise_level} x
{hidden_width} (with a couple of random seeds per cell), training a
surrogate for each configuration and recording:

  - one-step validation MSE (the metric ML papers usually report),
  - the learned leading Lyapunov exponent lambda1 vs the ground-truth
    Lorenz-63 value, and |error|,
  - whether sign(lambda1) chaos-detection was correct,
  - the attractor-divergence metric (marginal JS divergence vs the true
    long-horizon Lorenz trajectory).

Saves results to a CSV table and produces the four summary figures used in
the README. Also exposes a "quick" mode (tiny grid, few epochs) for fast
end-to-end integration testing.
"""
from __future__ import annotations

import csv
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .dynamics import lorenz_flow_map
from .lyapunov import lyapunov_spectrum, finite_difference_jacobian
from .surrogate import (
    generate_trajectory, make_training_pairs, train_surrogate,
    make_surrogate_map, make_surrogate_jacobian, surrogate_val_mse,
    iterate_surrogate,
)
from .attractor_metrics import marginal_js_divergence

DT = 0.01
TRUE_LAMBDA1_LITERATURE = 0.905  # Wolf et al. 1985 / Sprott reference value

RESULT_FIELDS = [
    "train_size", "noise_level", "hidden_width", "seed",
    "val_mse", "lambda1_true", "lambda1_surrogate", "lambda1_abs_error",
    "chaos_detected_correct", "attractor_js_divergence", "train_seconds",
]


@dataclass
class GridSpec:
    train_sizes: List[int] = field(default_factory=lambda: [200, 1000, 5000])
    noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.02, 0.1])
    hidden_widths: List[int] = field(default_factory=lambda: [8, 32, 128])
    seeds: List[int] = field(default_factory=lambda: [0, 1])
    val_size: int = 400
    n_hidden_layers: int = 2
    lyap_iters: int = 2000
    lyap_warmup: int = 300
    attractor_steps: int = 4000
    true_traj_len: int = 20_000


def quick_grid_spec() -> GridSpec:
    """A drastically reduced grid for fast (few-second) integration tests."""
    return GridSpec(
        train_sizes=[80],
        noise_levels=[0.0, 0.1],
        hidden_widths=[8],
        seeds=[0],
        val_size=50,
        n_hidden_layers=1,
        lyap_iters=150,
        lyap_warmup=50,
        attractor_steps=200,
        true_traj_len=1000,
    )


def _epochs_for_train_size(train_size: int, quick: bool = False) -> int:
    """Fewer epochs for larger datasets, since more gradient steps happen
    per epoch -- keeps total compute roughly balanced across the grid while
    still fitting the small (200-5000 sample) datasets well."""
    if quick:
        return 25
    if train_size <= 300:
        return 400
    if train_size <= 1500:
        return 150
    return 60


def compute_true_lambda1(n_iters: int = 20_000, warmup: int = 2_000,
                          dt: float = DT) -> float:
    step = lambda s: lorenz_flow_map(s, dt)
    x0 = np.array([1.0, 1.0, 1.0])
    exponents = lyapunov_spectrum(step, x0, n_iters=n_iters, dt=dt,
                                   warmup=warmup, renorm_interval=1)
    return float(exponents[0])


def _safe_surrogate_lambda1(net, mean, std, x0, n_iters, warmup, dt):
    """Estimate the surrogate's leading Lyapunov exponent, guarding against
    numerically unstable surrogates (a real failure mode for very
    low-data/high-noise/high-capacity configs) that could otherwise produce
    NaN/inf and crash the sweep.
    """
    step = make_surrogate_map(net, mean, std)
    jac = make_surrogate_jacobian(net, mean, std)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            exponents = lyapunov_spectrum(step, x0, n_iters=n_iters,
                                           jacobian_fn=jac, dt=dt,
                                           warmup=warmup, renorm_interval=1)
        lambda1 = float(exponents[0])
        if not np.isfinite(lambda1):
            return float("nan")
        return lambda1
    except Exception:
        return float("nan")


def _train_and_evaluate(train_size: int, noise_level: float, hidden_width: int,
                         seed: int, spec: GridSpec, lambda1_true: float,
                         true_long_traj: np.ndarray, quick: bool = False):
    """Train one surrogate config and evaluate all metrics for it. Returns
    (row_dict, net, mean, std, X_train, sur_traj) so callers can either just
    record the row (the grid sweep) or additionally reuse the trained
    surrogate/trajectory for plotting (figure generation).
    """
    data_seed = 1000 * seed + int(noise_level * 10_000) + train_size
    X_train, Y_train = make_training_pairs(
        train_size, dt=DT, noise_level=noise_level,
        seed=data_seed, warmup_steps=500,
    )
    val_seed = data_seed + 500_000
    X_val, Y_val = make_training_pairs(
        spec.val_size, dt=DT, noise_level=noise_level,
        seed=val_seed, warmup_steps=800,
    )

    t0 = time.time()
    epochs = _epochs_for_train_size(train_size, quick=quick)
    net, mean, std = train_surrogate(
        X_train, Y_train, hidden_width=hidden_width,
        n_hidden_layers=spec.n_hidden_layers, epochs=epochs,
        batch_size=32, lr=2e-3, seed=seed,
    )
    train_seconds = time.time() - t0

    val_mse = surrogate_val_mse(net, mean, std, X_val, Y_val)

    x0 = X_train[0].copy()
    lambda1_sur = _safe_surrogate_lambda1(
        net, mean, std, x0, spec.lyap_iters, spec.lyap_warmup, DT,
    )

    if np.isfinite(lambda1_sur):
        lambda1_abs_error = abs(lambda1_sur - lambda1_true)
        chaos_correct = bool((lambda1_sur > 0) == (lambda1_true > 0))
    else:
        lambda1_abs_error = float("nan")
        chaos_correct = False

    sur_start = X_train[min(50, len(X_train) - 1)].copy()
    sur_traj = iterate_surrogate(net, mean, std, sur_start, spec.attractor_steps)
    js = marginal_js_divergence(true_long_traj, sur_traj)["mean"]

    row = {
        "train_size": train_size,
        "noise_level": noise_level,
        "hidden_width": hidden_width,
        "seed": seed,
        "val_mse": val_mse,
        "lambda1_true": lambda1_true,
        "lambda1_surrogate": lambda1_sur,
        "lambda1_abs_error": lambda1_abs_error,
        "chaos_detected_correct": chaos_correct,
        "attractor_js_divergence": js,
        "train_seconds": train_seconds,
    }
    return row, net, mean, std, X_train, sur_traj


def run_grid(spec: GridSpec, out_dir: Optional[str] = None,
             verbose: bool = True, quick: bool = False) -> List[dict]:
    """Run the full (or quick) grid sweep. Returns a list of result-row
    dicts (one per (train_size, noise_level, hidden_width, seed) config).
    """
    t_start = time.time()

    if verbose:
        print("Computing ground-truth Lyapunov spectrum of true Lorenz flow map...")
    lambda1_true = compute_true_lambda1(
        n_iters=2_000 if quick else 20_000,
        warmup=200 if quick else 2_000,
    )
    if verbose:
        print(f"  lambda1_true = {lambda1_true:.4f} (literature ~{TRUE_LAMBDA1_LITERATURE})")

    if verbose:
        print("Generating long reference true trajectory for attractor comparison...")
    true_long_traj = generate_trajectory(spec.true_traj_len, dt=DT,
                                          warmup_steps=1000, seed=123)

    rows: List[dict] = []
    total_configs = (len(spec.train_sizes) * len(spec.noise_levels)
                     * len(spec.hidden_widths) * len(spec.seeds))
    done = 0

    for train_size in spec.train_sizes:
        for noise_level in spec.noise_levels:
            for seed in spec.seeds:
                for hidden_width in spec.hidden_widths:
                    row, *_ = _train_and_evaluate(
                        train_size, noise_level, hidden_width, seed, spec,
                        lambda1_true, true_long_traj, quick=quick,
                    )
                    rows.append(row)
                    done += 1
                    if verbose:
                        print(
                            f"[{done}/{total_configs}] train_size={train_size:5d} "
                            f"noise={noise_level:.2f} width={hidden_width:3d} "
                            f"seed={seed} -> val_mse={row['val_mse']:.4g} "
                            f"lambda1={row['lambda1_surrogate']:.4f} "
                            f"(|err|={row['lambda1_abs_error']:.4f}) "
                            f"chaos_ok={row['chaos_detected_correct']} "
                            f"js={row['attractor_js_divergence']:.4f} "
                            f"[{row['train_seconds']:.1f}s]"
                        )

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_results_csv(rows, os.path.join(out_dir, "grid_results.csv"))

    if verbose:
        print(f"Grid sweep completed in {time.time() - t_start:.1f}s "
              f"({total_configs} configs).")

    return rows


def train_representative_surrogate(train_size: int, noise_level: float,
                                    hidden_width: int, seed: int,
                                    spec: Optional[GridSpec] = None):
    """Retrain a single specific config (cheap -- a few seconds) and return
    everything needed to plot its long-horizon attractor against the true
    Lorenz attractor: (row, net, mean, std, sur_traj, true_long_traj).
    Used by the figure-generation step to visualize one representative
    surrogate without needing to keep all 54 grid models in memory.
    """
    spec = spec or GridSpec()
    lambda1_true = compute_true_lambda1(n_iters=20_000, warmup=2_000)
    true_long_traj = generate_trajectory(spec.true_traj_len, dt=DT,
                                          warmup_steps=1000, seed=123)
    row, net, mean, std, X_train, sur_traj = _train_and_evaluate(
        train_size, noise_level, hidden_width, seed, spec,
        lambda1_true, true_long_traj,
    )
    return row, net, mean, std, sur_traj, true_long_traj


def save_results_csv(rows: List[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_results_csv(path: str) -> List[dict]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append({
                "train_size": int(r["train_size"]),
                "noise_level": float(r["noise_level"]),
                "hidden_width": int(r["hidden_width"]),
                "seed": int(r["seed"]),
                "val_mse": float(r["val_mse"]),
                "lambda1_true": float(r["lambda1_true"]),
                "lambda1_surrogate": float(r["lambda1_surrogate"]),
                "lambda1_abs_error": float(r["lambda1_abs_error"]),
                "chaos_detected_correct": r["chaos_detected_correct"] == "True",
                "attractor_js_divergence": float(r["attractor_js_divergence"]),
                "train_seconds": float(r["train_seconds"]),
            })
        return rows
