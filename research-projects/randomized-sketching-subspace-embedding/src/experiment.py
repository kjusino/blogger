"""Sweep definitions for the randomized-sketching / subspace-embedding study, plus
the FULL and SMOKE parameter configs used by run_experiment.py.
"""

import time

import numpy as np

from .sketches import SKETCHES, srht_sketch
from .subspace import coherent_basis, incoherent_basis, random_least_squares_system
from .theory import predicted_k, subspace_distortion

FULL_CONFIG = dict(
    d=30, n=4096, eps_target=0.25, delta=0.1, const=2.0,
    trials=25,
    threshold_multipliers=[0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    scaling_trials=25,
    scaling_k_grid=[60, 120, 240, 480, 960, 1920],
    coherence_trials=25,
    coherence_k_grid=[60, 120, 240, 480, 960, 1920],
    ls_trials=25,
    ls_multipliers=[0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    ls_noise=0.1,
    timing_n=8192, timing_d=50,
    timing_k_grid=[50, 100, 200, 400, 800, 1600, 3200],
    timing_n_grid=[512, 1024, 2048, 4096, 8192, 16384],
    timing_k_fixed=300,
    timing_repeats=5,
)

SMOKE_CONFIG = dict(
    d=8, n=1024, eps_target=0.4, delta=0.2, const=2.0,
    trials=3,
    threshold_multipliers=[0.3, 1.0, 2.5],
    scaling_trials=3,
    scaling_k_grid=[20, 40, 80, 160, 320],
    coherence_trials=3,
    coherence_k_grid=[20, 40, 80, 160, 320],
    ls_trials=3,
    ls_multipliers=[0.5, 1.0, 2.0],
    ls_noise=0.1,
    timing_n=512, timing_d=12,
    timing_k_grid=[16, 32, 64],
    timing_n_grid=[128, 256, 512],
    timing_k_fixed=32,
    timing_repeats=2,
)


def _clip_k(k, d, n):
    return int(max(d + 2, min(n - 1, round(k))))


def run_threshold_sweep(cfg, rng):
    d, n, eps_target, trials = cfg["d"], cfg["n"], cfg["eps_target"], cfg["trials"]
    k0 = predicted_k(d, eps_target, cfg["delta"], cfg["const"])
    rows = []
    for name, fn in SKETCHES.items():
        for m in cfg["threshold_multipliers"]:
            k = _clip_k(k0 * m, d, n)
            distortions = []
            for _ in range(trials):
                Q = incoherent_basis(n, d, rng)
                distortions.append(subspace_distortion(fn(Q, k, rng)))
            distortions = np.array(distortions)
            rows.append(dict(
                sketch=name, k=k, k_over_k0=k / k0,
                success_rate=float(np.mean(distortions <= eps_target)),
                median_eps=float(np.median(distortions)),
                d=d, n=n, eps_target=eps_target,
            ))
    return rows, k0


def run_scaling_sweep(cfg, rng):
    d, n, trials = cfg["d"], cfg["n"], cfg["scaling_trials"]
    rows = []
    for name, fn in SKETCHES.items():
        for k in cfg["scaling_k_grid"]:
            distortions = []
            for _ in range(trials):
                Q = incoherent_basis(n, d, rng)
                distortions.append(subspace_distortion(fn(Q, k, rng)))
            rows.append(dict(sketch=name, k=k, median_eps=float(np.median(distortions)), d=d, n=n))
    return rows


def run_coherence_ablation(cfg, rng):
    d, n, trials = cfg["d"], cfg["n"], cfg["coherence_trials"]
    variants = [
        ("srht_precond", lambda A, k, r: srht_sketch(A, k, r, precondition=True)),
        ("uniform_sampling", lambda A, k, r: srht_sketch(A, k, r, precondition=False)),
    ]
    bases = [("incoherent", incoherent_basis), ("coherent", coherent_basis)]
    rows = []
    for basis_name, basis_fn in bases:
        for variant_name, fn in variants:
            for k in cfg["coherence_k_grid"]:
                distortions = []
                for _ in range(trials):
                    Q = basis_fn(n, d, rng)
                    distortions.append(subspace_distortion(fn(Q, k, rng)))
                rows.append(dict(
                    basis=basis_name, variant=variant_name, k=k,
                    median_eps=float(np.median(distortions)),
                    max_eps=float(np.max(distortions)), d=d, n=n,
                ))
    return rows


def run_least_squares_sweep(cfg, rng):
    d, n, trials, eps_target = cfg["d"], cfg["n"], cfg["ls_trials"], cfg["eps_target"]
    k0 = predicted_k(d + 1, eps_target, cfg["delta"], cfg["const"])
    rows = []
    for name, fn in SKETCHES.items():
        for m in cfg["ls_multipliers"]:
            k = _clip_k(k0 * m, d + 1, n)
            rel_errors = []
            for _ in range(trials):
                Q = incoherent_basis(n, d, rng)
                A, b, _ = random_least_squares_system(Q, rng, noise=cfg["ls_noise"])
                x_opt, *_ = np.linalg.lstsq(A, b, rcond=None)
                opt_res = np.linalg.norm(A @ x_opt - b)
                augmented = fn(np.hstack([A, b[:, None]]), k, rng)
                SA, Sb = augmented[:, :-1], augmented[:, -1]
                x_hat, *_ = np.linalg.lstsq(SA, Sb, rcond=None)
                sketch_res = np.linalg.norm(A @ x_hat - b)
                rel_errors.append((sketch_res - opt_res) / opt_res if opt_res > 1e-12 else 0.0)
            rel_errors = np.array(rel_errors)
            rows.append(dict(
                sketch=name, k=k, k_over_k0=k / k0,
                median_rel_excess=float(np.median(rel_errors)),
                p90_rel_excess=float(np.percentile(rel_errors, 90)),
                d=d, n=n, eps_target=eps_target,
            ))
    return rows, k0


def _time_sketch(fn, A, k, rng, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(A, k, rng)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def run_timing_vs_k(cfg, rng):
    n, d, repeats = cfg["timing_n"], cfg["timing_d"], cfg["timing_repeats"]
    A = rng.standard_normal((n, d))
    rows = []
    for name, fn in SKETCHES.items():
        for k in cfg["timing_k_grid"]:
            rows.append(dict(sketch=name, k=k, n=n, d=d, time_s=_time_sketch(fn, A, k, rng, repeats)))
    return rows


def run_timing_vs_n(cfg, rng):
    d, k_fixed, repeats = cfg["timing_d"], cfg["timing_k_fixed"], cfg["timing_repeats"]
    rows = []
    for name, fn in SKETCHES.items():
        for n in cfg["timing_n_grid"]:
            A = rng.standard_normal((n, d))
            k = min(k_fixed, n - 1)
            rows.append(dict(sketch=name, k=k, n=n, d=d, time_s=_time_sketch(fn, A, k, rng, repeats)))
    return rows


def run_all(cfg, seed):
    rng = np.random.default_rng(seed)
    threshold_rows, k0_threshold = run_threshold_sweep(cfg, rng)
    ls_rows, k0_ls = run_least_squares_sweep(cfg, rng)
    return dict(
        config=cfg,
        threshold=threshold_rows,
        k0_threshold=k0_threshold,
        scaling=run_scaling_sweep(cfg, rng),
        coherence=run_coherence_ablation(cfg, rng),
        least_squares=ls_rows,
        k0_least_squares=k0_ls,
        timing_vs_k=run_timing_vs_k(cfg, rng),
        timing_vs_n=run_timing_vs_n(cfg, rng),
    )
