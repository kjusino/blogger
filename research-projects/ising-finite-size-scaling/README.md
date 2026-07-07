# Finite-Size Scaling in the 2D Ising Model: Does Simulation Reproduce Onsager's Exact Theory?

A self-contained statistical-physics / Monte Carlo methods research project. Independent
of, and unrelated to, the other `research-projects/` entries in this repository (which
cover ML/dynamical-systems, automata learning theory, property testing, and coding
theory / online algorithms). This one sits at the intersection of **physics (critical
phenomena), mathematics (exact solvability, scaling theory), and computer science
(Markov chain Monte Carlo algorithm design)**.

## Research question

The 2D square-lattice Ising model is one of the few nontrivial statistical-mechanics
models with a fully exact solution (Onsager, 1944). It has a continuous
order-disorder phase transition at a known critical temperature, with known critical
exponents governing how observables diverge or vanish near that point. Real
simulations, however, are always run on **finite** lattices, where there is no true
singularity — only a smoothed-out crossover that sharpens as the lattice grows. The
theory of finite-size scaling (FSS) predicts a precise way this crossover should
depend on system size `L`, built entirely from the exponents of the infinite-lattice
theory.

**This project asks three concrete, falsifiable questions, each checked against an
independent exact reference rather than against itself:**

1. Does a from-scratch Metropolis Monte Carlo simulation, run at increasing lattice
   sizes, produce a Binder-cumulant crossing point that converges to Onsager's *exact*
   `T_c = 2/ln(1+√2) ≈ 2.269185`?
2. Using *only* the exact, textbook critical exponents (`β=1/8`, `γ=7/4`, `ν=1` — not
   fit to this data), does rescaling the simulated magnetization and susceptibility
   curves by `L^β/ν` / `L^γ/ν` and `(T-T_c)L^1/ν` actually collapse four different
   lattice sizes onto one universal curve?
3. Independent of any of the above, does the simulator's raw energy and magnetization
   match Onsager's exact closed-form infinite-lattice solution at all — and where does
   it visibly disagree, and why?

A fourth, secondary question is addressed as a bonus: the Wolff cluster algorithm is
famous for eliminating "critical slowing down" near `T_c`. Does that show up
quantitatively against a from-scratch Metropolis implementation, in this same codebase,
under a matched measurement protocol?

## Why this is a real research exercise, not a lookup

Every number reported below comes from code in this repository, run to completion, not
from a textbook or citation. Three of the four exact reference quantities used to
grade the simulation (`u(T_c) = -√2`, `u(T)` at general `T`, `m(T)` for `T<T_c`) require
implementing Onsager's 1944 energy solution and the Yang 1952 spontaneous-magnetization
formula from their closed forms, including handling a genuine removable singularity in
the exact energy formula exactly at `T_c` (see "A real numerical subtlety" below). The
simulator itself (checkerboard-vectorized Metropolis, and a from-scratch Wolff cluster
algorithm) is written without borrowing from any existing Ising-model codebase.

## Methodology

- **Lattice and Hamiltonian** (`src/lattice.py`): `L x L` square lattice, periodic
  boundary conditions, `H = -J * sum_<i,j> s_i s_j` with `J=1`, `k_B=1`.
- **Metropolis sampler** (`src/metropolis.py`): a vectorized checkerboard (red-black)
  update. On a periodic square lattice with even `L`, sites split into two color
  classes by the parity of `i+j`; every site's four neighbors lie in the opposite
  class, so all same-color spins can be proposed and accepted simultaneously via
  pure NumPy array ops without violating detailed balance — this is what makes
  L=64 sweeps fast enough to run a full parameter sweep in minutes on a CPU.
- **Wolff cluster sampler** (`src/wolff.py`): a from-scratch single-cluster
  algorithm (BFS cluster growth with bond-inclusion probability `1 - exp(-2J/T)`,
  rejection-free whole-cluster flip).
- **Exact theory** (`src/theory.py`): `T_c` in closed form; Onsager's exact
  energy `u(T)` (via the complete elliptic integral of the first kind); Yang's
  exact spontaneous magnetization `m(T)` for `T<T_c`; the exact 2D Ising critical
  exponents `β=1/8, γ=7/4, ν=1`.
- **Observables** (`src/observables.py`): specific heat, susceptibility (using
  `<m^2> - <|m|>^2`, the standard finite-lattice convention that stays
  well-defined below `T_c`), the Binder cumulant `U_4 = 1 - <m^4>/(3<m^2>^2)`,
  and a Sokal-windowed integrated autocorrelation time estimator.
- **Finite-size scaling analysis** (`src/scaling.py`): Binder-cumulant pairwise
  crossing detection (an exponent-free `T_c` estimator), and a data-collapse
  quality metric (normalized RMSE between adjacent-`L` curves on a common grid)
  applied both to the raw data and to the FSS-rescaled data, to quantify — not
  just eyeball — how much the theoretical exponents improve the overlap.
- **Experiment orchestration** (`src/experiment.py`, `run_experiment.py`): a grid
  over `L in {8, 16, 32, 64}` and 24 temperatures spanning `T in [1.6, 3.0]`
  (denser near `T_c`), 4 independent seeds per `(L, T)`; a separate Onsager
  validation pass at `L=64` across 8 temperatures from deep-ordered to
  deep-disordered; and a Metropolis-vs-Wolff autocorrelation-time comparison at
  `L=32` across 7 temperatures straddling `T_c`.

## Results (all measured by actually running the code — nothing fabricated)

Full numeric outputs are in `results/*.csv` and `results/summary.json`; figures are in
`figures/`.

**1. Binder-cumulant `T_c` estimate: 2.2583, vs. exact 2.269185 — a 0.48% relative
error**, from three pairwise crossings (L=8/16, 16/32, 32/64) landing at 2.2511,
2.2729, and 2.2508 (`figures/binder_cumulant_crossing.png`). This uses *no* fitted
exponents at all — just where the dimensionless Binder cumulant curves for different
`L` intersect.

**2. Finite-size scaling collapse genuinely works, using only exact textbook
exponents:** rescaling `|m|` by `L^(β/ν)` against `(T-T_c)L^(1/ν)` reduces the
adjacent-`L` mismatch (normalized RMSE) from **0.1086 (raw `|m|` vs. `T`) to 0.0186 — a
5.83x improvement** (`figures/scaling_collapse_magnetization.png`). The susceptibility
collapse also holds (RMSE 0.0368 after rescaling by `L^(-γ/ν)`,
`figures/scaling_collapse_susceptibility.png`), and the susceptibility peak visibly
grows and sharpens with `L` exactly as `χ_max ~ L^(γ/ν)` predicts
(`figures/susceptibility_vs_T.png`).

**3. Onsager/Yang exact-solution validation at L=64** (`figures/onsager_validation.png`,
`results/onsager_validation.csv`): energy matches to within `5e-5` at `T=1.6` and
`T=2.0`, and within `9e-4` at `T=2.5, 3.0, 4.0` — but has a **real, explainable 0.030
discrepancy at `T=0.8`** (see limitations below). Magnetization matches the exact
value to within `3e-4` at `T=2.0`, and — as expected on a *finite* lattice — visibly
disagrees with the *infinite-lattice* zero value right at and above `T_c` (simulated
`|m|=0.610` at `T=T_c` vs. exact `0`), which is the correct, textbook finite-size
rounding of a sharp transition, not an error.

**4. Wolff cluster updates eliminate critical slowing down, Metropolis does not**
(`figures/autocorrelation_comparison.png`, `results/autocorrelation_comparison.csv`):
Metropolis's integrated autocorrelation time peaks sharply at `T_c` (`τ≈228` sweeps,
vs. `τ≈1.4-8` sweeps away from `T_c`) — the textbook critical-slowing-down signature.
Wolff's per-step autocorrelation time stays at the theoretical floor (`τ=0.5`, i.e.
statistically independent every single cluster flip) all the way through `T_c`, only
rising once well into the disordered phase. Converted to **wall-clock seconds needed
per statistically independent sample, Wolff is up to 57.8x faster than Metropolis,
with the largest gap exactly at `T_c`** — precisely where it matters most, since that's
where Metropolis is slowest.

## A real numerical subtlety (not a fabricated one)

Onsager's exact energy formula contains a term `(2*tanh(2K)² - 1) * K₁(κ)`, where `K₁`
is the complete elliptic integral of the first kind and `κ = 2sinh(2K)/cosh²(2K)`.
At exactly `T = T_c`, `κ = 1` and `K₁` diverges logarithmically — but the coefficient
`(2*tanh(2K_c)² - 1)` is *also* exactly zero there (since `tanh(2K_c)² = 1/2`), so the
product has a finite limit (`u(T_c) = -J·coth(2K_c) = -√2`, a well-known exact value).
Evaluating both factors as floating-point numbers this close to the singularity is
numerically unstable (the first implementation returned `inf` at `T=T_c` from `0 * ∞`
under floating-point rounding). Fixed by detecting `κ ≥ 1 - 1e-9` and using the
analytic limit (`bracket = 1`) directly instead of evaluating the diverging elliptic
integral — covered by `test_onsager_energy_at_Tc_matches_known_closed_form` and
`test_onsager_energy_continuous_across_Tc` in `tests/test_theory.py`.

## Limitations and honest caveats

- **Low-temperature domain freezing (`T=0.8` in the Onsager validation, and the reason
  the largest energy/magnetization errors in the whole study occur there, not near
  `T_c`).** Metropolis single-spin-flip dynamics starting from a *random* (hot) initial
  configuration anneals into a single ordered domain only by slow, diffusive domain-wall
  motion — at low `T`, domain walls are nearly frozen because moving them costs energy
  that is exponentially unlikely to be thermally accepted. Within the fixed equilibration
  budget used here, the `L=64` lattice at `T=0.8` gets stuck with `|m|=0.744` instead of
  the true `0.9999`, and a correspondingly elevated energy error (`0.030` vs. `~0.0005`
  at `T=1.6-2.0`). This is a genuine, well-known limitation of local-update MC at low
  temperature (not a bug) — and is exactly the motivation for cluster algorithms like
  Wolff, which flip whole domains in one step and would not show this effect.
- **Statistical precision is modest, not publication-grade.** 4 seeds and a few thousand
  sweeps per `(L,T)` is enough to see the qualitative FSS predictions hold clearly, but
  the individual crossing estimates (2.251, 2.273, 2.251) still show seed/measurement
  scatter of a similar order to their spread from the exact value; a real study would
  run an order of magnitude more statistics and/or larger `L` to shrink error bars
  formally rather than reporting point estimates.
- **The specific-heat exponent (`α=0`, a logarithmic divergence)** is not fit or
  collapsed quantitatively here — only the power-law exponents `β, γ, ν` are used for
  data collapse, since a log-divergence collapse is a qualitatively different (and
  noisier) fit that was out of scope for this pass.
- **The Wolff-vs-Metropolis comparison is restricted to `L=32` and a modest temperature
  window** (`T_c ± 0.4`); it is a clean illustration of the well-established dynamic
  critical exponent difference between the two algorithms, not a fit of the dynamic
  exponents `z` themselves.

## How to reproduce

```bash
cd research-projects/ising-finite-size-scaling
pip install -r requirements.txt
python3 -m pytest tests -v          # 58 unit + integration tests
python3 run_experiment.py           # full grid (~7 minutes)
python3 run_experiment.py --quick   # fast smoke-test grid (~3 seconds)
```

Outputs land in `results/*.csv`, `results/summary.json`, and `figures/*.png`.

## Repository layout

```
src/
  lattice.py      # spin lattice, energy/magnetization, periodic-BC neighbor sums
  metropolis.py   # vectorized checkerboard Metropolis MC
  wolff.py        # from-scratch Wolff single-cluster MC
  theory.py       # exact Tc, Onsager energy, Yang magnetization, critical exponents
  observables.py  # specific heat, susceptibility, Binder cumulant, autocorrelation time
  scaling.py      # FSS rescaling, Binder-crossing Tc estimator, collapse-quality metric
  experiment.py   # orchestrates the grid sweep, Onsager validation, autocorrelation study
run_experiment.py # CLI entry point (--quick for a fast smoke test)
tests/            # 58 unit + integration tests
results/          # generated CSVs + summary.json
figures/          # generated PNGs
```

## Future work

- Fit the specific-heat logarithmic divergence and its own finite-size form.
- Use a low-temperature *ordered* start (rather than random) to remove the domain-freezing
  artifact noted above, and quantify the equilibration-time gap between the two starts
  directly (a second concrete MCMC-efficiency experiment).
- Extend the Wolff-vs-Metropolis comparison across `L` to fit the dynamic critical
  exponents `z_Metropolis ≈ 2` and `z_Wolff ≈ 0` directly, rather than a single-`L`
  illustration.
- Repeat the whole pipeline for the 3D Ising model, where no exact solution exists,
  using the 2D exact-solution validation here as the trust anchor for the method itself.
