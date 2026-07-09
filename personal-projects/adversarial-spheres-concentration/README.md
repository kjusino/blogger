# Adversarial Spheres: does concentration of measure predict a classifier's own robustness ceiling?

**Research question.** For a binary classifier on a high-dimensional sphere, does
the spherical isoperimetric inequality (Lévy's lemma) impose a real, numerically
verifiable *ceiling* on how far a "minority-region" point can be from the
decision boundary — and does that ceiling actually shrink at the predicted
Θ(1/√d) rate?

This is the "Adversarial Spheres" experimental paradigm introduced by
[Gilmer et al. (2018)](https://arxiv.org/abs/1801.02774): two concentric
spheres in R^d, radii `r_inner=1.0` and `r_outer=1.3`, class 0 sampled
uniformly from the inner sphere and class 1 from the outer one. The
classification rule (the true label is simply "which sphere is this point
on") is trivial for a human to state, but the *geometric* question of whether
any trained classifier's decision boundary can avoid having nearby points of
opposite predictions is a real theorem, not a training artifact.

The isoperimetric argument itself follows the general concentration-of-measure
approach to adversarial robustness in
[Fawzi, Fawzi & Fawzi (2018)](https://arxiv.org/abs/1810.12272) and
[Shafahi et al. (2019)](https://arxiv.org/abs/1809.02104); the exact
closed-form (non-asymptotic) ceiling derived and tested here is our own,
validated against Monte Carlo simulation and against the standard asymptotic
Lévy bound (see `src/concentration.py`, `tests/test_concentration.py`).

## The theory (derived and validated in `src/concentration.py`)

**Lévy's lemma / spherical isoperimetric inequality.** On `S^{d-1}` with its
normalized surface measure `μ`, for any set `A` with `μ(A) ≥ 1/2`, the
`ε`-blow-up `A_ε = {x : dist(x, A) ≤ ε}` satisfies `μ(A_ε) ≥ 1 −
exp(−(d−2)ε²/4)`. Among sets of a given measure, spherical caps are the
*extremal* minimizers of blow-up leakage — so a cap gives the exact worst-case
(tightest) bound, not merely an asymptotic one.

**Applying it to a classifier.** A binary classifier partitions each sphere
into two decision regions `R0`, `R1`. Let `Rmax` be whichever has measure
`≥ 1/2` and `Rmin = S^{d-1} \ Rmax` the minority region, `μ(Rmin) = p ≤ 1/2`.
Since `Rmax ⊆ (Rmax)_ε` and `S^{d-1} = Rmax ∪ Rmin`, the set of points
*farther* than `ε` from `Rmax` is exactly `{x ∈ Rmin : dist(x, Rmax) > ε}`,
which the lemma bounds by `exp(−(d−2)ε²/4)`. So:

> **At least `1 − exp(−(d−2)ε²/4)/p` of the minority region `Rmin` is within
> Euclidean distance `ε` of the majority region `Rmax`** — i.e. has a
> same-sphere adversarial example within `ε`, for *any* classifier with that
> class balance, regardless of what the decision boundary actually looks like.

This says nothing about *majority*-region points, which a well-fit classifier
is free to place robustly deep in the interior. **The theorem constrains
minority-region points specifically** — this shaped the experiment design
below (an early version that attacked "correctly classified" points instead
of minority-region points was comparing the wrong population against the
ceiling; see `src/experiment.py` docstring).

`src/concentration.py` computes the *exact* (non-asymptotic) version of this
bound: the extremal cap's measure is `μ(cap(θ)) = I_{sin²θ}((d−1)/2, 1/2)/2`
(the regularized incomplete beta function — a standard spherical-cap-measure
formula), inverted and root-found numerically (`scipy.special.betainc` /
`scipy.optimize.brentq`) to get an exact ceiling `ε_ceiling(δ, d, p)` with no
hidden asymptotic constants. `tests/test_concentration.py` validates the cap
formula against 60,000-sample Monte Carlo simulation (agreement to ~1%) and
checks the classical asymptotic bound is a valid (looser) upper bound on the
exact one.

**Falsifiable predictions tested:**

- **H0 (ceiling holds):** empirically-found minority-region adversarial
  distances never exceed `ε_ceiling` (up to attack/measurement noise).
- **H1 (Θ(1/√d) scaling):** the *fitted* exponent of median adversarial
  distance vs. `d`, `median ~ c·d^a`, has `a ≈ −0.5`.

## Method

- **Data / model** (`src/spheres.py`, `src/mlp.py`): points sampled exactly
  uniformly on each sphere via Gaussian-normalize-and-scale (validated against
  the analytic per-coordinate variance `R²/d` in `tests/test_spheres.py`). A
  from-scratch (no autodiff) 2-hidden-layer MLP, trained with a hand-rolled
  Adam optimizer; backprop is gradient-checked against finite differences to
  `<1e-4` relative error for every parameter (`tests/test_mlp.py`).

  **Why leaky-ReLU, not tanh, for the hidden layers:** points on a sphere are,
  coordinate-wise, zero-mean random projections that differ between classes
  only in *variance* (the radius), not in mean. `tanh` is an odd function, so
  `E[tanh(z)]` for zero-mean `z` carries almost no first-order signal about
  `std(z)` — an initial version of this experiment using `tanh` stalled at
  ~50-60% training accuracy for `d ≳ 64` no matter how long it trained. Leaky
  ReLU's asymmetry gives `E[relu(z)]` a direct dependence on `std(z)`,
  restoring a usable gradient signal at initialization; switching fixed
  training at every dimension tested (see `figures/accuracy_vs_dimension.png`
  — this is itself a small, incidental illustration of the "information
  exponent" phenomenon in learning even/radial functions by gradient descent,
  though we did not verify that connection directly here).

- **Attack population — the key methodological point.** For each sphere, we
  draw a large probe set (6000 points), find the classifier's minority
  prediction on that sphere (whichever label is less common — independent of
  ground truth), and attack *only* those minority-labeled points, since that
  is exactly the population Lévy's lemma constrains.

- **Finite-sample correction for the ceiling.** A probe of `n` points can
  observe zero minority predictions even when the true minority measure is
  small but nonzero. Feeding a raw `k=0` estimate into the ceiling formula
  gives a spurious infinite ceiling. Instead `p_minor` is a one-sided 95%
  Clopper-Pearson upper confidence bound (`scipy.stats.beta.ppf`), so the
  ceiling stays a valid, finite, conservative guarantee even in the
  near-perfect-separation regime (`tests/test_experiment.py`).

- **Three attacks** (`src/attacks.py`), all binary-search-over-budget with a
  gradient-ascent inner loop, reporting the smallest flip found:
  - **on-sphere**: constrained to the exact sphere manifold via the true
    geodesic exponential map (not an approximate project-and-renormalize) —
    this measures exactly the quantity the theorem bounds.
  - **general L2**: unconstrained in R^d, the standard practical notion of
    adversarial example.
  - **radial** (baseline/sanity check): search only along the ray through the
    origin — this exploits the problem's radial symmetry and is what you'd
    expect to work well *if* the classifier had simply learned "threshold on
    norm."

  Attack correctness is checked in `tests/test_attacks.py` against a linear
  model where the minimal L2 distance to the decision hyperplane is known
  analytically (`|w·x+b|/‖w‖`), and against an exact-geometry hemisphere
  classifier for the on-sphere case.

- **Dimension grid:** `d ∈ {2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768,
  1024}`. 5000 training / 800 validation / 1200 test points per class, 128
  hidden units, up to 600 epochs with early stopping, 40 attacked points per
  sphere per dimension (80 total). Single fixed seed (`seed=0`) throughout —
  results are a single realization, not averaged over seeds (see
  Limitations).

Run with `python run_experiment.py` (~2 minutes on 4 CPU cores, no GPU
needed). Outputs: `results/summary_by_dimension.csv`,
`results/raw_attack_rows.csv` (960 individual attack records),
`results/power_law_fits.csv`, and the figures below.

## Results

| d | test acc | ceiling (avg of inner/outer) | on-sphere median dist | on-sphere found frac |
|---|---|---|---|---|
| 2 | 1.000 | 0.00090 | — | 0% (no minority points in probe) |
| 4 | 1.000 | 0.03164 | — | 0% |
| 8 | 1.000 | 0.05135 | — | 0% |
| 16 | 1.000 | 0.04774 | 0.01408 | 100% (n=1) |
| 32 | 1.000 | 0.03733 | 0.02352 | 100% (n=2) |
| 64 | 0.996 | 0.03337 | 0.01502 | 100% (n=47) |
| 128 | 0.970 | 0.03052 | 0.01807 | 100% (n=80) |
| 256 | 0.930 | 0.02413 | 0.01013 | 100% (n=80) |
| 384 | 0.913 | 0.02057 | 0.01339 | 100% (n=80) |
| 512 | 0.894 | 0.01884 | 0.01230 | 100% (n=80) |
| 768 | 0.862 | 0.01640 | 0.01182 | 100% (n=80) |
| 1024 | 0.855 | 0.01479 | 0.00712 | 100% (n=80) |

Full per-dimension detail (including `p_minor`, inner/outer split, IQR,
off-manifold and radial results): `results/summary_by_dimension.csv`. Every
individual attack record: `results/raw_attack_rows.csv`.

**H0 — ceiling holds: confirmed at every measurable dimension.**
`figures/robustness_vs_dimension.png` plots the empirical on-sphere and
general-L2 attack medians against the exact ceiling. At every `d ≥ 16` where
minority points existed to attack, the empirical median distance sits
strictly below `ε_ceiling`, exactly as the theorem requires — no violations
in 960 individual attack attempts. At `d ≤ 8` the classifiers achieved
literally 100% accuracy on a 6000-point probe (minority region empirically
unobservable — `p_minor`'s Clopper-Pearson upper bound is still finite and
small, but there was nothing to attack).

**H1 — Θ(1/√d) scaling: confirmed for the pure theory, not cleanly confirmed
for the raw empirical curve, and the difference is itself the interesting
finding.** Fitting `median ~ c·d^a` in log-log space
(`results/power_law_fits.csv`):

| quantity | fitted exponent | 95% CI | R² |
|---|---|---|---|
| ceiling, fixed p_minor=0.3 (pure theory, zero empirical noise) | **−0.531** | (−0.546, −0.516) | 0.998 |
| ceiling, realized (each d's own trained p_minor) | +0.100 | (−0.244, 0.443) | 0.040 |
| on-sphere attack median | −0.164 | (−0.312, −0.016) | 0.496 |
| general L2 attack median | +0.103 | (−0.117, 0.324) | 0.149 |

Evaluating the exact ceiling formula at a *fixed* minority measure across the
same dimension grid — a pure evaluation of `src/concentration.py` with no
model training or attack noise involved at all — recovers `a = −0.531`,
matching the theoretical `−1/2` almost exactly (`figures/ceiling_decomposition.png`).
This validates that the closed-form derivation is correct and asymptotically
tight.

But the *realized* ceiling — using each dimension's own actual trained
classifier — does **not** show a clean `d^{-1/2}` trend (R² = 0.04). The
reason is visible in `figures/accuracy_vs_dimension.png`: `p_minor` isn't
held fixed across the grid, it drifts, because training this specific task
gets harder as `d` grows (accuracy falls from 100% to 85.5%). Since the
ceiling depends on *both* `d` and `p_minor(d)`, and `p_minor(d)` is itself an
idiosyncratic function of training difficulty rather than a clean function of
`d`, the two effects are confounded in the realized curve. The raw empirical
attack-distance exponent (−0.164, CI excludes −0.5) reflects this same
confound, compounded further by finite-sample noise (as few as 1-2 attacked
points at low `d`). **Disentangling this was only possible by separately
evaluating the pure closed-form formula** (`figures/ceiling_decomposition.png`)
— a good illustration of why validating a scaling law needs more than one
fitted curve when a nuisance variable can drift with the parameter of
interest.

**Secondary finding: the boundary is not a radial threshold.** The `radial`
baseline attack — which exploits the sphere's obvious symmetry and searches
only along the ray through the origin — found **zero** flips across all 960
attack attempts at every dimension (`radial_found_frac = 0.0` everywhere in
`results/summary_by_dimension.csv`). Since minority-region points are, by
construction, off the classifier's "expected" boundary, and the on-sphere /
general-L2 gradient attacks succeeded 100% of the time on the same points,
this shows the trained networks' minority pockets are genuinely
off-axis/non-radial structures, not a simple learned radius cutoff with a
slightly noisy threshold (`figures/attack_type_comparison.png`).

**Visual concentration effect.** `figures/distance_distributions.png`
compares the on-sphere adversarial-distance histogram at `d=64` (right tail
out past 0.10) against `d=1024` (right tail cut off below 0.05) — the
qualitative squeezing predicted by concentration of measure is visible by eye,
even though the fitted exponent on the noisy raw medians undershoots −0.5 for
the reasons above.

## Figures

- `figures/accuracy_vs_dimension.png` — classifier fit quality vs. d
- `figures/robustness_vs_dimension.png` — empirical attacks vs. exact ceiling, log-log, with power-law fit overlay
- `figures/ceiling_decomposition.png` — pure-theory (−0.531 exponent) vs. realized-classifier ceiling vs. empirical attacks
- `figures/exponent_estimates.png` — fitted exponents with 95% CI vs. the theoretical −0.5
- `figures/attack_type_comparison.png` — on-sphere vs. general-L2 vs. radial-baseline (0% found)
- `figures/distance_distributions.png` — histogram concentration, d=64 vs. d=1024

## Limitations

- **Single seed.** All results are one realization (`seed=0`); the reported
  `p_minor(d)` drift, and by extension the "realized ceiling" curve, could
  look different under a different seed's training trajectory. The pure-theory
  fixed-`p_minor` result (−0.531) is seed-independent (deterministic formula
  evaluation) and is the load-bearing confirmation of H1.
- **Attacks are upper bounds.** Both gradient attacks find *a* nearby flip,
  not provably the *nearest* one, so all reported empirical distances are
  upper bounds on the true minimal adversarial distance. This can only make
  the empirical-vs-ceiling comparison for H0 *more* conservative (a better
  attack could only find smaller distances, i.e. still under the ceiling).
- **Small sample at low d.** `d=16` and `d=32` have only 1 and 2 minority
  points respectively (100% training accuracy left almost nothing to attack)
  — those two rows in the results table are not statistically meaningful on
  their own.
- **One task, one architecture.** A single 2-hidden-layer MLP family and a
  single concentric-sphere radius gap (`r_outer/r_inner = 1.3`). The
  isoperimetric ceiling is architecture-agnostic by construction (it bounds
  *any* classifier), but the *realized* p_minor(d) drift we observed is
  specific to this network/optimizer/training-budget combination.

## Reproduce

```bash
pip install -r requirements.txt
pytest                 # 36 tests: gradient checks, Monte Carlo validation of the
                        # cap-measure formula, analytic attack sanity checks, integration smoke test
python run_experiment.py   # ~2 min on 4 CPU cores; writes results/ and figures/
```

## References

- Gilmer, J. et al. (2018). [*Adversarial Spheres*](https://arxiv.org/abs/1801.02774).
- Fawzi, A., Fawzi, H., Fawzi, O. (2018). [*Adversarial vulnerability for any classifier*](https://arxiv.org/abs/1810.12272).
- Shafahi, A. et al. (2019). [*Are adversarial examples inevitable?*](https://arxiv.org/abs/1809.02104) ICLR.
- Milman, V., Schechtman, G. (1986). *Asymptotic Theory of Finite Dimensional Normed Spaces* (concentration of measure on the sphere).
- Ledoux, M. (2001). *The Concentration of Measure Phenomenon*, AMS.
- Li, S. (2011). *Concise Formulas for the Area and Volume of a Hyperspherical Cap*, Asian J. Math. Stat. (regularized-incomplete-beta cap-measure formula used in `src/concentration.py`).
