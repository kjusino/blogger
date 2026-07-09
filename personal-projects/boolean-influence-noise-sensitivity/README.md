# Influence and Noise Sensitivity of Boolean Functions

A numerical study of two foundational results in the analysis of Boolean
functions — the **Kahn–Kalai–Linial (KKL) isoperimetric influence bound**
and the **Benjamini–Kalai–Schramm (BKS) noise-sensitivity theorem** — plus
an exploratory look at a family with no known closed-form theory (random
k-DNF formulas). This area (Boolean function analysis / Fourier analysis on
the hypercube) is a core tool in theoretical CS: hardness of approximation,
learning theory, social choice, and circuit complexity all lean on it, and
it's an active line at Khoury (e.g. Boolean/circuit complexity work in the
theory group).

## Research question

For a Boolean function `f: {-1,+1}^n -> {-1,+1}`, define:

- **Influence** `Inf_i(f) = Pr[f(x) != f(x with coordinate i flipped)]`,
  and **total influence** `I(f) = sum_i Inf_i(f)`.
- **Noise sensitivity** `NS_delta(f) = Pr[f(x) != f(y)]`, where `y` is a
  `delta`-noisy copy of `x` (each coordinate independently resampled
  uniformly with probability `delta`).

Three classical claims:

1. **KKL (1988):** every `f` has some coordinate with
   `Inf_i(f) = Omega(Var(f) * log(n) / n)`.
2. **Majority:** `I(Maj_n) = Theta(sqrt(n))`, with constant `sqrt(2/pi)`.
3. **BKS (1999) / Ben-Or–Linial Tribes:** the `Tribes(w,s)` function
   (`n = w*s`, OR of `s` ANDs of width `w`, tuned so `w ~ log2(s)`) has
   *every* coordinate influence `Theta(log n / n)` (matching KKL up to a
   constant — it's essentially tight) **and** noise sensitivity
   `NS_delta(Tribes_n) = Theta(delta)` *uniformly in n*, which is much
   smaller than a "typical" balanced function of the same influence profile.

**Question this project asks:** do these hold up numerically across a wide
range of `n`, and — since no such theorem exists for random k-DNF formulas —
how does that family's influence structure move between the
Parity/Majority and Tribes regimes as clause width `k` grows?

## Methodology

- **Exact computation where possible.** Majority's per-coordinate influence
  has a closed form (`Inf_i(Maj_n) = C(n-1,(n-1)/2) / 2^(n-1)`, computed in
  log-space via `lgamma` so it's numerically stable out to `n ~ 2*10^5`).
  Tribes' influence also has a closed form
  (`Inf_i = 2^-(w-1) * (1-2^-w)^(s-1)`), derived and unit-tested against
  brute-force truth-table enumeration for small `(w,s)`.
- **Monte Carlo where no closed form exists** (random k-DNF, and noise
  sensitivity for every family — noise sensitivity only ever needs function
  *evaluations*, never a truth table, so it scales to any `n`).
- **A subtlety the naive experiment design walked into, and how it was
  fixed.** The first version of this experiment tried to distinguish
  Majority (`NS_delta ~ sqrt(delta)`) from Tribes (`NS_delta ~ delta`) by
  fitting a power law to `NS_delta` vs. `delta` at small `delta`, fixed `n`.
  That fit converged to exponent ≈ 1 for *every* family (see
  `results/summary.json` → `noise_sensitivity_small_delta_exponents`).
  The reason: `Stab_rho(f) = sum_S rho^|S| f_hat(S)^2`, so for `rho = 1 -
  delta` and small `delta`, `Stab ~ 1 - delta*I(f)`, giving
  `NS_delta(f) ~ delta*I(f)/2` **for any Boolean function** as `delta -> 0`
  at fixed `n`. The `sqrt(delta)` vs. `delta` distinction is a statement
  about the limit `n -> infinity` taken *before* `delta -> 0` — it lives in
  a different regime than "small `delta`, fixed `n`." The fix
  (`noise_sensitivity_vs_n` in `run_experiment.py`) fixes `delta` and grows
  `n` instead, which is where the theorems actually separate. This is
  documented in the code and in `summary.json` as
  `noise_sensitivity_small_delta_caveat`, since it's exactly the kind of
  order-of-limits trap worth flagging rather than quietly re-running until
  the numbers "look right."
- **Universal check.** The KKL ratio
  `max_i Inf_i(f) * n / (Var(f) * log2(n))` is computed for every Majority
  and Tribes instance in the sweep (102 instances); KKL guarantees this
  stays positive (bounded away from 0) as `n` grows. This is a genuine
  falsifiable check — a single negative ratio anywhere would be either a bug
  or a refutation of a 35-year-old theorem.

## Success metrics

| Claim | Metric | Result |
|---|---|---|
| Majority `I(Maj_n) = Theta(sqrt(n))` | log-log regression exponent vs. theory's 0.5 | **0.497** (95% CI [0.4963, 0.4982]), R² = 0.99995 |
| Tribes max influence `= Theta(log n / n)` | stability of `Inf*n/log2(n)` vs. n (regression slope ≈ 0) | slope = **0.0008** (flat — ratio stabilizes as predicted) |
| KKL bound never violated | `min` isoperimetric ratio across 102 instances | **0.50**, always > 0 |
| Majority's noise sensitivity converges to the CLT/Sheppard limit `arccos(1-delta)/pi` as n grows (delta=0.2 fixed) | gap to limit shrinks with n | **True** — gap goes from 0.024 (n=11) to <0.002 (n≥201); limit = 0.2048 |
| Tribes' noise sensitivity stays below the Sheppard baseline uniformly in n | `NS_0.2(Tribes_n) < limit` for all tested n, with margin | **True** for every n tested (0.172 → 0.080, moving *further* below as n grows to ~13,000) |

All numeric outputs are in `results/*.csv` and `results/summary.json`;
figures are in `figures/`.

## Findings

1. **Majority's `Theta(sqrt(n))` law is essentially exact even at small n.**
   The fitted exponent (0.497) matches the theoretical 0.5 to three
   significant figures, and the fitted curve is visually indistinguishable
   from `sqrt(2n/pi)` across five decades of `n` (`figures/majority_scaling.png`).

2. **Tribes' influence tracks `Theta(log n / n)`, and the KKL ratio plot
   shows *why* Tribes is special.** `figures/kkl_bound_check.png` plots the
   isoperimetric ratio for both families: Tribes' ratio is flat (~0.5–0.7)
   across six orders of magnitude in `n` — it sits close to the KKL lower
   bound, which is exactly the point of the Ben-Or–Linial construction.
   Majority's ratio *grows* without bound (its influence is much larger
   than the minimum KKL guarantees, since every coordinate carries equal,
   relatively large `Theta(1/sqrt(n))` influence rather than being
   concentrated near the extremal bound).

3. **The order-of-limits trap (see Methodology) is itself a finding.** The
   small-delta exponent fit gives ≈1 for Majority, Tribes, Parity, *and*
   random DNF alike — it is not informative about which asymptotic regime a
   function is in. `figures/noise_sensitivity_vs_n.png` is the corrected
   experiment: at a *fixed* `delta = 0.2`, Majority's noise sensitivity
   converges upward to the Sheppard/CLT limit (0.205) by `n ≈ 200`, while
   Tribes' noise sensitivity, evaluated at matched `n` with `w` retuned to
   `~log2(n)` at each step, not only stays below that limit but keeps
   *decreasing* as `n` grows to 13,000 — direct numerical evidence for why
   BKS singles out Tribes as noise-stable in a way "typical" balanced
   functions (Majority) are not.

4. **Random k-DNF (exploratory arm, no known theorem):** total influence at
   fixed `n=45` grows monotonically with clause width `k`, from 1.0 at
   `k=1` (Parity-like — a width-1 "AND" is just a literal, so an OR of them
   behaves close to a low-influence threshold function) through the
   matched-`n` Tribes reference (I≈2.9, crossed around `k=4`) and trending
   toward the Majority reference (I≈5.4) by `k=6`
   (`figures/random_dnf_influence_trend.png`). This is consistent with the
   general intuition that wider clauses at balanced density push a random
   formula's influence profile away from the KKL-tight Tribes regime and
   toward the "typical balanced function" regime that Majority represents
   — but, unlike the other three families, there is no closed-form theorem
   to check this against; it's reported as an open empirical trend, not a
   verified law.

## Project structure

```
src/functions.py          Boolean function families (Parity, Majority, Tribes, Random k-DNF)
src/influence.py           Exact (brute force, n<=20) and Monte Carlo influence estimators
src/noise_sensitivity.py   Monte Carlo noise-sensitivity estimator + closed forms (Parity, Sheppard limit)
src/fitting.py              Power-law regression + KKL isoperimetric-ratio check
src/plotting.py             Figure generation
run_experiment.py           Orchestrates the full pipeline; `--quick` runs a small grid
tests/                       38 unit tests + 1 end-to-end integration test (pytest)
results/                     CSV + summary.json outputs from the full run
figures/                     PNG figures from the full run
```

## Reproducing

```bash
pip install -r requirements.txt
python -m pytest                 # 40 tests, ~6s
python run_experiment.py         # full run, ~1-2 minutes -> results/, figures/
python run_experiment.py --quick # small grid, ~1s, for fast iteration
```
