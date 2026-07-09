# Does curvature explain when greedy submodular maximization struggles?

A self-contained, from-scratch research project in **combinatorial
optimization / submodular set-function maximization**. Checked against the
39 other open research-project PRs in this repository (coding theory,
quantum error correction, random matrix theory, differential privacy,
compressed sensing, Boolean function analysis, network epidemiology,
algorithmic game theory for *online* routing/matching, statistical physics,
causal discovery, property testing, automata learning, and more) — none
touch submodular optimization, matroid-adjacent curvature theory, or the
classical Nemhauser–Wolsey–Fisher greedy guarantee, so this is new
territory for the series.

## Research question

For **monotone submodular maximization under a cardinality constraint**
(`max_{|S| <= k} f(S)`, NP-hard in general), the greedy algorithm — repeatedly
add the element with the largest marginal gain — has two classical
approximation guarantees:

- **Worst-case (Nemhauser, Wolsey & Fisher, 1978):** `f(S_greedy) >= (1 - 1/e) * OPT`
  for *any* monotone submodular `f`. This bound is tight in the worst case
  over *all* such functions, but says nothing about any particular instance.
- **Curvature-refined (Conforti & Cornuéjols, 1984; see also Vondrák, 2010):**
  `f(S_greedy) >= ((1 - e^{-c}) / c) * OPT`, where the *total curvature*
  `c = 1 - min_j [ f(j | V\{j}) / f(j | {}) ] ∈ [0, 1]` measures how much an
  element's marginal value can degrade once everything else is already
  chosen. `c = 0` means `f` is exactly modular (additive) and greedy is
  provably *exact*; `c = 1` recovers the classical `1 - 1/e` bound as a
  special case.

**H1 (stated up front, falsifiable):** the curvature-refined bound both (a)
is never violated — it's a theorem, and violating it numerically would mean
a bug — and (b) is a *more informative* characterization of greedy's
realized performance than the constant worst-case bound, particularly for
low-curvature instances where the constant bound is needlessly pessimistic.

## Methodology

**Function family.** A weighted-coverage instance (`f(S)` = total weight of
"features" covered by the union of `S` — the textbook max-coverage
submodular function) is built from:
- one exclusive **private feature** per element `i`, worth
  `w_i ~ Uniform(1, 10)` — its guaranteed private value;
- `n_groups = 10` **overlap groups**, each covering a random subset of
  elements (independently, with probability 0.35, at least 2 members
  forced) with feature weight `redundancy_mult * Uniform(1, 10)`.

An earlier draft used a *single* feature block shared identically by every
element. That was a real design bug caught during development: because
every element covered the exact same shared block, `f(S)` reduced to
`(sum of private weights in S) + constant`, so the optimal `k`-subset was
trivially "the `k` largest private weights" — and greedy, which also picks
by largest marginal gain, found it *exactly* every time, regardless of how
the theorem's curvature statistic was dialed. The curvature number moved,
but the optimization problem never got harder. Random, *partial* group
membership (some elements overlap with some others, not all with all) is
what actually creates non-trivial combinatorial tension between competing
subsets — confirmed by `redundancy_mult = 0` giving exactly `curvature = 0`
and exactly-optimal greedy (a hard-coded regression test), while
`redundancy_mult > 0` produces real, measurable greedy suboptimality.

**Sweep.** `redundancy_mult` ranges over 13 values from 0 to 30 (log-ish
spaced), crossed with `n ∈ {10, 14, 18}` (kept small enough for **exact**
brute-force optimum via full enumeration — up to `C(18,11) ≈ 48,620`
subsets, ~0.5s), 6 random seeds per configuration, and 3 cardinality budgets
`k ∈ {0.25n, 0.4n, 0.6n}` — 702 total (instance, k) trials over 234 distinct
instances. Curvature is *measured* per instance from its exact definition,
not assumed from the design knob — the two are related but not in closed
form for this instance family (see the "what we found" section below for
why that distinction turned out to matter).

## Success metrics (defined before running the full sweep)

| # | Metric | Result | Pass? |
|---|--------|--------|-------|
| M1 | Validity: curvature bound never violated (proven theorem — any violation is a bug) | 0/702 violations | ✅ |
| M2 | Informativeness: Spearman ρ(curvature, 1−ratio) > 0, p < 0.01 | ρ = 0.448, p = 6.1×10⁻³⁶ | ✅ |
| M3 | Curvature bound tighter than constant bound at characterizing each instance's worst-observed ratio (lower MAE) | curvature MAE 0.270 vs. constant MAE 0.358 | ✅ |
| M4 | Near-modular instances (c < 0.05) solved almost exactly (mean ratio ≥ 0.999) | mean ratio = 1.000 | ✅ |

**All four metrics pass** (`results/summary.json`, `overall_pass: true`).

## What we found

1. **The theorem holds, exactly, everywhere.** Across all 702 trials, the
   realized ratio never dipped below `(1 - e^{-c})/c`
   (`figures/validity_histogram.png`) — the expected result for a correctly
   implemented proof, and a real bug-catcher (an earlier version of the
   curvature formula had a sign error that this exact check caught during
   development).

2. **Both bounds are wildly loose for "generic" random instances — but the
   curvature bound is the less-loose one.** Even at curvature ≈ 1, where
   the worst-case theorem only promises 63.2%, greedy still realized a mean
   ratio of **99.1%** and a minimum ratio of **86.2%** across our full
   grid — far above either bound (`figures/ratio_vs_curvature.png`). This
   matches a well-known theme in empirical approximation-algorithms
   research (see also this repo's price-of-anarchy project, where only a
   *specially constructed* worst-case network came close to its bound):
   worst-case guarantees are rarely tight for random instances. Still, the
   curvature bound is measurably the better predictor of each instance's
   worst-observed ratio (M3: MAE 0.27 vs. 0.36) — informative even when far
   from tight.

3. **The hardest instances are at *intermediate* redundancy, not the
   theoretical extreme.** `figures/ratio_vs_redundancy_mult.png` shows mean
   ratio dipping to its *lowest* point (0.982) around `redundancy_mult ≈
   1.2` (curvature ≈ 0.94) — then *partially recovering* toward ~0.99–1.00
   as `redundancy_mult` climbs further toward 30 (curvature → 0.997), even
   though curvature itself keeps rising monotonically the whole time. A
   plausible mechanism: at extreme redundancy, the overlap groups so
   thoroughly dominate every element's value that "cover as many distinct
   groups as possible" becomes the obviously-correct strategy, and greedy
   finds it just as easily as brute force does; the *genuinely* ambiguous
   trade-offs — pick the two elements with high private value that happen
   to collide on one shared group, or the three lower-private-value
   elements that partition cleanly? — arise at intermediate overlap
   intensity, not at either extreme. This is a real, if unplanned, finding:
   curvature (the worst-case-over-all-such-functions quantity) is not the
   same thing as "how hard *this particular randomly generated instance
   family* turns out to be in practice," even though it remains a valid
   (if loose) universal lower bound throughout.

## What's included

- `src/submodular.py` — `WeightedCoverageFunction` (value, marginal gain,
  exact curvature) and `make_grouped_redundancy_instance` (the tunable
  instance generator)
- `src/greedy.py` — the standard greedy algorithm
- `src/optimal.py` — exact brute-force optimum (small `n`)
- `src/theory.py` — the two closed-form bounds
- `src/experiment.py` — the sweep driver and the four success metrics
- `src/plots.py` — all figure generation
- `run_experiment.py` — CLI entry point (`--quick` for a ~1s sanity check,
  full run in ~50s)
- `tests/` — 44 unit + integration tests: submodularity/monotonicity
  invariants (checked on random instances), the `redundancy_mult=0 ⟹
  curvature=0 ⟹ greedy is exact` regression test that caught the original
  design bug, brute-force-never-beaten sanity checks, both theorem bounds
  checked never-violated on random instances, and end-to-end pipeline tests
  including a determinism check
- `figures/` — 6 plots (ratio vs. curvature, validity histogram, gap vs.
  curvature, MAE comparison, extremes boxplot, ratio vs. redundancy
  intensity)
- `results/raw_results.csv`, `results/per_instance_results.csv`,
  `results/summary.json` — full production-run data (702 rows)

## Limitations

- Brute-force optimality restricts `n` to ≤ 18; larger-`n` behavior is not
  directly tested (though nothing in the theory or the instance family
  depends on scale in a way that would suggest this changes).
- The instance family is one specific (if reasonably rich) parametrization
  of coverage functions; other submodular families (matroid rank functions,
  facility location, entropy-based functions) could show different
  curvature-vs-difficulty relationships.
- The intermediate-redundancy-is-hardest finding (point 3 above) is
  observational, not proven — it's a legitimate empirical pattern in this
  instance family, offered with a plausible mechanism, not a new theorem.

## Reproduce

```bash
cd personal-projects/submodular-greedy-curvature
pip install -r requirements.txt
python3 -m pytest -q                # 44 tests, ~1s
python3 run_experiment.py --quick   # fast smoke test, ~1s
python3 run_experiment.py           # full production sweep, ~50s
```
