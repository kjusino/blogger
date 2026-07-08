# Does the PC Algorithm's Causal-Discovery Sample Complexity Match Theory?

**Status: results pending — this section will be filled in once `run_experiment.py` completes.**

## Research question

Causal discovery algorithms try to recover a directed acyclic graph (DAG) of
cause-effect relationships from purely observational data, using conditional
independence tests instead of interventions. The classical **PC algorithm**
(Spirtes, Glymour & Scheines) does this by starting from a complete graph and
deleting an edge `i-j` whenever it finds *some* set of other variables that
renders `i` and `j` conditionally independent.

Kalisch & Buhlmann (2007) proved that, for sparse Gaussian DAGs with `p`
variables and maximum node degree `d`, the PC algorithm recovers the exact
skeleton (the undirected edge structure) with probability approaching 1 once
the sample size satisfies

```
n = Theta(d^2 log p)
```

This project asks: **does that asymptotic prediction actually show up in
finite, simulatable sample sizes** — and can we recover both exponents (2 for
degree, 1 for log p) from data alone, the same way physicists check a
critical exponent against Onsager's exact solution?

This is a different regime from most causal-discovery benchmarks, which
report accuracy at one or two sample sizes on one graph. Here the outcome
variable is a **sample-complexity threshold** (`n50`, the sample size at
which exact-recovery probability crosses 50%), estimated separately for many
`(p, d)` configurations, then regressed against theory in log-log space.

## Why this needed more than "generate data, run PC, plot"

Two subtleties showed up during development that would otherwise have
silently produced a study of the wrong thing:

1. **Unfaithfulness.** A randomly-weighted linear-Gaussian SEM can, by pure
   chance, produce a genuine edge whose implied partial correlation is
   (near-)zero for some conditioning set the algorithm tests, due to
   cancelling paths. That's a *population-level* property with nothing to do
   with sample size — no amount of data fixes it. Left unchecked, it caps
   exact-recovery probability below 1 even as `n -> infinity`, which would
   look like a violation of the theory but is really a generator bug.
2. **The "strong faithfulness" / beta-min condition.** The `n = Theta(d^2 log
   p)` bound implicitly assumes the *weakest* true-edge signal in the graph
   is bounded away from zero by some constant. Densely-connected random SEMs
   routinely produce edges whose partial correlation is technically nonzero
   but astronomically small for *some* (non-separating) conditioning set,
   demanding impractically large `n` to detect and swamping the `p`/`d`
   scaling this experiment is trying to isolate.

Both are handled by `src/dag_generator.generate_faithful_dag`, which computes
the SEM's *exact* population covariance analytically (no sampling noise) and
rejection-samples edge weights until (a) an oracle run of the PC skeleton
search on that exact covariance recovers precisely the true skeleton, and
(b) every true edge's weakest signal, measured along the algorithm's own
test sequence, exceeds a fixed margin (`min_margin = 0.10`).

A third, more mundane issue also mattered: **conditioning-set-size
significance must shrink with `p`.** A fixed significance level `alpha` has
a constant per-test Type-I error rate that does *not* shrink as `n` grows;
with `O(p^2)` pairs tested, spurious edges survive at a rate that grows with
`p` unless `alpha` is scaled down. This project uses a Bonferroni correction
`alpha = 0.05 / C(p, 2)` across all pairs — exactly the mechanism that
produces the `log p` term in the Kalisch-Buhlmann proof, so this isn't a
tuning hack, it's an explicit implementation of an assumption the theorem
already requires.

## Method

1. **DAG + SEM generation** (`src/dag_generator.py`): random DAGs on `p`
   nodes in a fixed topological order, maximum skeleton degree capped at
   `d`, linear-Gaussian structural equations with edge weights drawn from
   `+/- Uniform(0.4, 1.2)`. `generate_faithful_dag` rejection-samples until
   the SEM is faithful with a guaranteed margin (see above).
2. **PC-stable skeleton search** (`src/pc_algorithm.py`): the order-independent
   variant of Colombo & Maathuis (2014) — neighbor sets are frozen at the
   start of each conditioning-set-size level, and edge deletions are applied
   only after every pair is tested at that level. Conditioning sets are
   searched from **both** endpoints of a pair (an easy-to-miss detail: the
   true separating set can live in `neighbors(i)` or `neighbors(j)`, and
   they differ in general — searching only one side silently leaves spurious
   edges in the skeleton no matter how large `n` gets). Followed by
   v-structure (collider) orientation and Meek's rule R1 for a CPDAG, though
   the sample-complexity analysis itself targets the **undirected skeleton**,
   matching what the theorem bounds.
3. **Independence test**: Fisher z-transform of the partial correlation,
   computed from the empirical covariance matrix by inverting the relevant
   sub-matrix.
4. **Recovery metric**: exact skeleton recovery (structural Hamming distance
   = 0 against the true skeleton), averaged over many independent trials
   (fresh random DAG + fresh samples each trial) to get a recovery
   *probability* at each `n`.
5. **n50 estimation**: for each `(p, d)`, a coarse geometric sweep locates
   the approximate 50%-recovery crossing, then a finer grid of samples
   around that point (more trials each) is used to estimate `n50` by linear
   interpolation on the empirical curve (robust) and by a logistic-curve fit
   (for the transition steepness).
6. **Scaling fits**: `log(n50)` regressed against `log(p)` (with `d` fixed
   at 2, `p` swept) and against `log(d)` (with `p` fixed at 14, `d` swept),
   via ordinary least squares. Theory predicts slopes of 1 and 2
   respectively.

### A practical detail: bounding worst-case cost

At very small `n`, tiny Bonferroni-corrected significance levels give
near-zero test power, so almost no edges get pruned and the skeleton search
can stall near the complete graph — a combinatorial blow-up (testing
`C(p-1, k)` conditioning sets per pair for growing `k`). This is capped by
`max_cond_set_cap(p, d) = min(p - 2, d + 3)`, mirroring the `m.max`
parameter of practical PC implementations (e.g. R's `pcalg`): generous slack
above the true degree, but bounded, so the search cost stays tractable even
before the graph has been pruned down to its sparse true structure.

## Success metrics

- **Primary**: the log-log regression slope of `n50` vs `p` should be close
  to 1 (theory: `n50 ~ log p`, so plotting against `log(p)` linearizes it —
  the regression here is `log(n50)` vs `log(p)`, i.e. it tests whether n50
  behaves as a *power* of p; see note below on how this is interpreted
  against the log p prediction).
- **Primary**: the log-log regression slope of `n50` vs `d` should be close
  to 2 (theory: `n50 ~ d^2`).
- **Secondary**: a "rescaled collapse" plot — recovery-probability curves
  for different `p` (or `d`) plotted against `n / log(p)` (or `n / d^2`)
  should approximately collapse onto a single curve if the theoretical
  scaling holds.

## Results

*(Filled in after `run_experiment.py` runs — see `results/*.csv` and
`figures/*.png`.)*

## Limitations

- Sample sizes are modest (`p` up to ~80, `d` up to 3) to keep the whole
  study autonomously runnable in minutes rather than hours; the asymptotic
  theorem is a limit statement, and finite-`n50` estimates from a handful of
  `(p, d)` points are a noisy proxy for the true exponent.
- The "strong faithfulness" margin is enforced at a single fixed threshold
  (0.10) rather than studied as its own axis — Uhler et al. (2013) showed
  the strong-faithfulness constant itself shrinks quickly with graph
  density, which is part of why higher-degree, larger-`p` configurations
  were harder to generate satisfying SEMs for.
- Only linear-Gaussian SEMs are studied; the PC algorithm's theory extends
  beyond this setting but the exact-covariance faithfulness oracle relies on
  it.

## Repository layout

```
src/
  dag_generator.py        random faithful linear-Gaussian SEM generation
  partial_correlation.py  partial correlation + Fisher z CI test
  pc_algorithm.py          PC-stable skeleton search + CPDAG orientation
  metrics.py                SHD, precision/recall, n50 estimation, log-log fits
  experiment.py              sweep orchestration (coarse -> fine n50 search)
  plotting.py                 figure generation
tests/                        unit + integration tests (pytest)
run_experiment.py               entry point; writes results/ and figures/
results/                          CSV outputs
figures/                          PNG plots
```

## Reproducing

```
pip install -r requirements.txt
pytest                 # unit + integration tests
python3 run_experiment.py   # full sweep, ~15-25 minutes
```

## References

- Spirtes, Glymour, Scheines. *Causation, Prediction, and Search* (2000).
- Kalisch, M. & Buhlmann, P. "Estimating High-Dimensional Directed Acyclic
  Graphs with the PC-Algorithm." *JMLR* 8 (2007): 613-636.
- Colombo, D. & Maathuis, M. H. "Order-Independent Constraint-Based Causal
  Structure Learning." *JMLR* 15 (2014): 3741-3782.
- Uhler, C., Raskutti, G., Buhlmann, P., & Yu, B. "Geometry of the
  faithfulness assumption in causal inference." *Annals of Statistics* 41.2
  (2013): 436-463.
