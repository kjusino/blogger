# Do randomized sketches hit the Johnson–Lindenstrauss subspace-embedding bound — and does coherence break naive row sampling?

An autonomous, from-scratch simulation study of **randomized numerical linear
algebra**: the family of results (Sarlos 2006; Ailon & Chazelle 2006; Clarkson &
Woodruff 2013) showing that a `d`-dimensional subspace of `R^n` can be embedded
into `R^k` with `k = O(d/eps^2) << n` while preserving all pairwise distances to
within `(1 ± eps)`, using random linear maps that are cheap to apply. This is the
theoretical backbone of fast approximate least-squares solvers, streaming PCA,
randomized SVD, and nearest-neighbor search — an area none of the prior
`research-projects/` studies in this repo touch (they cover random matrix theory,
2D Ising finite-size scaling, quantum error correction thresholds, LDPC coding,
online-algorithm competitive ratios, algorithmic game theory, and cache-oblivious
algorithms, but not randomized/sketching linear algebra).

## Research question

Three sketch families are compared — **Gaussian** (dense, the textbook JL sketch),
**SRHT** (Subsampled Randomized Hadamard Transform: a random sign flip + Hadamard
mix, then uniform subsampling), and **CountSketch** (hash each row into one of `k`
buckets with a random sign, oblivious and `O(nnz(A))`):

1. **Do all three actually achieve the predicted `(1 ± eps)` subspace-embedding
   guarantee at `k = Theta(d/eps^2)`, with the predicted `eps(k) ~ k^-0.5` scaling?**
2. **SRHT is a *subsampling* scheme — it only ever touches `k` of the `n` rows.**
   Does the random-sign + Hadamard "mixing" step before subsampling actually
   matter, or is it decorative? Concretely: does plain uniform row sampling
   (skip the mixing) fail on a subspace whose leverage is concentrated on a few
   rows ("coherent"), while succeeding on a subspace with uniformly spread
   leverage ("incoherent")?
3. **Does the subspace-embedding guarantee translate into a useful downstream
   guarantee** — does sketch-and-solve least squares stay within the predicted
   relative error of the exact solution?
4. **Do the sketches' construction times actually reflect their asymptotic
   complexity** — `Theta(n*d*k)` for Gaussian vs. `Theta(n*d*log n)` for SRHT
   (independent of `k`) vs. `Theta(nnz(A))` for CountSketch (independent of both
   `d` and `k`)?

All four are falsifiable and tested via a from-scratch simulation (no
`scikit-learn`/`sklearn.random_projection` — only `numpy`/`scipy` primitives, with
a hand-rolled vectorized Fast Walsh–Hadamard Transform for the SRHT).

## Background

For a matrix `A` with `n` rows and an orthonormal basis `Q` (`n x d`, `Q^T Q = I`)
for `col(A)`, a sketch `S^T` (`k x n`) is an **`eps`-subspace embedding** of `col(A)`
iff `(1-eps)||x||^2 <= ||S^T Q x||^2 <= (1+eps)||x||^2` for *every* `x in R^d`. Since
`Q` has orthonormal columns this is *exactly* equivalent to every singular value
`sigma` of `S^T Q` satisfying `sigma^2 in [1-eps, 1+eps]` — so the worst-case
distortion over the entire subspace is computed exactly via one SVD:

```
eps = max(1 - sigma_min(S^T Q)^2, sigma_max(S^T Q)^2 - 1)
```

(`src/theory.py::subspace_distortion`). This is the same reduction used throughout
the sketching literature and avoids the statistical noise of sampling random `x`.
The classical sample-complexity bound is `k = C * (d + log(1/delta)) / eps^2` for
an unoptimized constant `C` — the experiments below treat the **scaling** (linear
in `d`, quadratic in `1/eps`) as the falsifiable claim, and empirically calibrate
`C` rather than assuming a specific textbook constant.

**Leverage scores.** Row `i`'s leverage score is `||Q_i||^2` (row norms of the
orthonormal basis); they sum to `d`. A subspace is *incoherent* if leverage is
spread ~uniformly (`~d/n` per row) and *coherent* if it is concentrated on a few
rows. `SRHT`'s speed comes entirely from touching only `k` rows — if leverage is
concentrated on rows the subsampling step doesn't happen to pick, the sketch
misses the subspace's structure entirely. The random sign flip + Hadamard
transform is designed to spread any input subspace's leverage near-uniformly
*before* subsampling, at the cost of an `O(n*d*log n)` pass.

## Methodology

`src/sketches.py` implements all three sketches from primitives (including a
fully-vectorized, `O(n*d*log n)`, `log2(n)`-pass Fast Walsh–Hadamard Transform —
correctness checked against `scipy.linalg.hadamard` for the exact matrix on small
`n`). `src/subspace.py` builds two test subspaces: `incoherent_basis` (QR of a
Gaussian matrix — uniform leverage) and `coherent_basis` (span of the first `d`
standard basis vectors — leverage 1 on `d` rows, 0 elsewhere: the maximally
coherent case). `src/experiment.py` runs five sweeps, each repeated over many
random trials (25 for the production run):

| Sweep | Tests | Figure |
|---|---|---|
| **Threshold** | success rate (`eps <= eps_target`) vs. `k / k0` where `k0` is the predicted sample complexity, incoherent subspace, all 3 sketches | `fig1` |
| **Scaling law** | median distortion vs. `k` on a log-log grid, fit `eps ~ a*k^b`, check `b ~ -0.5` | `fig2` |
| **Coherence ablation** | SRHT with vs. without the sign+Hadamard preconditioning step, on incoherent vs. coherent subspaces (2x2) | `fig3` |
| **Downstream least squares** | sketch-and-solve relative excess residual vs. `k / k0` (on `d+1`, since `b` generally isn't in `col(A)`) | `fig4` |
| **Timing vs. k** | construction wall-clock vs. `k`, fixed `n=8192, d=50` | `fig5` |
| **Timing vs. n** | construction wall-clock vs. `n` (log-log), fit exponents | `fig6` |

Production config: `d=30`, `n=4096`, `eps_target=0.25`, 25 trials/point. A `--smoke`
config (`d=8, n=1024`, 3 trials/point) runs the identical pipeline in seconds for
CI-style sanity checking.

## Results

**Threshold & scaling (P1, P2).** All three sketches show a genuine threshold
transition in success rate at `k ~ 1.5-2x` the predicted `k0` (`k0 = 1034` for
`d=30, eps=0.25`): success rate is 0 below `k0`, then rises sharply to 1.0 by
`k = 1.5-2 * k0` (`fig1`). The scaling-law fits are essentially exact:
`eps(k) ~ k^-0.56` (Gaussian, R²=1.00), `k^-0.64` (SRHT, R²=1.00), `k^-0.56`
(CountSketch, R²=1.00) — all close to the theoretical `-0.5` exponent (`fig2`).
The predicted *scaling* holds cleanly; the predicted *constant* is loose (as
flagged in `theory.py`) — the empirical crossover constant is closer to 1.5-2x
our guessed `C=2`, i.e. `C_true ~ 3-4`, consistent with textbook bounds being
worst-case and non-tight.

**Coherence ablation (P3) — the sharpest finding.** On the *incoherent* subspace,
plain uniform row sampling and full SRHT preconditioning are statistically
indistinguishable at every `k` tested (e.g. median `eps = 0.175` vs. `0.176` at
`k=1920`) — unsurprising, since there's no concentrated leverage to miss. On the
*coherent* (spiky) subspace, SRHT preconditioning tracks the incoherent curve
closely (`eps = 0.209` at `k=1920`), but **plain uniform sampling is catastrophic
at every scale tested**: median `eps = 33.1` at `k=120`, `7.5` at `k=480`, and
still `1.13` at `k=1920` — nearly double `k0` and still far outside any usable
embedding tolerance (`fig3`). The mechanism is exactly what leverage-score theory
predicts: uniform sampling of `k=1920` rows out of `n=4096` frequently misses
enough of the `d=30` "spike" rows entirely, and missing a spike row means losing
that entire slice of the subspace's energy — no amount of *more* uniform samples
below `n` fixes this reliably, whereas the random sign + Hadamard mixing step
provably spreads that energy across all rows first. **The preconditioning step is
load-bearing, not decorative** — it's the entire reason SRHT is allowed to
subsample instead of reading every row.

**Downstream least squares (P4).** Sketch-and-solve relative excess residual
decays smoothly from ~5% (`k = 0.3*k0`) to under 1% (`k = 2-3*k0`) for all three
sketches, staying comfortably inside the `eps_target = 0.25` bound at every tested
`k >= 0.3*k0` (`fig4`) — the subspace-embedding guarantee does translate directly
into a useful accuracy guarantee for the actual downstream problem, not just an
abstract distance-preservation statement.

**Timing (P5).** At fixed `n=8192, d=50`, Gaussian construction time scales
linearly with `k` (6.6ms at `k=50` to 502ms at `k=3200` — a 76x increase for a 64x
increase in `k`), while **SRHT (~14ms) and CountSketch (~7ms) are flat across the
entire `k` range**, exactly as `Theta(n*d*log n)` and `Theta(nnz(A))` predict
(`fig5`). At `k=3200`, Gaussian is **36x slower than SRHT and 68x slower than
CountSketch** for an *equally valid* embedding. Varying `n` instead (fixed
`d=50, k=300`) gives fitted exponents `1.02` (Gaussian), `1.21` (SRHT — the extra
`log n` factor bends the log-log line slightly upward), `0.94` (CountSketch) —
matching `Theta(n)`, `Theta(n log n)`, `Theta(n)` respectively (`fig6`).

## Limitations

- The sample-complexity constant `C` in `predicted_k` is a rough textbook
  placeholder, not fit to this data; only the *scaling* in `d` and `1/eps` is
  treated as the falsifiable claim (see Results above).
- `coherent_basis` is the single most adversarial case (leverage concentrated on
  exactly `d` rows, each with the maximum possible leverage score of 1) — real
  workloads typically have intermediate coherence; the ablation demonstrates the
  *mechanism*, not a calibrated real-world failure rate.
- Timing is single-threaded, single-machine, dense-`A` wall-clock via
  `time.perf_counter` (median of 5 repeats) — useful for relative comparison
  between sketches, not an absolute hardware benchmark. CountSketch's advantage
  would be far larger still on genuinely sparse `A`, which isn't tested here.
- All distortion values are the **exact worst-case** `eps` via one SVD per trial,
  not a Monte Carlo estimate over sampled `x` — this is more rigorous than
  sampling, but means results are specific to the tested `(n, d)` pairs rather
  than a continuous surface.

## Repo layout

```
src/sketches.py      Gaussian, SRHT (+ FWHT), CountSketch
src/subspace.py       incoherent/coherent test subspaces, least-squares system builder
src/theory.py         predicted_k, exact subspace_distortion via SVD, power-law fit
src/experiment.py      the 5 sweeps + FULL_CONFIG / SMOKE_CONFIG
src/plots.py           the 6 figures
run_experiment.py      CLI (`--smoke` for a fast sanity run)
tests/                 33 unit + integration tests
figures/                6 PNGs (committed, from the production run)
results/                raw_results.csv (126 rows) + summary.json (committed)
```

## Reproducing

```bash
pip install -r requirements.txt
pytest                       # 33/33, ~5s
python run_experiment.py --smoke   # ~2s sanity check
python run_experiment.py           # full production sweep, ~80s, seed=42 (deterministic
                                    # for every sweep except the timing wall-clock measurements)
```
