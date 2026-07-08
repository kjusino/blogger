# Does cache-oblivious matrix multiplication really hit the Θ(n³/(B√M)) ideal-cache bound?

An autonomous, from-scratch simulation testing the central claim of
**cache-oblivious algorithms** — that a recursive divide-and-conquer
algorithm which never references the machine's cache parameters can match
a *cache-aware, hand-tuned* algorithm's memory-traffic optimality. This is
a foundational result at the intersection of algorithms and computer
architecture (Hong & Kung 1981; Frigo, Leiserson, Prokop & Ramachandran
1999, "Cache-Oblivious Algorithms," FOCS) — the theoretical backbone for
why libraries like FFTW and cache-oblivious B-trees work well across
wildly different memory hierarchies without per-machine tuning.

## Research question

For n×n matrix multiplication under the **ideal-cache model** (a
fully-associative cache of `M` words, `B`-word cache lines, LRU
replacement), three falsifiable predictions from the literature:

1. **Naive** matmul (any fixed loop order, no explicit blocking) gets
   automatic *spatial* reuse but no *temporal/capacity* reuse: it should
   incur **Θ(n³/B)** misses once the cache is large enough to retain one
   "row-block" working set (`M ≥ n·B`), collapsing sharply to **Θ(n³)**
   below that threshold — a step function, not a smooth curve.
2. A **cache-aware blocked** algorithm (explicit tiling, tile size
   `t = Θ(√M)`) and a **cache-oblivious recursive** algorithm (which
   never sees `B` or `M`, and just recursively halves whichever
   dimension is currently largest) should both hit **Θ(n³/(B√M))**
   misses — and, crucially, the oblivious algorithm should match the
   blocked algorithm's constant *without* being retuned per `(B, M)`.
3. This only holds under the **tall-cache assumption**, `M = Ω(B²)`.
   Violate it (`M < B²`) and cache-obliviousness should lose its
   advantage.

**Success metric:** for each prediction, fit `log(misses)` vs.
`log(varied parameter)` by OLS across a swept parameter (holding the
others fixed in the regime where the prediction applies) and check the
fitted slope lands within ±0.3 of the theoretical exponent, with r² high
enough that the fit is meaningful. For the two step-function/threshold
claims (1 and 3), the pass condition is a genuine before/after miss-count
gap that lands in the direction and rough magnitude theory predicts.

## What "ideal cache" simulation means here

There is no real hardware in this study. `src/cache_sim.py` implements an
`IdealCache`: a fully-associative, LRU cache of `M` words organized into
`L = M/B` cache lines of `B` consecutive words, in a flat word-addressed
space. `src/matmul.py` instruments every array read/write against a
shared cache instance for three algorithms:

- `multiply_naive` — textbook `i, j, k` loop order, row-major storage.
  The running sum is kept in a Python local (not counted as cache
  traffic — this matches the literature convention that registers are
  free), so only the three n×n arrays' accesses are instrumented.
- `multiply_blocked` — classic three-level-blocked matmul, tile size
  `t = ⌊√(M/3)⌋`, snapped down to the nearest exact divisor of `n` to
  avoid ragged-tile boundary noise (a real, but distinct, effect from the
  one this study is measuring).
- `multiply_oblivious` — the "split the largest of the three dimensions"
  recursive algorithm (the version taught in MIT 6.172), bottoming out
  in a small base case. It never reads `B` or `M`.

All three are numerically exact (verified against a reference
implementation, including on non-power-of-two and prime `n`); only their
cache-miss counts differ.

## Methodology: the regimes matter as much as the exponents

The first full run exposed something the naive Θ(n³) intuition misses.
Naive matmul with row-major storage gets *automatic spatial locality*: as
the middle loop (`j`) advances, consecutive columns of the second operand
share cache lines, so a full "row-block" of `B` columns is reused before
eviction — **provided the cache can hold the `n` cache lines that group
needs** (`M ≥ n·B`). Below that threshold, the reused set doesn't fit and
every access misses. This means naive's cost is genuinely `Θ(n³/B)`
*conditional on* `M ≥ n·B`, not flat `Θ(n³)` — and separately, testing
`Θ(n³/(B√M))` for the blocked/oblivious algorithms only makes sense once
`n` is large enough relative to `√M` that the *bandwidth-bound* term
dominates the *compulsory-miss* term (`n²/B`, from touching each element
once) — otherwise the whole problem nearly fits in cache and the curve
looks sub-cubic. Four sweeps, each engineered to stay in the regime the
claim under test actually requires:

| # | Sweep | Fixed | Varies | Tests |
|---|-------|-------|--------|-------|
| E1 | `scaling_n` | B=8, M=2048 | n ∈ {80..192} | slope 3 in n, all three algorithms (n stays ≫ √M=45 and n·B stays ≪ M) |
| E2 | `scaling_B` | n=64, M=4096 | B ∈ {2..32} | slope −1 in B, all three algorithms |
| E3 | `scaling_M` | n=64, B=8 | M ∈ {512..8192} | slope −0.5 in M, blocked/oblivious only (M stays ≫ B²=64) |
| E3b | `naive_capacity_cliff` | n=64, B=8 | M ∈ {128..1024}, dense around 512 | the Θ(n³)→Θ(n³/B) step at M = n·B, naive only |
| E4 | `tall_cache_boundary` | n=64, B=32 | M ∈ {128..8192}, crossing B²=1024 | tall-cache assumption necessity, all three |

Each `(n, B, M, algorithm)` cell is one simulation: fill A and B with
random values (seeded), reset the cache counters, run the algorithm,
record hits/misses. 109 cells total, ~70 seconds on a single core.

## Results (production run, `results/summary.json`, `results/raw_results.csv`)

**All eight power-law fits landed within tolerance:**

| Fit | Predicted | Fitted | r² |
|---|---|---|---|
| scaling_n / naive | 3.00 | 2.98 | 1.000 |
| scaling_n / blocked | 3.00 | 2.97 | 0.997 |
| scaling_n / oblivious | 3.00 | 2.95 | 0.994 |
| scaling_B / naive | −1.00 | −1.00 | 1.000 |
| scaling_B / blocked | −1.00 | −0.97 | 0.952 |
| scaling_B / oblivious | −1.00 | −0.91 | 0.951 |
| scaling_M / blocked | −0.50 | −0.47 | 0.951 |
| scaling_M / oblivious | −0.50 | −0.50 | 0.988 |

(figure: `figures/scaling_n.png`, `figures/scaling_B.png`,
`figures/scaling_M.png`, summarized in `figures/fit_summary.png`)

**Naive's capacity cliff** (`figures/naive_capacity_cliff.png`): at
`n=64, B=8` the threshold is `M = n·B = 512`. Just below it, misses are
flat at 299,008 (≈ `n³` with a small extra constant from the A/C
operands); just above it they drop to 33,792 — an **8.8× drop**, matching
the predicted `B = 8×` almost exactly, and the transition is a genuine
step, not a gradual bend.

**Tall-cache boundary** (`figures/tall_cache_boundary.png`): at
`n=64, B=32` (`B² = 1024`), blocked and oblivious both track naive
closely (or worse — blocked briefly *exceeds* naive) while
`M/B² < 1`, then drop sharply below naive once the tall-cache assumption
holds (`M/B² ≥ 1`). Cache-obliviousness is not free — it needs the same
structural assumption the theorem states.

**Does oblivious really match blocked without tuning?** At the largest
`n` in E1 (n=192, B=8, M=2048): naive uses **9.0×** more misses than
oblivious; blocked and oblivious differ by only **1.10×** — with blocked
*hand-tuned per (B, M)* via `t = ⌊√(M/3)⌋` and oblivious using none of
that information. Fitted leading constants (`misses / (n³/(B√M))`) at
that point: naive 1.01 (matches its own `n³/B` form), blocked 5.57,
oblivious 5.07 — oblivious is not just asymptotically competitive, it's
numerically ahead of the tuned baseline here.

## Limitations

- This is a software simulation of the *ideal-cache model*, not real
  hardware: no associativity limits, no TLB, no write-back/write-through
  distinction, no multi-level hierarchy, and LRU rather than a real
  processor's approximate-LRU or random replacement. The model's own
  theorem (FLPR99, Lemma 2) is what justifies substituting LRU for the
  "optimal offline" replacement the Θ-bounds are stated for (LRU is
  2-competitive with optimal for this cache model).
- `default_tile` snaps to an exact divisor of `n` specifically to remove
  ragged-tile boundary noise from the blocked-vs-oblivious comparison;
  that noise is real (visible in an earlier, unpruned run in the same
  branch history) but is a distinct phenomenon from the asymptotic bound
  under test here.
- Every sweep is a 1-D slice through a 3-D `(n, B, M)` parameter space,
  deliberately kept inside the regime each claim requires (documented in
  the Methodology table). Reading the code path for the regime
  boundaries (`n ≫ √M`, `M ≫ n·B`, `M ≫ B²`) matters for interpreting
  any *new* sweep — get it wrong and you reproduce the same "off by a
  regime" confound the first draft of this study hit (see git history:
  fixing that confound, not a bug in the simulator, is what turned a
  vague "slope ≈ 2.5, MISS" result into the clean table above).

## Repository layout

```
src/cache_sim.py      IdealCache (LRU, word-addressed), Array1D, Matrix
src/matmul.py         naive / blocked / oblivious multiply, default_tile
src/theory.py         power-law fitting (OLS on log-log data), predicted exponents
src/experiment.py     the five sweeps (E1, E2, E3, E3b, E4)
src/plots.py          all figure generation (matplotlib, Agg backend)
run_experiment.py     CLI entry point (production sweep or --smoke)
tests/                62 unit + integration tests
figures/               6 PNGs from the production run
results/              raw_results.csv (109 rows), summary.json
```

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                   # 62 tests, ~13s
python run_experiment.py --smoke          # tiny sizes, sanity check, ~1s
python run_experiment.py                  # full production sweep, ~70s
```
