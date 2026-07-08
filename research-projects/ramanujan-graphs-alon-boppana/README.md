# Random Regular Graphs: Do They Really Become Asymptotically Ramanujan?

**Status:** complete, autonomously executed, reproducible end-to-end.

## Research question

Take a `d`-regular graph `G` on `n` vertices (every vertex has exactly `d`
neighbors). Its adjacency spectrum always has a top eigenvalue `lambda_1 = d`
(the all-ones vector). How large can the *rest* of the spectrum be forced to
be? Define

```
lambda(G) = max_{i >= 2} |lambda_i(G)|
```

the largest eigenvalue magnitude excluding the top one. Small `lambda(G)`
means `G` is a good spectral expander -- the property that makes expander
graphs useful throughout theoretical CS (derandomization, error-correcting
codes, sorting networks, PCPs). Two classical results bound how small
`lambda(G)` can possibly be, and both are asymptotic statements about
*sequences* of graphs, not concrete finite-`n` guarantees:

1. **Alon-Boppana bound** (Alon 1986; quantified by Nilli 1991 -- see
   Hoory, Linial & Wigderson's survey, Thm 5.2): for any sequence of
   `d`-regular graphs `G_n` with `n -> infinity`,
   `liminf lambda(G_n) >= 2*sqrt(d-1)`. A graph meeting this bound at finite
   `n` is called **Ramanujan**; `2*sqrt(d-1)` is the best possible
   asymptotic spectral gap any bounded-degree expander family can have.

2. **Friedman's theorem** (2003 announcement; 2008 Memoirs of the AMS
   proof of Alon's "second eigenvalue conjecture"): a *uniformly random*
   `d`-regular graph on `n` vertices is asymptotically almost Ramanujan --
   for every `eps > 0`, `P(lambda(G_n) <= 2*sqrt(d-1) + eps) -> 1` as
   `n -> infinity`.

Friedman's proof is one of the hardest and most celebrated results in
spectral graph theory (the Memoirs monograph runs to 100+ pages of trace
methods on random walks on graphs). This project doesn't attempt to
re-derive it -- it asks the purely computational question a first read of
the theorem statement raises:

> **If you actually sample random `d`-regular graphs and diagonalize them,
> does `lambda(G)` really approach `2*sqrt(d-1)`, from which side, at what
> rate, and how does that rate depend on `d`?**

This sits squarely in spectral graph theory / theoretical CS (expander
graphs), is fully self-contained (no external data, no GPU -- sparse linear
algebra and a graph generator), and is answerable to near machine precision
at the validation step and with clean statistics at the empirical step.

## Methodology

### 1. Two independent random-graph generators (`src/ramanujan_spectra/graphs.py`)

- **`pairing_model_regular_graph`**: a from-scratch implementation of the
  classical configuration/pairing model with full restart on failure
  (Bollobas 1980) -- pair up `d*n` stubs uniformly at random, and if that
  produces a self-loop or repeated edge, throw the whole pairing away and
  retry. Conditional on success this is *exactly* uniform over labeled
  simple `d`-regular graphs; no approximation, only rejection. The catch:
  the probability a single random pairing is already simple falls off fast
  in `d` (empirically ~14% at `d=3`, ~2% at `d=4`, well under 1% by `d=6`,
  and unusably small by `d=10`), so this generator is only practical for
  small/moderate `d`.
- **`networkx_regular_graph`**: networkx's `random_regular_graph`, which
  repairs a bad pairing via targeted edge swaps instead of discarding it,
  so it stays fast for every degree used here (including `d=10`). This is
  the generator used for the main sweep.
- **Cross-validation** (`experiment.compare_generators`,
  `results/generator_cross_validation.csv`): at `d in {3,4}`, `n in
  {100,500}`, 150 independent graphs from each generator are compared on
  mean `lambda(G)`. The two independent implementations agree to within
  **0.58 pooled standard errors** in the worst case across all four
  `(d, n)` combinations tested -- no evidence either generator is biased
  relative to the other.

### 2. Eigenvalue computation (`src/ramanujan_spectra/spectrum.py`)

`extremal_eigenvalues` computes `lambda_1`, the second-largest algebraic
eigenvalue, and the most negative eigenvalue (`lambda(G)` needs both ends
of the spectrum, since a bipartite `d`-regular graph always has
`lambda_min = -d` exactly, making `lambda(G) = d` regardless of the rest of
the spectrum). Dense `numpy.linalg.eigvalsh` is used below `n=300`;
`scipy.sparse.linalg.eigsh` (Lanczos, `which="LA"`/`"SA"`) above that,
with automatic `ncv`/`maxiter` escalation on non-convergence. Every result
asserts `lambda_1 ≈ d` as an internal consistency check.

**Validation against hand-known exact spectra**
(`results/exact_validation.png`, `.csv`): `K_{d+1}` (spectrum `{d, -1 x d}`),
`K_{d,d}` (spectrum `{d, -d, 0 x (2d-2)}`), and the Petersen graph
(strongly regular, spectrum `{3, 1 x5, -2 x4}`, a famous concrete example of
a Ramanujan graph: `lambda(G)=2 <= 2*sqrt(2)≈2.828`). Max absolute error
across all seven fixtures: **2.7e-15** -- machine precision.

### 3. The sweep (`src/ramanujan_spectra/experiment.py`)

`d in {3, 4, 6, 10}` x `n in {64, 128, 256, 512, 1024, 2048, 4096, 8192}`
(log-spaced), with 40 independent random graphs per cell for `n <= 512`,
20 for `n in {1024, 2048}`, and 10 for `n in {4096, 8192}` (the eigensolver
dominates cost at large `n`, so fewer, not more, trials there) -- **880
random regular graphs generated and diagonalized in total**. Every draw
uses a deterministic seed derived from `(d, n, trial)` via
`numpy.random.SeedSequence`, so the whole sweep is exactly reproducible.
For each graph: `lambda(G)`, connectivity, and a bipartite-like flag
(`lambda_min ≈ -d`) are recorded (`results/trials.csv`); per-cell
aggregates -- mean/std/min/max of `lambda(G)`, fraction connected, and two
epsilon-indexed fractions described below -- are in `results/summary.csv`
and `results/summary.json`.

## Results

**0. Sanity checks, clean across the board.** All 880 sampled graphs were
connected and none was bipartite-like -- expected for `d >= 3` regular
graphs at these sizes, but worth confirming rather than assuming, since a
disconnected or bipartite sample would silently corrupt the spectral
statistics (a disconnected regular graph gets a second copy of `lambda_1`,
and a bipartite graph gets `lambda(G)=d` regardless of everything else).

**1. Mean `lambda(G)` converges to `2*sqrt(d-1)`, for every degree tested**
(`results/lambda2_vs_n.png`). At `d=3`: from `2.780 ± 0.036` (`n=64`) to
`2.8276 ± 0.0010` (`n=8192`), vs. bound `2.8284`. At `d=10`: from
`5.512 ± 0.160` (`n=64`) to `5.9925 ± 0.0067` (`n=8192`), vs. bound `6.0`.
Same pattern at `d=4` and `d=6`. The standard deviation shrinks by roughly
20-30x from smallest to largest `n` at every degree -- the distribution is
concentrating, not just its mean drifting.

**2. Convergence is *from below*, and it's a clean power law**
(`results/gap_convergence_loglog.png`). This is the finding a first read of
Alon-Boppana doesn't obviously predict: the theorem states an asymptotic
*lower* bound (`liminf >= 2*sqrt(d-1)`), which is fully consistent with
individual finite graphs sitting below it, but it doesn't say that's what
typically happens. Empirically, it is what happens here -- **every one of
the 32 (d, n) cells has a negative mean gap** (`lambda(G)` below the bound
on average), and `|mean[lambda(G)] - 2*sqrt(d-1)|` decays as a strikingly
clean power law in `n` on a log-log plot:

| d  | fitted exponent (gap ∝ n^α) | R²    |
|----|------------------------------|-------|
| 3  | α = -0.90                    | 0.904 |
| 4  | α = -0.79                    | 0.984 |
| 6  | α = -0.80                    | 0.990 |
| 10 | α = -0.85                    | 0.996 |

Four exponents, independently fit from 8 points each, all land in a narrow
`[-0.90, -0.79]` band despite `d` ranging over more than a factor of 3 --
suggestive of a common (or at least similar) finite-size scaling law across
degrees, though with only 4 degrees tested this is an empirical observation
here, not a claim of a proven universal exponent.

**3. Individual graphs do occasionally exceed the bound -- but the
*excess* shrinks fast even though the *frequency* doesn't.** Across the
sweep, 99/880 graphs (11.3%) have `lambda(G)` strictly above `2*sqrt(d-1)`,
and this fraction doesn't visibly trend toward 0 as `n` grows (it's
noisily ~5-20% at every `n` tested, consistent with fluctuations around a
limiting distribution rather than a shrinking-probability tail). What does
shrink is the *size* of the excess: the single worst violation in the
entire sweep is `+0.169` (`d=6, n=64`), while every exceedance at `n in
{4096, 8192}` is under `0.0054` (full list in `results/trials.csv`, filter
`gap > 0`). This is exactly consistent with Friedman's theorem as literally
stated -- "for every fixed `eps > 0`, `P(lambda(G) <= bound + eps) -> 1`"
allows exactly this pattern (a persistent but shrinking-magnitude population
of near-bound exceedances) and does not require the exceedance *frequency*
itself to vanish.

**4. Two-sided concentration, quantified** (`results/near_ramanujan_fraction.png`):
tracking `P(|lambda(G) - 2*sqrt(d-1)| <= eps)` (both overshoot *and*
undershoot count against you) shows the qualitatively cleanest picture --
at every degree, this fraction rises from well below 1 at `n=64` to
pinned at 1.0 within one to two orders of magnitude of `n` growth, and it
takes *more* vertices to converge as `d` grows (at `eps=0.05`, `d=3`
reaches 1.0 by `n≈256`; `d=10` doesn't reach it until `n≈2048`) --
i.e. higher-degree graphs need to be larger before their spectral gap
"looks Ramanujan," even though all four degrees converge at a similar power
law in the log-log sense (finding 2).

## Repo layout

```
ramanujan-graphs-alon-boppana/
├── README.md
├── pyproject.toml / requirements.txt
├── src/ramanujan_spectra/
│   ├── theory.py            # Alon-Boppana bound, Ramanujan / near-Ramanujan / concentration checks
│   ├── graphs.py             # two independent d-regular graph generators + known exact-spectrum fixtures
│   ├── spectrum.py           # dense/sparse extremal-eigenvalue computation
│   ├── experiment.py         # the sweep, aggregation, power-law fits, generator cross-validation
│   ├── plots.py               # all matplotlib figures
│   └── run_experiment.py     # end-to-end driver
├── tests/                     # 48 unit + integration tests (pytest)
└── results/                   # CSVs, summary.json, all PNGs (committed)
```

## Reproducing

```bash
cd research-projects/ramanujan-graphs-alon-boppana
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/ -v                          # 48 tests, ~3s

python -m ramanujan_spectra.run_experiment  # full sweep, ~75s
# regenerates results/*.csv, results/summary.json, and all PNGs
```

## References

- Alon. "Eigenvalues and expanders." *Combinatorica* 6(2) (1986).
- Nilli (Noga Alon). "On the second eigenvalue of a graph." *Discrete
  Mathematics* 91(2) (1991) -- the quantified Alon-Boppana bound.
- Friedman. "A proof of Alon's second eigenvalue conjecture and related
  problems." *Memoirs of the American Mathematical Society* 195 (2008).
- Hoory, Linial, Wigderson. "Expander graphs and their applications."
  *Bulletin of the AMS* 43(4) (2006) -- survey covering Alon-Boppana
  (Thm 5.2) and Ramanujan graphs.
- Bollobas. "A probabilistic proof of an asymptotic formula for the number
  of labelled regular graphs." *European Journal of Combinatorics* 1(4)
  (1980) -- the pairing/configuration model.
