# Spiked Covariance: Does the Sample Eigenvalue/Eigenvector Transition Really Sit at the BBP Threshold?

**Status:** complete, autonomously executed, reproducible end-to-end.

## Research question

Take `n` i.i.d. samples `x_i ~ N(0, Sigma)` in `p` dimensions, where
`Sigma = I_p + lam * v v^T` is the identity plus a single rank-one "spike"
of strength `lam` in an unknown direction `v` (`|v|=1`). This is the
canonical high-dimensional PCA model (Johnstone's spiked covariance
model): can you recover `v` from the top eigenvector of the sample
covariance `S = (1/n) X^T X`, and when does the top sample eigenvalue
even signal that a spike is there at all?

As `n, p -> infinity` with `p/n -> c in (0,1)` fixed, random matrix theory
gives a sharp answer — the **Baik-Ben Arous-Peche (BBP) transition**
(Baik, Ben Arous & Peche 2005 for the complex Wishart case; Baik &
Silverstein 2006 and Paul 2007 for the real spiked-covariance case used
here):

- **Below** a critical spike strength `lam* = sqrt(c)`, the top sample
  eigenvalue merges into the bulk of the Marchenko-Pastur distribution —
  it converges to the same value `(1+sqrt(c))^2` whether or not a spike
  is actually present — and the top sample eigenvector carries **zero**
  asymptotic information about `v` (`|<u_hat, v>|^2 -> 0`).
- **Above** threshold, the top eigenvalue detaches from the bulk to
  `(1+lam)(1+c/lam)`, and the eigenvector becomes informative:
  `|<u_hat, v>|^2 -> (1 - c/lam^2) / (1 + c/lam)`.

Both formulas are continuous at `lam = lam*` (a standard self-consistency
check, verified in `tests/test_theory.py::test_continuity_at_threshold`).
This project asks the quantitative question directly:

> **Does a straightforward Monte Carlo simulation — no special-purpose
> random matrix machinery, just sampling and `eigh`— actually reproduce
> both halves of the BBP formula (eigenvalue location *and* eigenvector
> alignment) across the full sub/supercritical range, and how fast does
> the finite-size system converge to the asymptotic prediction as it
> grows?**

This sits at the intersection of high-dimensional statistics, random
matrix theory, and machine learning (spiked covariance is the theoretical
backbone of when PCA works vs. fails in high dimensions). It's fully
self-contained: synthetic Gaussian data only, exact linear algebra
(`numpy.linalg.eigh`), no external datasets, no human judgment calls
beyond the sweep design.

## Methodology

### 1. Sampling the spiked covariance model

`src/bbp_transition/model.py` draws `x_i = Sigma^{1/2} z_i` with
`z_i ~ N(0, I_p)` i.i.d., using the closed-form rank-one square root
`Sigma^{1/2} = I_p + (sqrt(1+lam)-1) v v^T` — this avoids ever forming or
decomposing the `p x p` matrix `Sigma`, so sampling is `O(np)` per draw
rather than `O(p^3)`.

### 2. Theory

`src/bbp_transition/theory.py` implements the two closed-form BBP curves
above (`theoretical_top_eigenvalue`, `theoretical_alignment_sq`), plus the
Marchenko-Pastur bulk edge and the threshold `lam* = sqrt(c)` itself.

### 3. Experiments (`src/bbp_transition/experiment.py`)

- **Main grid sweep:** `p=150`, four aspect ratios
  `c in {0.1, 0.3, 0.5, 0.7}`, and nine spike strengths per `c` expressed
  as *ratios* of that `c`'s own threshold,
  `lam/lam* in {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3}` (so every `c` is
  probed at the same relative distance from criticality), 50 independent
  trials per cell. For each cell: mean top eigenvalue, mean squared
  alignment, 95% CI (t-distribution), and relative/absolute error against
  theory.
- **Detection-threshold estimation:** for `c in {0.2, 0.5, 0.8}`, a finer
  25-point `lam` grid (40 trials/point) with the empirical mean-alignment
  curve linearly interpolated to find where it first crosses `0.05` — an
  operational "can you detect the spike at all" threshold — compared
  against the theoretical `lam* = sqrt(c)`.
- **Finite-size scaling:** fixing `c=0.3` and `lam` at `0.5*lam*` (deep
  subcritical) and `2*lam*` (deep supercritical), the dimension is swept
  `p in {50, 100, 200, 400, 800}` (80 trials each) to see how fast
  `|empirical eigenvalue - theory|` shrinks as the system grows.

## Results

Full numeric output: `results/results.csv` (36 main-sweep cells) and
`results/summary.json` (all aggregate statistics). Four figures in
`results/`.

**1. The main sweep matches theory closely, away from the critical point
itself.**

Across all 36 `(c, lam)` cells, the top-eigenvalue relative error against
theory maxes out at **3.0%** (`max_rel_err_top_eigenvalue` in
`summary.json`), with most cells well under 1%. `results/eigenvalue_vs_lambda_c0.3.png`
and `results/phase_diagram.png` show this directly — the empirical curve
tracks the theoretical `(1+lam)(1+c/lam)` / bulk-edge piecewise curve
essentially on top of it.

The eigenvector alignment tells the more interesting story
(`results/alignment_vs_lambda_c0.3.png`): away from `lam = lam*`, the
match is tight (e.g. at `c=0.3`, `lam=3*lam*`: empirical `0.746` vs.
theory `0.752`). But **exactly at** `lam = lam*` — where the theoretical
curve has a kink from `0` to a positive slope — the empirical alignment
is `0.171` against a theoretical value of exactly `0`, the single largest
disagreement in the whole sweep (`max_abs_err_alignment = 0.171`). This
is not a bug: it's finite-size rounding of a genuine phase transition,
the same phenomenon documented for the Ising model's finite-size-smeared
critical point in this repo's other statistical-physics project — at
`p=150` the system is far from the `p -> infinity` limit where the
transition is a true discontinuity in slope, so trials right at the
threshold pick up substantial spurious alignment from sampling noise in
directions correlated by chance with `v`.

**2. A naive "alignment > 0.05" detector systematically underestimates
the true threshold.**

`results/threshold_crossing.png` plots the full fine-grained alignment
curve for three aspect ratios. Locating the empirical crossing of a fixed
`0.05` alignment threshold and comparing to `lam* = sqrt(c)`:

| c   | theory `lam*` | empirical `lam_hat` | rel. error |
|-----|---------------|----------------------|-----------:|
| 0.2 | 0.4472        | 0.2911               | 34.9%      |
| 0.5 | 0.7071        | 0.5064               | 28.4%      |
| 0.8 | 0.8944        | 0.6507               | 27.3%      |

The empirical crossing consistently sits **27-35% below** the true
asymptotic threshold, at every `c` tested. This is the same finite-size
rounding as finding 1, seen from the other side: because the transition
is smeared rather than sharp at `p=150`, alignment rises measurably above
noise-floor *before* the true `lam*`. A practitioner using a simple
"is the top eigenvalue/eigenvector informative" heuristic on
moderate-dimensional data would, on this evidence, expect to *detect* a
spike (with a lax enough threshold) somewhat before the asymptotic theory
says detection is guaranteed to work in the `n,p -> infinity` limit — a
genuinely useful, quantitative caveat to the clean textbook statement of
the BBP result.

**3. Finite-size convergence is well underway by `p ~ a few hundred`, and
faster above threshold than below.**

`results/finite_size_scaling.png` (log-log) shows
`|empirical eigenvalue - theory|` vs. `p` for `lam = 0.5*lam*` (deep
subcritical) and `lam = 2*lam*` (deep supercritical). Below threshold the
error falls smoothly from `0.079` (`p=50`) to `0.015` (`p=800`), a
log-log slope of **-0.64** — close to the `-2/3` (Tracy-Widom) fluctuation
scaling expected for a sample eigenvalue pinned to the bulk edge. Above
threshold the error is smaller throughout (`0.036` down to `~0.004-0.006`
by `p=100` and beyond) and levels off rather than continuing to shrink
cleanly — consistent with supercritical fluctuations converging faster
(the theoretical `O(n^{-1/2})` Gaussian regime rather than `O(n^{-2/3})`
Tracy-Widom), to the point that with only 80 trials/cell the residual
`~0.004-0.006` plateau is plausibly just Monte Carlo noise in the mean
rather than remaining finite-size bias — a limitation of trial count
noted honestly here rather than over-fit with a spurious slope.

## Repo layout

```
spiked-covariance-bbp-transition/
├── README.md
├── pyproject.toml / requirements.txt
├── src/bbp_transition/
│   ├── model.py            # spiked-covariance sampling + sample covariance
│   ├── theory.py            # closed-form BBP eigenvalue/alignment predictions
│   ├── experiment.py        # Monte Carlo sweeps (grid, threshold, finite-size)
│   ├── plots.py              # all matplotlib figures
│   └── run_experiment.py    # end-to-end driver
├── tests/                    # 28 unit + integration tests (pytest)
└── results/                  # results.csv, summary.json, all PNGs (committed)
```

## Reproducing

```bash
cd research-projects/spiked-covariance-bbp-transition
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/ -v                        # 28 tests, ~5s

python -m bbp_transition.run_experiment  # full sweep, ~1 min
# regenerates results/results.csv, results/summary.json, and all PNGs
```

## References

- Baik, Ben Arous, Peche. "Phase transition of the largest eigenvalue for
  nonnull complex sample covariance matrices." *Annals of Probability* 33
  (2005).
- Baik, Silverstein. "Eigenvalues of large sample covariance matrices of
  spiked population models." *Journal of Multivariate Analysis* 97
  (2006).
- Paul. "Asymptotics of sample eigenstructure for a large dimensional
  spiked covariance model." *Statistica Sinica* 17 (2007).
- Johnstone, Paul. "PCA in High Dimensions: An Orientation." *Proceedings
  of the IEEE* 106 (2018) — review connecting the BBP transition to
  practical high-dimensional PCA.
