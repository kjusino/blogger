# Random Walk on the Hypercube: Does Mixing Really Cut Off at (n ln n)/2 — and Does the Classical Bound Get the *Shape* Right?

**Status:** complete, autonomously executed, reproducible end-to-end.

## Research question

Consider the "coordinate-refresh" walk on `{0,1}^n`: at each step, pick a
coordinate `i` uniformly at random and overwrite `x_i` with a fresh,
independent fair coin flip. This is one of the canonical examples in the
theory of Markov chain mixing times (Diaconis, Graham & Morrison 1990;
Diaconis 1988, Ch. 6) and the textbook illustration of the **cutoff
phenomenon**: rather than decaying smoothly, the distance to stationarity
stays near its maximum for a long time and then collapses to near zero over
a comparatively short window. Diagonalizing the chain in the Walsh-Hadamard
(Fourier) basis of `{0,1}^n` gives exact eigenvalues `lambda_k = 1 - k/n`
(multiplicity `C(n,k)`, for a spectral subset of size `k`), and the standard
Diaconis-Shahshahani chi-square (Cauchy-Schwarz) argument turns this into an
upper bound on total variation (TV) distance predicting a cutoff at

```
t*(n) = (n ln n) / 2,   window = Theta(n)
```

This project asks two nested questions, both fully computational:

> **Does the *exact* mixing curve of this chain — not an upper bound, the
> real total-variation distance — actually exhibit a cutoff at `(n ln n)/2`
> with a `Theta(n)` window? And separately: does the classical chi-square
> bound, which is what actually produces that `(n ln n)/2` prediction, get
> the *shape* of the mixing curve right, or only its location and scale?**

The second question turns out to have a genuinely two-sided answer (see
Results): the cutoff *location* and *window scaling* match the chi-square
theory closely and increasingly well as `n` grows, but the chi-square
bound's own universal *profile shape* is provably not what the true curve
converges to — a persistent, non-vanishing gap that is easy to miss if you
only check the cutoff time and window, which is exactly what most
expositions of this example do.

This sits squarely in probability theory / theoretical CS (Markov chain
mixing times, spectral methods, the cutoff phenomenon — a major research
area associated with Diaconis, Aldous, Peres, Levin and others), is
completely self-contained (no external data, no GPU, just linear algebra and
Monte Carlo), and is answerable to machine precision at the core (via an
exact reduction, described below) rather than only through simulation noise.

## Methodology

### 1. An exact reduction: from `2^n` states to `n+1`

The full chain on `{0,1}^n` is intractable to simulate exactly for `n` in
the hundreds or thousands. But the chain (and its start state, `0^n`) is
invariant under permuting coordinates, so the law of the state at every time
`t` is *exchangeable* — and since the symmetric group acts transitively on
subsets of a fixed size, conditioning an exchangeable distribution on the
Hamming weight always gives the **uniform distribution over subsets of that
weight**. This is a standard but easy-to-overlook fact, and it means the
Hamming weight `W_t` is an *exact* lumping of the chain:

```
TV(Law(X_t), Uniform({0,1}^n))  ==  TV(Law(W_t), Binomial(n, 1/2))     (exactly, for every t)
```

`W_t` is itself a birth-death chain on `{0, ..., n}` (`src/hypercube_cutoff/chain.py`):
from weight `w`, go to `w-1` w.p. `w/(2n)`, to `w+1` w.p. `(n-w)/(2n)`, and
stay w.p. `1/2`. Forward-propagating an `(n+1)`-length probability vector
through this chain gives the *exact* TV-distance curve for `n` in the
thousands in well under a second — no Monte Carlo needed for the main
scaling study. `chain.py` also brute-forces the literal `2^n`-state chain for
small `n` (`n <= 14`) to check the lumping claim empirically, not just
algebraically.

### 2. Theory (`src/hypercube_cutoff/theory.py`)

Exact eigenvalues `lambda_k = 1-k/n`; the exact chi-square distance
`chi_sq(t) = sum_{k=1}^n C(n,k) lambda_k^{2t}` (computed in log-space via
`scipy.special.gammaln` + `logsumexp`, accurate for `n` in the tens of
thousands where `C(n, n/2)` alone overflows double precision); the
Diaconis-Shahshahani bound `TV(t) <= sqrt(chi_sq(t))/2`; the asymptotic
cutoff time `t*(n) = (n ln n)/2`; a rescaling `c = (2t - n ln n)/n` (so
`t = (n/2)(ln n + c)`); and the `n -> infinity` limit of the *bound itself*
under that rescaling, `limiting_profile(c) = sqrt(exp(exp(-c)) - 1) / 2`,
obtained from the small-`k` approximation `C(n,k) ~ n^k/k!` of the
chi-square sum.

### 3. Monte Carlo (`src/hypercube_cutoff/simulate.py`)

Two independent simulators, used to validate that the exact algebra actually
describes real sample paths: (a) literal bit-vector walkers (`num_trials`
independent length-`n` states, each step overwriting a random coordinate),
and (b) a weight-only simulator using the birth-death jump probabilities
directly (valid because of the exact lumging, and cheaper for large `n`).
`empirical_tv_distance` estimates TV from a batch of sampled weights with a
bootstrap confidence interval.

### 4. Experiments (`src/hypercube_cutoff/experiment.py`)

- **Lumping validation** (`n in {8, 10, 12}`): compare the exact lumped
  chain, the exact brute-force `2^n` chain, and Monte Carlo bit-vector
  simulation (20,000 trials) at several times spanning the cutoff window.
- **Cutoff-location / data-collapse sweep** (`n in {50, 100, 200, 400, 800,
  1600, 3200}`, no Monte Carlo — purely exact): for each `n`, compute the
  exact TV curve over a 41-point grid in the rescaled variable
  `c in [-4, 6]`, then (i) interpolate the empirical half-mixing time
  `t_{1/2}` (where exact TV crosses 0.5) and compare it to `t*(n)`, (ii)
  measure the window as `t_{0.25} - t_{0.75}` and check it scales linearly
  in `n`, (iii) check whether the rescaled exact-TV curves collapse onto
  *each other* (using the largest-`n` curve as a reference), and (iv)
  separately check whether they collapse onto the chi-square bound's
  `limiting_profile`.
- **Monte Carlo validation** (`n=30`, 20,000 trials): compare exact TV
  against empirical estimates (with 95% bootstrap CIs) from both simulators
  across the cutoff window.

## Results

Full numeric output: `results/*.csv` and `results/summary.json`. Six figures
in `results/`.

**1. The lumping is exact, not approximate.** Across `n in {8,10,12}` and
several times, the brute-force `2^n`-state chain and the lumped birth-death
chain agree to `2.22e-16` — machine epsilon (`results/lumping_check.png`,
left panel; `max_abs_diff_lumped_vs_bruteforce` in `summary.json`). Literal
bit-vector Monte Carlo (20,000 trials) lands inside its own 95% CI around the
exact value 95.2% of the time — consistent with nominal coverage
(`results/lumping_check.png`, right panel).

**2. The cutoff is real, and its location/window match theory increasingly
well as n grows.** `results/tv_curves_vs_t.png` shows the raw mixing curves
for `n = 100, 400, 1600, 3200`: on a log scale, each curve is flat near 1
for a long stretch and then collapses sharply — the qualitative signature of
a cutoff, not a smooth exponential decay. Quantitatively
(`results/cutoff_time_scaling.png`), the relative error between the
empirical half-mixing time and `t*(n) = (n ln n)/2` shrinks monotonically
from **14.9% at n=50 to 7.4% at n=3200**. The window (`results/window_scaling.png`)
fits `window ∝ n^1.01` — matching the theoretical `Theta(n)` prediction
almost exactly — and `window/n` itself converges from 1.22 to 1.284 as `n`
grows.

**3. The rescaled exact-TV curves collapse onto each other — but *not* onto
the classical chi-square bound's universal profile.** This is the most
interesting finding. Plotting exact TV against the rescaled variable `c`
(`results/data_collapse.png`), the curves for `n = 100, 400, 1600, 3200` sit
essentially on top of each other — a real cutoff-window data collapse. But
they do **not** converge to the chi-square bound's `limiting_profile`
curve: there's a clear, persistent gap (e.g. at `c=0`, exact TV `≈0.38` vs.
the bound's asymptotic prediction `≈0.655`). `results/collapse_errors.png`
makes this precise: the **self-collapse error** (max deviation from the
largest-`n` curve in the sweep) shrinks steadily — `0.031 → 0.017 → 0.0086 →
0.0047 → 0.0019 → 0.0007` as `n` doubles from 50 to 1600, roughly halving
each time (an `O(1/n)`-type finite-size correction to a genuine limiting
profile) — while the **gap to the chi-square-bound profile** stays flat at
`≈0.52` across the *entire* two-decade range of `n` tested.

In other words: the textbook chi-square argument correctly predicts *where*
and *how fast* (in the `Theta(n)` sense) this chain cuts off, but the
specific profile curve it hands you is not the curve the chain actually
converges to — a real quantitative gap between "the bound has the right
order" and "the bound is asymptotically sharp," which is easy to conflate
when a paper or course only checks the cutoff location and window and never
plots the rescaled curves against each other. (Separately, I verified the
*asymptotic approximation itself* is correct: the exact finite-`n`
chi-square bound converges to `limiting_profile(c)` as `n` grows — checked
at `n=20,000` in `tests/test_theory.py` — so the gap in finding 3 is a
genuine gap between the true TV and the Cauchy-Schwarz bound, not an
error in the asymptotic expansion of the bound itself.)

**4. Monte Carlo simulation of the literal chain matches the exact curve.**
At `n=30` (`results/mc_validation.png`), literal bit-vector Monte Carlo (20,000
trials per point) tracks the exact TV curve across the whole cutoff window,
landing inside its 95% CI at **100%** of the tested points; the weight-only
simulator (which relies on the same lumping argument) does so at **92.3%** —
both consistent with the exact chain being correct and the lumping being
valid for actual sample paths, not just in the algebra.

## Repo layout

```
hypercube-walk-cutoff/
├── README.md
├── pyproject.toml / requirements.txt
├── src/hypercube_cutoff/
│   ├── theory.py          # exact eigenvalues, chi-square bound, cutoff time, universal profile
│   ├── chain.py            # exact birth-death lumped chain + brute-force 2^n validation
│   ├── simulate.py         # Monte Carlo: literal bit-vector walkers + weight-only walkers
│   ├── experiment.py       # the three sweeps (lumping validation, cutoff scaling, MC validation)
│   ├── plots.py             # all matplotlib figures
│   └── run_experiment.py   # end-to-end driver
├── tests/                   # 45 unit + integration tests (pytest)
└── results/                 # CSVs, summary.json, all PNGs (committed)
```

## Reproducing

```bash
cd research-projects/hypercube-walk-cutoff
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/ -v                          # 45 tests, ~2s

python -m hypercube_cutoff.run_experiment  # full sweep, ~6s
# regenerates results/*.csv, results/summary.json, and all PNGs
```

## References

- Diaconis, Graham, Morrison. "Asymptotic analysis of a random walk on a
  hypercube with many dimensions." *Random Structures & Algorithms* 1(1)
  (1990).
- Diaconis. *Group Representations in Probability and Statistics*. IMS
  Lecture Notes-Monograph Series 11 (1988).
- Diaconis. "The cutoff phenomenon in finite Markov chains." *Proceedings
  of the National Academy of Sciences* 93(4) (1996).
- Levin, Peres, Wilmer. *Markov Chains and Mixing Times*, 2nd ed. American
  Mathematical Society (2017) — standard reference for the cutoff
  phenomenon and this chain's spectral analysis.
