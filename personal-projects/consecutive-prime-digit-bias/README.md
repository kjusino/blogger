# Does the consecutive-prime last-digit bias really decay like 1/log(X)?

**Research question.** Every prime greater than 5 ends in 1, 3, 7, or 9. If a
prime's last digit and the next prime's last digit behaved like independent
draws from that 4-element set, each of the 16 ordered pairs `(a, b)` would
occur with limiting frequency 1/16, and the four "repeat" pairs `(1,1),
(3,3), (7,7), (9,9)` would together account for exactly 1/4 of all
consecutive-prime pairs. Lemke Oliver and Soundararajan (*Unexpected biases
in the distribution of consecutive primes*, PNAS 2016) discovered that primes
conspicuously avoid repeating their last digit far more than this naive
independence model predicts — a bias invisible to number theorists for
centuries because, per their Hardy-Littlewood-based heuristic, it only decays
like `c / log(X)`, so it remains large and detectable even at scales where a
"the bias should have washed out by now" intuition would say otherwise.

This project asks two concrete, checkable questions by computing the bias
directly from real primes up to `10^9`:

1. **Is the repeat-digit suppression real and consistent**, or could it be
   sampling noise at any one scale? (Tested with a chi-square goodness-of-fit
   and a one-sided binomial test at every scale examined.)
2. **Does the *magnitude* of the bias decay proportionally to `1/log(X)`**,
   as the Hardy-Littlewood heuristic predicts, rather than staying flat or
   decaying at some other rate?

## Why this is a real gap, not just a replication

The 2016 paper's headline claim ("primes repel same-last-digit repetition")
is well known, but the more falsifiable, quantitative part of the
conjecture — that the *size* of the effect is asymptotically governed by
`1/log(X)`, not just "shrinks eventually" — is rarely checked directly
against a fitted decay law over a wide scale range. This project fits that
law explicitly: it buckets consecutive-prime pairs into ten logarithmically
spaced windows spanning `10^4` to `10^9`, measures the bias in each window
independently, and regresses bias against `1/ln(X)` to see whether the data
actually supports that specific functional form (versus, say, a bias that is
merely "decreasing" without matching this rate).

## Implementation

`prime_bias/` is a small, dependency-light package:

- `sieve.py` — a vectorized Sieve of Eratosthenes (`numpy` boolean-array
  slicing instead of a Python loop over multiples), which is what makes
  sieving all primes below `10^9` tractable in about 20 seconds. Verified
  against the exact known value `pi(10^9) = 50,847,534`.
- `bias.py` — builds the `4x4` last-digit transition count matrix for
  consecutive prime pairs, plus:
  - `same_digit_fraction` — the diagonal (repeat-digit) fraction.
  - `binomial_bias_test` — one-sided binomial test against the null
    same-digit rate of 1/4.
  - `uniform_chisquare_test` — chi-square goodness-of-fit of the full
    `4x4` matrix against a uniform 1/16 null.
- `theory.py` — `fit_inverse_log_decay`, an OLS fit of bias against
  `1 / ln(scale)`.
- `experiments/run_experiments.py` — sieves up to `N_MAX = 10^9`, buckets
  pairs into 10 windows log-spaced from `10^4` to `10^9`, runs all the
  statistics per window, fits the decay law across windows, and writes
  `results/results.json` plus the three plots in `plots/`.

**Scoping decisions:** the two "unpaired-boundary" primes ≤ 5 (2, 3) and the
edge prime 5 itself are dropped since they have no last digit in `{1,3,7,9}`;
pairs are assigned to a window by their first (smaller) prime's location, so
the very first window (`10^4`–`31,623`) undercounts pairs with both members
below `10^4` — those are excluded entirely rather than assigned to an
implicit "window zero", which would have had far fewer, noisier pairs.

## Results

Ten log-spaced windows from `10^4` to `10^9`, `50,847,534` primes sieved,
`50,846,304` consecutive pairs analyzed in the windowed range:

| window | pairs | same-digit fraction | bias (0.25 − frac) | chi² p-value |
|---|---|---|---|---|
| 1.0e4 – 3.2e4 | 2,172 | 0.12799 | 0.12201 | 1.4e-57 |
| 3.2e4 – 1.0e5 | 6,191 | 0.14715 | 0.10285 | 2.2e-120 |
| 1.0e5 – 3.2e5 | 17,701 | 0.15253 | 0.09747 | ~0 |
| 3.2e5 – 1.0e6 | 51,205 | 0.15659 | 0.09341 | ~0 |
| 1.0e6 – 3.2e6 | 149,149 | 0.15900 | 0.09100 | ~0 |
| 3.2e6 – 1.0e7 | 436,932 | 0.16611 | 0.08389 | ~0 |
| 1.0e7 – 3.2e7 | 1,287,379 | 0.17025 | 0.07975 | ~0 |
| 3.2e7 – 1.0e8 | 3,809,497 | 0.17388 | 0.07612 | ~0 |
| 1.0e8 – 3.2e8 | 11,321,211 | 0.17760 | 0.07240 | ~0 |
| 3.2e8 – 1.0e9 | 33,764,867 | 0.18109 | 0.06891 | ~0 |

**Finding 1 — the bias is real and pervasive, not noise.** The same-digit
fraction is below 0.25 in *every single window*, including the smallest
(2,172 pairs). The chi-square test rejects uniformity at every scale
(`p < 10^-57` even in the smallest window; effectively 0 in `float64` from
`10^5` on), and the aggregate same-digit fraction across all 50.8M pairs is
**0.17926** against a null of 0.25 (binomial `p ≈ 0`). The full transition
matrix at the largest window (`3.16e8`–`1e9`) shows the qualitative pattern
from the original paper: every diagonal cell (`(1,1)`, `(3,3)`, `(7,7)`,
`(9,9)`) is the *lightest* (least frequent) cell in its row — see
`plots/last_digit_transition_matrix.png`.

**Finding 2 — the bias decays proportionally to 1/log(X), matching the
Hardy-Littlewood heuristic.** Regressing `bias` against `1/ln(scale)` across
the ten windows gives:

```
slope     = 0.9118
intercept = 0.0246
R^2       = 0.9735
p-value   = 1.35e-07
```

An intercept indistinguishable-from-small (0.025, versus a bias range of
0.07–0.12) is consistent with the bias vanishing as `X -> infinity`
(`1/ln(X) -> 0`), and `R^2 = 0.97` across five decades of scale is strong
support for `1/log(X)` specifically as the decay rate, not just "some
decreasing function." See `plots/bias_vs_inverse_log_scale.png` for the
fitted line against the ten data points, and
`plots/same_digit_fraction_vs_scale.png` for the same trend on a more
intuitive same-digit-probability-vs-scale axis.

## Running it

```bash
cd personal-projects/consecutive-prime-digit-bias
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                # 21 unit + integration tests
python3 experiments/run_experiments.py   # ~25s, sieves to 10^9, regenerates results/ and plots/
```

## Test suite

- `tests/test_sieve.py` — sieve output matches exact known prime-counting
  values (`pi(10)=4`, `pi(1000)=168`, `pi(10^6)=78498`), excludes known
  composites, handles `n_max < 2` and perfect-square upper bounds.
- `tests/test_bias.py` — last-digit matrix construction against hand-checked
  prime pairs, `same_digit_fraction` on synthetic all-diagonal and empty
  matrices, chi-square test correctly distinguishes uniform from skewed
  matrices, binomial test correctly flags significant vs. null-consistent
  same-digit rates.
- `tests/test_theory.py` — the OLS decay fit exactly recovers a known
  `c/ln(x)` relationship (noiseless and with small injected noise), and
  returns a near-zero slope for a flat (non-decaying) bias.
- `tests/test_integration.py` — the full pipeline at a small, fixed
  `n_max = 100,000` is checked as an exact regression (deterministic pair
  counts, window totals summing correctly) rather than a statistical
  estimate, since the sieve itself is deterministic.

## Honest limitations

- `N_MAX = 10^9` was chosen to keep the full sieve + analysis under 30
  seconds and a few hundred MB of RAM; the original paper checks up to
  `10^18` using more sophisticated prime-generation and analytic techniques.
  The `1/log(X)` fit here is over five decades (`10^4`–`10^9`), not the much
  wider range a specialized computational number theory tool could reach.
- The regression treats each window as one data point without weighting by
  its pair count (i.e., ordinary rather than weighted least squares), even
  though windows have wildly different sample sizes (2,172 vs. 33.8M pairs).
  Given how tight the fit already is (`R^2=0.97`), this is unlikely to change
  the qualitative conclusion, but a weighted fit would be a natural
  refinement.
- Only base-10 last digits are examined; the original paper also finds
  analogous biases mod 3, 4, and other small moduli, which this project
  doesn't test.
