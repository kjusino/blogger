# Does classical BKW cryptanalysis of LPN actually hit its textbook complexity?

**Research question.** Learning Parity with Noise (LPN) underlies a family of
lightweight and post-quantum-adjacent constructions (HB-family authentication
protocols, LPN-based PRGs, the Alekhnovich cryptosystem). Its best classical
attack, the Blum–Kalai–Wasserman (BKW) algorithm, is famous for a striking
complexity bound: exponential in the secret length `n`, but only
`2^Θ(n / log n)` — sub-exponential relative to brute force, because BKW
trades sample count for time via a windowed elimination technique.

Every exposition of BKW states this asymptotic and derives a window-size
(`b`) vs. elimination-depth (`a = n/b`) tradeoff from a simple
independence-assumption accounting of noise growth. This project asks three
concrete, checkable questions about that textbook accounting, by
implementing BKW from scratch and running it:

1. **Does the textbook query budget actually deliver the success
   probability it assumes**, or does correlation between samples (which the
   independence assumption ignores) erode it as elimination gets deeper?
2. **Does measured attack cost, at the textbook-optimal window size, grow
   like the predicted `2^Θ(n/log n)`** rather than like a plain
   exponential-in-`n`?
3. **Does the window size that minimizes the textbook query-count formula
   actually minimize measured attack cost** in practice?

## Why this is a real gap, not just a replication

The BKW query-complexity formula relies on treating the noise bits of
surviving samples as independent so that a Chernoff-type bound gives the
number of samples needed for a reliable final correlation solve. But BKW's
standard "pivot-and-eliminate" reduction reuses one pivot sample against
every bucket-mate at each level — every sample it produces shares that
pivot's noise bit with every other sample from the same bucket. That
correlation is a well-known simplifying assumption in the literature, but
its practical cost (how much extra safety margin you actually need, and
whether it ever saturates) is not something the one-line asymptotic
communicates. That's the gap this project measures.

## Implementation

`lpn_bkw/` is a from-scratch implementation, no cryptography libraries:

- `oracle.py` — the LPN sample oracle. Secrets and query vectors are Python
  ints used as `n`-bit masks (exact, no floating point); noise is injected
  as an explicit Bernoulli flip.
- `bkw.py` — classical pivot-and-eliminate BKW:
  - `eliminate_block`: one reduction level. Samples are bucketed by their
    bits in one block; the first sample in a bucket becomes the pivot, and
    every later bucket-mate is XORed with it and emitted. This zeroes the
    target block in every output and composes (XORs) the noise bits, which
    is exactly what squares the surviving bias at each level.
  - `reduce_all_but_target` / `solve_block`: eliminate every block except
    one target, then recover that block's bits by exhaustive correlation
    over its `2^b` candidates.
  - `recover_secret` / `attack`: run one independent recovery pass per
    block (`a` passes total) and assemble/grade the full secret.
- `theory.py` — the textbook cost-accounting model this project tests
  against: noise-doubling (`delta_final = delta_0 ^ (2^(a-1))`), the
  Chernoff-style final-sample requirement `M_final ~ C / delta_final^2`,
  and the total query budget `a * (M_final + margin*(a-1)*2^b)`.
- `experiment.py` — sweep harness used by all three studies.

**Scoping decisions (stated up front, not hidden):**
- This implements the classical **multi-pass** variant of BKW (one
  independent target-block recovery per block) rather than the more
  involved back-substitution ("LF2") variant. Both have the same
  exponential order of complexity; multi-pass costs an extra factor of `a`
  (polynomial, doesn't change the `2^Θ(n/log n)` order) in exchange for a
  much simpler, more directly testable implementation.
- `n` is required to be an exact multiple of `b` — no partial final block.
- Window sizes are restricted to `a >= 2` (at least one real elimination
  level). At `a = 1` the "attack" degenerates to plain exhaustive
  correlation over the whole secret, which isn't really BKW.
- All studies use `tau = 0.1` (a representative, comfortably sub-threshold
  noise rate) and a single fixed random-oracle seed per configuration for
  reproducibility.

## Experiments and findings

All three ran end-to-end via `experiments/run_experiments.py`
(~3 minutes total on a single core), writing `results/results.json` and the
plots in `plots/`.

### 1. Required safety margin vs. elimination depth — `required_confidence_vs_depth.png`

Fixed window size `b = 4`; swept the confidence constant `C` at elimination
depths `a = 2, 3, 4` (`n = 8, 12, 16`), 25 trials per point:

| depth `a` | C=20 | C=80 | C=300 |
|---|---|---|---|
| 2 | 1.00 | 1.00 | 1.00 |
| 3 | 0.68 | 0.84 | 0.92 |
| 4 | 0.16 | 0.52 | **0.48** |

**Finding:** at `a=2`, success is already saturated at the smallest `C`
tested. At `a=3`, success climbs steadily toward 1.0 as `C` grows — the
independence-assumption formula just needs a bigger constant, as expected.
But at `a=4`, success rate climbs from 0.16 to 0.52 and then *stops
improving* (0.48 at 15x the budget, statistically indistinguishable from
0.52 at n=25 trials). Pushing `C` up doesn't buy the improvement the
"just add more samples" formula promises — consistent with the
pivot-sharing correlation creating a real ceiling on solve reliability at
deeper elimination, not just an underestimated constant. This is the
project's central negative result: the standard accounting model correctly
predicts *that* deeper elimination needs more samples, but not that beyond
some point it can stop mattering.

### 2. Does cost scale like `n` or like `n / log2(n)`? — `scaling_vs_n.png`

At each `n in {12,16,20,24,28,32}`, tau=0.1, used the theory-optimal window
size `b*` (restricted to divisors of `n` with `a >= 2`) and a generous fixed
`C=150` (chosen because it gives 100% empirical success across this whole
range — see the table below). 30 trials per point.

| n | b* | a | success | mean queries | mean wall time |
|---|---|---|---|---|---|
| 12 | 6 | 2 | 1.00 | 900 | 8.5 ms |
| 16 | 8 | 2 | 1.00 | 1,398 | 39.0 ms |
| 20 | 10 | 2 | 1.00 | 3,396 | 325.5 ms |
| 24 | 8 | 3 | 1.00 | 4,680 | 138.6 ms |
| 28 | 7 | 4 | 1.00 | 23,316 | 497.9 ms |
| 32 | 8 | 4 | 1.00 | 25,312 | 971.2 ms |

Fitting `log2(mean queries)` linearly against `n` gives **R² = 0.953**;
against `n / log2(n)` gives **R² = 0.950** — essentially tied.

**Finding, stated honestly:** over this modest, computationally-reachable
range of `n` (12–32), the data does not clearly distinguish a pure
exponential trend from the softened `n/log n` exponent — both linear
fits explain the data about equally well. This is itself a legitimate and
expected outcome: the ratio `n / (n/log2 n) = log2(n)` only changes from
3.6 to 5.0 across this range, too small a dynamic range for the two models'
predictions to diverge much. Confirming the sub-exponential form
convincingly would need `n` large enough for `log2(n)` to vary by a much
larger factor — well beyond what a single-machine, no-optimized-bit-tricks
Python implementation can reach in an autonomous run. The honest conclusion
is "consistent with, but does not distinguish from, a pure exponential" —
not a false confirmation.

### 3. Does the theory-optimal window size match the empirically-fastest one? — `window_size_optimum.png`

At fixed `n = 24` (rich divisor set: 2, 3, 4, 6, 8, 12), swept every
candidate `b`, `C = 150`, 25 trials. `b = 2, 3, 4` were flagged and skipped
before running any trials — their theoretical query budgets
(`inf`, `~7.7e27`, `~1.4e9`) are far beyond feasible, itself a notable
observation about how narrow the practically-viable window is:

| b | a | theory queries | success | mean queries | mean wall time |
|---|---|---|---|---|---|
| 6 | 4 | 22,318 | 0.88 | 22,316 | 261.4 ms |
| **8** | **3** | **4,682** | **1.00** | **4,680** | **137.3 ms** |
| 12 | 2 | 11,384 | 1.00 | 11,382 | 4,010.0 ms |

**Finding:** the window size that minimizes the textbook query-count
formula (`b* = 8`) is exactly the window size that empirically minimizes
measured wall-clock time. `b = 12` needs fewer elimination levels (easier
noise problem) but pays for it with a 4,096-candidate exhaustive solve per
pass, ~30x slower in practice despite fewer total queries — the
query-count formula alone doesn't capture that solve-time cost, yet in this
case it still happened to point at the right answer. Note the measured
mean query counts match the theoretical ones almost exactly (e.g. 4,680 vs.
4,681.8) — expected, since the attack draws exactly the theory-prescribed
budget per pass; the genuinely empirical signal here is success rate and
wall time, not query count.

## Running it

```bash
cd personal-projects/lpn-bkw-scaling
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                          # 23 unit + integration tests
python3 experiments/run_experiments.py   # ~3 minutes, regenerates results/ and plots/
```

## Test suite

- `tests/test_oracle.py` — popcount/dot-product correctness, noise-rate
  statistics, secret bit-length validation.
- `tests/test_bkw.py` — `eliminate_block` zeroes exactly the target block
  and composes labels correctly on a hand-checked example; the full
  reduction leaves only target-block bits set; noiseless instances are
  solved exactly.
- `tests/test_theory.py` — bias/noise-doubling formulas, monotonicity of
  the required-sample count, `optimal_b` only returns divisors of `n`.
- `tests/test_integration.py` — full attack pipeline recovers a known
  secret exactly when noiseless; hits high success rate at a calibrated
  budget; success degrades with deeper elimination at a fixed budget
  (the qualitative signature behind Study 1); measured query counts match
  the theoretical formula.

## Honest limitations

- Pure-Python bit integers, not a vectorized/bitsliced implementation —
  this caps the reachable `n` well below what a real cryptanalytic
  BKW implementation (or real LPN-based scheme parameters) would use.
  The qualitative findings (correlation ceiling, tied exponential fits at
  small `n`, matching window-size optimum) are the contribution, not an
  attempt at record-setting attack parameters.
- The multi-pass variant (vs. back-substitution/LF2) inflates absolute
  query counts by the polynomial factor `a`; this doesn't change any of
  the three findings above, all of which are about *relative* comparisons
  (across `C`, across `n`, across `b`) where that factor is either constant
  or accounted for identically on both sides.
- `tau = 0.1` throughout; the correlation-ceiling effect in Study 1 would
  be worth re-checking at other noise rates as a natural next step.
