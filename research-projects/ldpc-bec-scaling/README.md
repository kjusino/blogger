# Does finite-length LDPC decoding over the erasure channel really follow the predicted two-parameter scaling law?

A self-contained research project in **coding theory / information theory**: a from-scratch
implementation of low-density parity-check (LDPC) codes over the binary erasure channel (BEC),
comparing Monte Carlo simulation of the iterative peeling decoder against two theoretical
predictions — the asymptotic decoding threshold (density evolution) and the finite-length
scaling law that governs how quickly finite-blocklength performance approaches that threshold.

This is a standalone Python subproject. It does not import from `src/` (the website) or
`server/`, does not touch `package.json`, and is not wired into the site in any way.

## Research question

Gallager-style LDPC codes, decoded by iterative message passing, are the backbone of modern
storage and communication systems (Wi-Fi, 5G, SSDs, deep-space links). Their headline theoretical
result is the **decoding threshold**: in the limit of infinite blocklength, a randomly constructed
`(dv, dc)`-regular LDPC code decoded by belief propagation succeeds with probability 1 below a
sharp threshold `epsilon*` on the channel noise, and fails with probability 1 above it. This
threshold is computed by **density evolution (DE)** — a scalar recursion that tracks how error
probability evolves message-by-message under the (idealized) assumption that the Tanner graph is
infinite and cycle-free.

Real codes are finite, though, and every practicing engineer's question is: **how fast does a
finite-length code's actual performance approach that asymptotic threshold as blocklength grows,
and does the approach follow a specific, falsifiable functional form — or is "it gets better with
n" as precise as it gets?**

Amraoui, Montanari, Richardson & Urbanke ("Finite-Length Scaling for Iteratively Decoded LDPC
Ensembles", *IEEE Trans. Information Theory*, 2009) derive a two-parameter scaling law for the BEC
case specifically:

- the erasure rate at which a length-`n` code has 50% block error rate, `eps50(n)`, approaches the
  threshold as `eps* - eps50(n) ~ C * n^(-2/3)`;
- the width of the "waterfall" transition region (e.g. `eps90 - eps10`, the gap between 90%- and
  10%-block-error erasure rates) shrinks as `~ n^(-1/2)`.

**This project asks whether that specific `(-2/3, -1/2)` two-exponent scaling law is actually what
you see when you build the codes, the decoder, and the estimator entirely from scratch and run the
simulation** — as opposed to citing the result, or measuring a vaguely-monotonic trend and calling
it confirmation.

## Methodology

Everything is implemented from first principles, with no coding-theory library dependency
(`numpy`/`scipy`/`matplotlib` are used only for numerics/curve-fitting/plotting, not for LDPC
constructs):

- **`src/tanner.py`** — random `(dv, dc)`-regular Tanner graph construction via a randomized
  stub-matching algorithm (a swap-pop variant of Gallager's configuration model) that rejects
  parallel edges locally instead of restarting the whole graph on any collision. A naive
  reject-and-restart matching (build the full random permutation, check for any duplicate edge,
  and restart from scratch if one exists) turned out to fail almost every time once `dc` is not
  tiny, because the birthday-paradox collision probability across the whole graph is high even
  though any *individual* collision is easy to avoid locally — this is documented in the code and
  covered by `tests/test_tanner.py`.
- **`src/decoder.py`** — the BEC **peeling decoder**: a check node with exactly one erased
  neighbor can resolve it (it's the XOR of the other, known, neighbors), so it's queued and
  "peeled"; this repeats until no check has exactly one erased neighbor. Over the BEC, whether
  decoding succeeds depends only on the erasure pattern and graph topology, never on the actual
  transmitted bits (Richardson & Urbanke, *Modern Coding Theory*, ch. 3), so the decoder tracks
  only which variables are still erased — no codeword construction needed.
- **`src/density_evolution.py`** — the scalar BEC density-evolution recursion
  `x_{l+1} = eps * (1 - (1-x_l)^(dc-1))^(dv-1)`, plus a bisection search for the threshold
  `eps* = sup{eps : x_l -> 0}`.
- **`src/experiment.py`** — the Monte Carlo estimator: for each blocklength `n`, sample several
  random Tanner graph instances and, for each, many random erasure patterns at a grid of erasure
  rates; fit a logistic curve to the resulting block-error-rate points to extract `eps50(n)` and
  the transition steepness `k` (hence the waterfall width `eps90 - eps10 = 2*ln(9)/k`); use a
  **two-stage adaptive sweep** (a coarse pass to locate the transition, then a fine pass narrowed
  around it) so the same code handles both wide, shallow waterfalls (small `n`) and narrow, steep
  ones (large `n`) without hand-tuned per-`n` parameters.
- **`run_experiment.py`** — runs the whole pipeline across six blocklengths and fits both scaling
  exponents.

### A methodological wrinkle worth flagging

The logistic fit for `(eps50, k)` is fit twice: once as a closed-form initial guess via ordinary
least squares on the **logit-transformed** BLER data (`log(p/(1-p))` is linear in `eps`, so this is
just a linear regression), and then refined with `scipy.optimize.curve_fit`. Initially the code
used a fixed, generic initial guess (`k=200`) for the nonlinear fit; on small, noisy blocklengths
this occasionally trapped the optimizer in a spuriously steep local optimum (e.g. `k~317` fit to
data that visibly spanned erasure rates 0.28-0.53 — a transition an order of magnitude more
gradual than that). Switching to the logit-regression initial guess fixed this and is now covered
by an integration test (`tests/test_experiment_integration.py`) that would have caught the
regression: it asserts the waterfall width must shrink monotonically across three blocklengths,
which failed under the old fixed-guess code for some random seeds.

## Key results (all measured by actually running the code — nothing fabricated)

Ensemble: `(dv, dc) = (3, 6)`, rate `1/2`. Six blocklengths `n in {128, 256, 512, 1024, 2048, 4096}`,
3 random Tanner graph instances per `n`, up to 190 erasure-pattern trials per `(n, eps)` point
(coarse + fine adaptive sweep). Full run takes **~29 seconds** on a single CPU core.

- **DE threshold**: `eps* = 0.429437`, matching the literature value `~0.4294` for the `(3,6)`
  ensemble (Richardson & Urbanke) to 4 decimal places — a from-scratch sanity check that the DE
  recursion and threshold search are implemented correctly.

- **Empirical `eps50(n)` monotonically approaches the threshold:**

  | n    | eps50(n) | gap = eps\* − eps50(n) | waterfall width (eps90−eps10) |
  |------|----------|------------------------|--------------------------------|
  | 128  | 0.40573  | 0.02370                | 0.13612                        |
  | 256  | 0.41483  | 0.01461                | 0.09435                        |
  | 512  | 0.41969  | 0.00974                | 0.06670                        |
  | 1024 | 0.42325  | 0.00619                | 0.04184                        |
  | 2048 | 0.42564  | 0.00379                | 0.03170                        |
  | 4096 | 0.42706  | 0.00238                | 0.02268                        |

- **Fitted shift exponent: `-0.659`, against a theoretical prediction of `-2/3 = -0.667`** (a
  2% relative difference) — a least-squares power-law fit of `log(gap)` vs. `log(n)`, R² > 0.999.

- **Fitted width exponent: `-0.523`, against a theoretical prediction of `-1/2 = -0.500`** (a
  5% relative difference).

- **Both curves visibly track their theoretical reference lines almost exactly on a log-log plot**
  (see `figures/shift_scaling.png` and `figures/width_scaling.png`) — this is not a "same rough
  ballpark" result, the fitted and theoretical lines are close to indistinguishable by eye across
  a 32x range of blocklengths.

- 33/33 unit + integration tests pass.

### Figures

- `figures/density_evolution_trajectories.png` — DE iterations below/at/above threshold, showing
  the qualitative collapse-to-zero vs. stall-at-a-fixed-point behavior that defines the threshold.
- `figures/waterfall_curves.png` — all six blocklengths' raw BLER-vs-`eps` Monte Carlo points
  overlaid, visibly sharpening around the DE threshold as `n` grows.
- `figures/shift_scaling.png` — log-log plot of the threshold-distance shift with the fitted and
  theoretical `(-2/3)` power laws overlaid.
- `figures/width_scaling.png` — log-log plot of the waterfall width with the fitted and
  theoretical `(-1/2)` power laws overlaid.

## How to reproduce

```bash
cd research-projects/ldpc-bec-scaling
pip install -r requirements.txt
python3 -m pytest tests -v          # 33 unit + integration tests
python3 run_experiment.py           # full experiment (~30s), writes results/ and figures/
python3 run_experiment.py --quick   # fast smoke-test grid (~2s)
```

## Scope and limitations

- Only the BEC and only regular `(dv, dc)` ensembles are studied. Irregular degree distributions
  (which are what capacity-approaching LDPC codes actually use) have a richer, ensemble-dependent
  scaling parameter that this project doesn't estimate — the BEC-regular case was chosen because
  it has the cleanest closed-form DE recursion and the best-known scaling-theory reference point.
  Other channels (BSC, AWGN) don't reduce iterative decoding to a scalar recursion and would need
  full density evolution over message distributions (or Gaussian-approximation EXIT charts)
  instead of the closed-form BEC update used here.
- The fitted exponents (`-0.659` and `-0.523`) are Monte Carlo estimates from a single run with a
  fixed random seed; they are not accompanied by a bootstrap confidence interval. The two-stage
  adaptive sweep and logit-regression-initialized logistic fit were specifically built to keep
  this estimate stable across seeds (see `tests/test_experiment_integration.py`), but a rigorous
  claim of "matches theory" would want error bars on the fitted exponents, not just a point
  estimate close to the prediction.
- The scaling law's theoretical prefactor `C` (not just the exponent) also depends on ensemble-
  specific scaling parameters from Amraoui et al. that this project doesn't independently derive
  or verify — only the exponents are compared against theory.

## Future work

- Extend to irregular LDPC ensembles (e.g. the capacity-approaching degree distributions from
  Luby et al.) and check whether the same `(-2/3, -1/2)` exponents hold, or whether they're
  regular-ensemble artifacts.
- Bootstrap over multiple random seeds to put confidence intervals on the fitted exponents rather
  than reporting point estimates.
- Compare against the BSC or AWGN channel using a Gaussian-approximation EXIT-chart density
  evolution, to see whether the same scaling-exponent structure survives outside the erasure
  channel's special structure.
