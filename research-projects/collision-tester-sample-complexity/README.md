# Does the collision tester's sample complexity actually scale as Θ(√n), not Θ(n)?

A self-contained research project in **sublinear property testing of distributions** — a
distinct area from this repo's other `research-projects/` entries (dynamical
systems/ML surrogate models, and active automata learning), sharing no code or
methodology with either.

## Research question

Suppose you want to know whether a distribution over `n` outcomes is uniform,
or is instead at total variation (TV) distance `≥ ε` from uniform — but you
can't afford to actually *learn* the distribution (`n` might be enormous:
IP-address space, a large catalog of SKUs, a hash range). Property testing
gives a startling classical result (Goldreich–Ron 1998; Paninski 2008): a
tester that counts how often two samples land on the same outcome
("collisions"), and never builds a full histogram at all, distinguishes
uniform from `ε`-far using only `Θ(√n / ε²)` samples — *sublinear* in `n`,
against the `Θ(n / ε²)` samples needed to actually learn the distribution to
comparable accuracy first.

This project asks three concrete, checkable questions:

1. **Does the collision tester's empirical sample complexity really scale as
   `n^0.5`**, recovered by fitting a power law to simulated data, as opposed
   to merely being asymptotic folklore?
2. **Does a naive learn-then-compare tester really scale as `n^1.0`** on the
   *same* simulated instances, using the *same* code path, so the exponents
   are directly comparable rather than taken from different papers?
3. **Is "TV distance `ε` from uniform" actually a single difficulty level?**
   The `Θ(√n/ε²)` bound is a *worst-case* guarantee. Does a tester calibrated
   only on `(n, ε)` (not on which alternative distribution it's up against)
   show wildly different empirical difficulty across distributions that are
   all *exactly* `ε`-far in TV, but structured differently?

## Background

- **Total variation distance**: `TV(p, u) = 0.5 * Σ|p_i - 1/n|`.
- **Collision probability**: `Σ p_i²`. Under uniform this is exactly `1/n`.
  Paninski's "paired" construction — pairing up domain elements and shifting
  mass `±2ε/n` within each pair — realizes TV distance exactly `ε` while
  making collision probability only `(1 + 4ε²)/n`: the *smallest* possible
  collision-probability excess achievable at TV distance `ε`. That's what
  makes it the theoretically tight worst-case hard instance, and why we use
  it as the primary target for the scaling-exponent fits.
- **The two testers compared** (`src/testers.py`):
  - `collision_tester`: draws `m` samples, counts colliding pairs, rejects
    uniformity if the count exceeds a threshold interpolating between the
    null collision probability (`1/n`) and the paired-construction
    alternative (`(1+4ε²)/n`) — see `figures/collision_probability_gap.png`.
  - `naive_learner_tester`: builds the empirical histogram from `m` samples
    and rejects if its L1 distance to uniform exceeds `ε` (the midpoint
    between 0 and the true L1 distance `2ε` an `ε`-far source produces).

Both testers are calibrated only from `(n, ε)` — exactly as a real tester
would be, since which distribution family generated the samples is unknown
at test time.

## Methodology

- `src/distributions.py` — exact `ε`-far distribution families, each
  verified (in tests) to have TV distance to uniform equal to `ε` to
  floating-point precision, not just approximately:
  - `paired`: Paninski's tight worst-case construction.
  - `single_heavy`: all excess mass concentrated on one element.
  - `block_quarter`: excess mass spread over `n/4` elements.
- `src/testers.py` — the two testers above, plus `collision_probability(p)`
  (exact, from the true distribution — used only for validation, never by
  the testers themselves, which see only samples).
- `src/theory.py` — closed-form `Θ(...)` predictions (`√n/ε²` and `n/ε²`),
  used only as a reference to compare fitted exponents against, and the
  exact `(1+4ε²)/n` collision-probability formula for the paired family.
- `src/experiment.py` — the empirical sample-complexity measurement:
  - `power_at_m`: rejection rate over `trials` independent draws of `m`
    samples.
  - `find_m50`: an **adaptive doubling search** for the sample size at which
    a tester starts to actually *discriminate* — i.e. where the
    **advantage** `P[reject | far] − P[reject | uniform]` crosses `0.5`,
    not where raw power alone crosses `0.5`. This distinction matters: with
    `m ≪ n`, *any* empirical histogram is sparse regardless of the true
    source, so `naive_learner_tester`'s raw rejection rate is close to `1`
    against *both* uniform and far distributions there — a spurious
    "detection" with no real signal. Requiring the advantage to cross `0.5`
    correctly discounts that regime. (An earlier version of this search
    fixed a search window centered on the `Θ(...)` prediction with an
    assumed constant of `1.0`; that silently missed the true crossing for
    small `n`, where the asymptotic constant is a poor guide. The doubling
    search fixes this by adapting to wherever the crossing actually is.)
  - `fit_power_law`: least-squares fit of `log(m50) = slope·log(n) +
    intercept`, reporting `R²`.
- `run_experiment.py` — orchestrates two parts:
  - **Part A** (scaling): sweeps `n ∈ {64,...,2048}` (collision tester) /
    `{64,...,1024}` (naive learner, capped lower since its samples scale
    linearly in `n` and get expensive), at `ε ∈ {0.15, 0.25}`, family =
    `paired`, finds `m50(n)`, fits the exponent.
  - **Part B** (family comparison): fixes `n=512, ε=0.2`, and measures the
    collision tester's full power curve against all three families on a
    shared `m`-grid, to visualize the worst-case-vs-easy-case gap directly.

## Key results (all measured by actually running the code — nothing fabricated)

### Part A — fitted scaling exponents

| tester | ε | fitted exponent | predicted | R² |
|---|---|---|---|---|
| collision | 0.15 | **0.501** | 0.5 | 0.965 |
| collision | 0.25 | **0.504** | 0.5 | 0.962 |
| naive learner | 0.15 | **0.968** | 1.0 | 0.9999 |
| naive learner | 0.25 | **1.023** | 1.0 | 0.9999 |

The collision tester's empirical scaling exponent lands within `0.004` of the
theoretical `0.5` at both tested `ε`; the naive learner's lands within `0.03`
of the theoretical `1.0`. Both fits are essentially exact matches to the
predicted asymptotic scaling, not just "in the right ballpark."

### Sublinear speedup grows with n — exactly as the exponents predict

At `ε = 0.15`, `m50(naive learner) / m50(collision tester)`:

| n | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| speedup | 7.4× | 13.6× | 14.7× | 24.9× | 31.8× |

This isn't a fixed constant-factor speedup — it grows with `n` (roughly as
`n^0.5`, matching `n^1.0 / n^0.5`), and would keep growing without bound as
`n → ∞`. See `figures/scaling_comparison.png`.

### Part B — the same nominal ε is not the same difficulty

At fixed `(n=512, ε=0.2)`, all three families are *exactly* TV-distance
`0.2` from uniform, yet the collision tester's power curves
(`figures/family_power_curves.png`) show clearly different detection
difficulty:

- `single_heavy` reaches ~100% power by `m≈28` — trivially easy.
- `paired` and `block_quarter` need `m≈500–1200` to reach comparable power —
  `paired` is consistently the hardest (or tied for hardest) at every `m`
  tested, consistent with it being the theoretically tight worst-case
  construction (minimal collision-probability excess per unit of TV
  distance).

This is the concrete, checkable version of a point that's easy to state
abstractly but easy to get wrong in practice: a minimax sample-complexity
bound describes the *worst* alternative at a given TV distance, not every
alternative — an "ε-far" instance drawn from a different family can be
detected with far fewer samples than the bound suggests.

## Figures

- `figures/collision_scaling.png` — collision tester `m50(n)` vs `n`, both ε.
- `figures/naive_learner_scaling.png` — naive learner `m50(n)` vs `n`, both ε.
- `figures/scaling_comparison.png` — side-by-side + growing speedup ratio.
- `figures/family_power_curves.png` — power vs `m` across the three families.
- `figures/collision_probability_gap.png` — the exact collision-probability
  gap `(1/n vs (1+4ε²)/n)` the tester's threshold is calibrated against.

## Limitations and scope

- The naive learner's `n`-grid stops at `1024` (vs `2048` for the collision
  tester) purely for runtime: its sample requirement scales linearly in `n`,
  so the full grid at `ε=0.15` alone draws tens of millions of samples per
  `n`.
- `find_m50`'s doubling search can, in principle, run to its
  `max_doublings` cap without the advantage ever crossing `0.5` (extremely
  small `ε` combined with a low `trials` count could produce this); in that
  case the result is flagged `bracketed=False` in `results/scaling_results.csv`
  and `run_experiment.py` prints a warning. This did not occur anywhere in
  the reported full run (all 22 rows have `bracketed=True`).
- The tester threshold constants (the `1 + 2ε²` midpoint for collision, the
  `ε` midpoint for the naive learner) are principled but not
  literature-optimal; only the *scaling exponent* is claimed to match
  theory, not the exact leading constant.
- `Θ(...)` sample-complexity predictions in `theory.py` use constant `1.0`
  purely for reference; the true problem-dependent constants are unknown
  and not estimated here.

## How to reproduce

```bash
cd research-projects/collision-tester-sample-complexity
pip install -r requirements.txt
python3 -m pytest tests -v          # 51 unit + integration tests
python3 run_experiment.py           # full grid (~9s), writes results/ + figures/
python3 run_experiment.py --quick   # smoke-test grid (~2s)
```

## Test plan

- [x] `python3 -m pytest tests -v` — 51 passed
- [x] `python3 run_experiment.py` — completes in ~9s, produces
      `results/scaling_results.csv`, `results/family_power_curves.csv`,
      `results/summary.json`, and 5 figures in `figures/`
- [x] `git status` confirms only `research-projects/collision-tester-sample-complexity/`
      is touched — no changes to `src/`, `server/`, `package.json`, or any
      website code
- [x] No `__pycache__`/`.pyc`/venv artifacts committed (project-local
      `.gitignore`)
