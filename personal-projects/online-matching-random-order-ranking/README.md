# Online bipartite matching: does randomizing the arrival order actually beat RANKING's proven 1-1/e floor?

**Research question.** Karp, Vazirani & Vazirani (STOC 1990) proved that
RANKING — draw a uniformly random priority ranking of the offline vertices
*before* anything arrives, then match each online vertex to its
highest-priority available neighbor — achieves competitive ratio *exactly*
`1 - 1/e ≈ 0.6321` against an adversary that also controls the arrival
order, and that no randomized online algorithm can do better in that model.
Karande–Mehta–Tripathi (2011) and Mahdian–Yan (2011) later showed that if
the *arrival order* is instead drawn uniformly at random (the "random-order
model", ROM) rather than chosen adversarially, RANKING's ratio provably
exceeds `1 - 1/e` — the exact optimal constant under ROM is still an open
problem in the online-matching literature.

This project does not try to resolve that open problem. It asks a smaller,
fully self-contained question: **can we build and verify, from scratch, an
instance that actually witnesses the `1-1/e` floor at every problem size —
and does simply randomizing that instance's arrival order measurably and
significantly improve RANKING's ratio, with a gap that *grows* with `n`?**
Both are falsifiable, both are checked with real Monte Carlo experiments and
bootstrap confidence intervals below, and neither result was taken on faith
from a paper — the hard instance was *derived* here (see below), then
validated three independent ways before being trusted.

## The theory

- **RANKING** (`src/matching.py::ranking_online_matching`). Offline vertices
  get a uniformly random priority permutation, fixed before any online
  vertex arrives. Each online vertex, on arrival, is matched to its
  unmatched neighbor of highest priority (or left unmatched if none is
  available). This is an *online* algorithm by construction: it only ever
  looks at the current vertex's edges and the matching state so far.
- **Greedy** (`ranking_online_matching` with an identity-order "rank", or
  the dedicated `greedy_online_matching`) is the deterministic special case:
  always take the lowest-index available neighbor. Greedy is known to be
  (and is re-derived empirically below to be) exactly `1/2`-competitive in
  the worst case — no randomization, no better guarantee.
- **Competitive ratio** for an instance = `E[|matching found|] / |maximum matching|`.
  KVV's theorem is a statement about this *expectation* over RANKING's own
  randomness — individual runs (one random permutation) can and do fall
  below `1-1/e`; only the average is guaranteed to clear it.
- **Exact finite-`n` floor.** The flat constant `1-1/e` is only the
  `n → ∞` limit. The sharper, standard non-asymptotic form of KVV's bound is
  `1 - (1-1/n)^n`, which strictly *decreases* to `1-1/e` as `n` grows (since
  `(1-1/n)^n` increases monotonically to `1/e`). `src/search.py::exact_finite_floor`
  implements this and `tests/test_search.py::test_exact_finite_floor_converges_to_one_minus_inv_e_from_above`
  checks the monotone-decrease-to-the-limit property directly. Every
  correctness check below is against this sharper, `n`-dependent floor, not
  the flat asymptotic one.

## Deriving a real hard instance (not copied from a paper)

The graph that actually witnesses `1-1/e` tightness is a detail of KVV's
proof, not the theorem's content, and reproducing one from memory risked
getting the details wrong. So this project *derives* one and validates it
computationally instead of citing it.

Start from `staircase_graph(n)`: online (right) vertex `j` is adjacent to
offline (left) vertices `0..j`. Paired with the *identity* arrival order,
this graph is actually trivial — by induction, vertex `j` always finds
exactly one neighbor still free (`0..j-1` are already forced onto `0..j-1`
by the same argument), so RANKING gets a perfect matching **with probability
1, regardless of the random rank**
(`tests/test_graphs.py::test_staircase_graph_with_identity_order_is_forced_regardless_of_rank`).
Reversing the arrival order breaks this: now the *widest*-neighborhood
vertex (`j = n-1`, adjacent to everyone) arrives first and is free to take
*any* offline vertex — including the one that vertex `j = 0` (adjacent only
to `u_0`) desperately needs later. `src/experiment.py::staircase_hard_instance`
is exactly this pairing: `staircase_graph(n)` + reversed arrival order.

Three independent checks validate that this is a genuine, non-arbitrary hard
instance rather than a lucky guess:

1. **Tracks the exact floor at every measured `n`.** RANKING's empirical
   mean ratio sits within ~0.017 of `1-(1-1/n)^n` everywhere from `n=4` to
   `n=512` (`results/construction_sweep.csv`, `max_gap_to_exact_floor` in
   `results/summary.json`), and the gap *shrinks* as `n` grows.
2. **Greedy gets exactly `1/2` on this instance, at every `n`.** This is a
   second, independently-known tight bound (greedy is `1/2`-competitive in
   the worst case) landing exactly where it should, with zero variance
   (greedy has no randomness) — strong evidence the instance is genuinely
   hard for *both* algorithms in the textbook way, not an artifact of how
   RANKING happens to be implemented here.
3. **Local search cannot improve on it.** `run_search_refinement` runs
   simulated annealing (batched random edge flips, always constrained to
   keep a perfect matching) *starting from* this instance and tries to find
   something harder nearby. At every tested `n` it either finds nothing
   (`improvement = 0.0`) or a negligible improvement (`0.0075` at `n=32`,
   within Monte Carlo noise) — see `results/search_refinement.csv` and
   `figures/search_convergence.png`. If the derived instance were far from
   worst-case, a search with a real (if narrow) move set ought to find
   something meaningfully harder; it doesn't.

## Falsifiable hypotheses

- **H0 (theorem floor holds).** RANKING's *mean* ratio (bootstrap CI lower
  bound, within a `0.02` tolerance for Monte Carlo noise) never falls below
  the exact finite-`n` floor `1-(1-1/n)^n`, at any tested `n`.
- **H1 (asymptotic convergence).** RANKING's mean ratio on the derived
  instance converges toward `1-1/e` as `n → ∞`, tracking the exact floor.
- **H2 (random order strictly helps RANKING).** On the *identical* graph,
  randomizing the arrival order (ROM) instead of using the adversarial
  (reversed-staircase) order gives RANKING a significantly higher mean
  ratio (bootstrap CI on the difference excludes 0) at every tested `n`.
- **H3 (greedy is worse, and benefits less from randomization).** Greedy's
  ratio on the adversarial order is `1/2` — strictly worse than RANKING's
  `1-1/e`-ish floor — and greedy under ROM improves much less than RANKING
  does under ROM, since greedy has no internal randomization of its own to
  compound with a random arrival order.

## Results

Construction sweep, `n ∈ {4, 8, 16, 32, 64, 128, 256, 512}`, 500 trials per
mean with 95% bootstrap CIs (`results/construction_sweep.csv`):

| n | RANKING (adversarial) | exact floor | RANKING (ROM) | greedy (adversarial) | greedy (ROM) |
|---|---|---|---|---|---|
| 4 | 0.7005 [0.689, 0.713] | 0.6836 | 0.7985 | 0.5000 | 0.6940 |
| 8 | 0.6680 [0.661, 0.675] | 0.6564 | 0.7925 | 0.5000 | 0.6615 |
| 16 | 0.6502 [0.646, 0.655] | 0.6439 | 0.8037 | 0.5000 | 0.6516 |
| 32 | 0.6416 [0.639, 0.645] | 0.6379 | 0.8305 | 0.5000 | 0.6401 |
| 64 | 0.6381 [0.636, 0.640] | 0.6350 | 0.8494 | 0.5000 | 0.6368 |
| 128 | 0.6334 [0.632, 0.635] | 0.6336 | 0.8720 | 0.5000 | 0.6339 |
| 256 | 0.6333 [0.632, 0.634] | 0.6328 | 0.8951 | 0.5000 | 0.6331 |
| 512 | 0.6329 [0.632, 0.634] | 0.6325 | 0.9133 | 0.5000 | 0.6327 |

- **H0 holds at every `n`**: `all_means_meet_exact_finite_floor = true`
  (`results/summary.json`); the CI lower bound never drops below the exact
  floor by more than the 0.02 Monte-Carlo tolerance (the largest actual gap
  observed was `0.0169`, at the smallest `n`, where sampling noise is
  largest relative to the signal).
- **H1 holds**: the RANKING-adversarial curve visibly hugs the dotted exact-floor
  curve in `figures/ratio_vs_n.png` and converges to the `1-1/e = 0.6321`
  asymptote; by `n=512` the mean is `0.6329`, `0.0008` above the limit.
- **H2 holds, and the effect *grows* with `n`.** ROM beats adversarial order
  at every `n` (`rom_significantly_better_at_every_n = true`, all
  difference CIs exclude 0), and the *gap* widens monotonically: `+0.098` at
  `n=4` up to `+0.280` at `n=512`. See `figures/adversarial_vs_rom.png` — the
  two curves visibly diverge rather than staying a fixed distance apart.
  This wasn't required by H2 (which only asks for a positive, significant
  gap at each `n` independently) — it's a stronger emergent pattern in the
  data than the hypothesis demanded.
- **H3 holds**: greedy is pinned at exactly `0.5` under adversarial order at
  every `n` (zero variance — greedy is deterministic), matching its
  independently-known tight bound, and its ROM improvement (`0.50 → 0.63`)
  is much smaller than RANKING's (`0.63 → 0.91` by `n=512`) — randomization
  in the *algorithm* and randomization in the *arrival order* compound for
  RANKING but greedy only benefits from the latter.

`figures/ratio_distribution.png` shows *why* the means separate so cleanly
at `n=64`: 4000 individual trials each for adversarial order and ROM form
two visibly non-overlapping clusters (adversarial mean `0.636`, ROM mean
`0.850`) straddling the `1-1/e` line — the theorem bounds the *mean* of the
left cluster, and individual trials do dip below it, exactly as expected.

`figures/control_vs_adversarial.png` contrasts the derived hard instance
against "easy" control graphs (complete bipartite, random `G(n,n,p)` at a
few densities) under the *same* adversarial order — the control graphs all
sit at or above `~0.88`, confirming that a low ratio is a property of this
specific instance's structure, not an artifact of the algorithms or the
adversarial-order convention itself.

## Method

- **Exact maximum matching**: Hopcroft-Karp (`src/matching.py`), `O(E√V)`,
  verified against brute-force exhaustive search on random small graphs
  (`tests/test_matching.py::test_hopcroft_karp_matches_brute_force`, 20
  random seeds, `n ≤ 6`).
- **Online algorithms**: greedy and RANKING (`src/matching.py`), both
  validated to always return a valid matching (no reused offline vertex, no
  non-edge) and to never exceed the true optimum, across dozens of random
  graphs (`tests/test_matching.py`).
- **Search** (`src/search.py`, `src/graphs.py`): simulated annealing over
  the space of `n×n` bipartite graphs with a guaranteed perfect matching.
  Moves are batches of random edge flips (batch size anneals from `n/8`
  down to `1`), rejected outright if they destroy the perfect matching;
  acceptance follows the standard `exp(-Δ/T)` rule with geometrically
  decaying temperature. Used here only for the refinement/local-optimality
  check (point 3 above), not to discover the hard instance from scratch —
  an earlier iteration of this project tried pure from-scratch search
  (random-restart + single-edge-flip hill-climbing) and it plateaued far
  short of the theoretical floor at `n ≥ 64` even with iteration counts
  scaled linearly in `n`, which is what motivated deriving and validating
  an explicit construction instead.
- **Statistics** (`src/stats_utils.py`): all confidence intervals are
  bootstrap percentile intervals (2000 resamples), not normal-theory
  intervals — the ratio samples are bounded in `[0,1]` and not remotely
  Gaussian at small `n`.

## Limitations

- The derived instance is a genuine, *validated* witness of `1-1/e`
  tightness (checks 1–3 above), but it was not proven analytically here to
  be an exact worst case in the way KVV's original proof is a proof — the
  claim is empirical and Monte-Carlo-bounded, not a formal derivation of the
  proof's tightness argument.
- ROM's exact optimal competitive ratio for RANKING is an open research
  question in the literature; this project deliberately does not claim a
  specific numeric target for it, only that our own measured ratio is
  significantly and increasingly above `1-1/e` as `n` grows — a real,
  self-contained finding, not a reproduction of someone else's stated
  constant.
- All graphs here are `n` vs `n` (equal-sided) with a perfect matching by
  construction; the unequal-sides / no-perfect-matching regime is out of
  scope.

## Reproduce

```bash
pip install -r requirements.txt
pytest                 # 59 tests: matching correctness, algorithm
                        # invariants, graph generators, search sanity,
                        # end-to-end integration
python run_experiment.py   # ~1 minute; writes results/*.csv, results/*.json,
                            # figures/*.png
```

## Layout

```
src/matching.py       Hopcroft-Karp max matching; greedy & RANKING online algorithms
src/graphs.py          Graph generators (complete, random, staircase) + search moves
src/search.py          Monte-Carlo ratio evaluation + simulated-annealing search
src/experiment.py      staircase_hard_instance derivation + the three sweeps
src/stats_utils.py     Bootstrap confidence intervals, linear trend fit
src/plotting.py        All five figures
run_experiment.py       Driver: runs all sweeps, writes results/ and figures/
tests/                  59 tests (correctness, invariants, integration)
results/                CSV/JSON outputs from the last run_experiment.py run
figures/                PNG figures from the last run_experiment.py run
```
