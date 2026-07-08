# Is the price of anarchy in selfish routing really independent of network topology?

A self-contained research project in **algorithmic game theory / network
optimization** — a distinct area from this repo's other open
`research-projects/` entries (high-dimensional convex geometry, spectral
graph theory, Markov-chain mixing times, random matrix theory, quantum
circuit optimization, topological data analysis, statistical learning
theory, random geometric graphs, differential privacy, statistical
physics, online algorithms with predictions, coding theory, sublinear
property testing, active automata learning, and learned dynamical-systems
surrogates), sharing no code or methodology with any of them.

## Research question

When selfish drivers each pick the route that's fastest *for them*, traffic
doesn't settle into the routing that's fastest for everyone collectively.
The **price of anarchy (PoA)** — Koutsoupias & Papadimitriou, 1999 — is the
worst-case ratio between the total travel cost at this selfish (Wardrop)
equilibrium and the cost of the socially optimal routing.

Roughgarden's landmark result (*"The Price of Anarchy Is Independent of the
Network Topology"*, JCSS 2003; the p=1 case is Roughgarden & Tardos, JACM
2002) says something much stronger than "PoA is bounded": for latency
functions that are **polynomials of degree at most `p` with nonnegative
coefficients**, the worst-case PoA over *every conceivable network* equals
the worst-case PoA already achievable on the *simplest possible network* —
two parallel links between one source and one sink. A million-edge city
grid cannot produce a worse ratio than a two-edge toy example can.

This project asks three concrete, checkable questions:

1. **Does the empirical PoA on a large battery of randomly generated,
   structurally complex networks ever exceed the closed-form bound**
   `beta(p) = 1 / (1 - p*(p+1)^(-(p+1)/p))`, derived here from first
   principles (not copied from a citation), for `p in {1, 2, 3, 4}`?
2. **Is that bound actually tight**, i.e. does a network as simple as two
   parallel links come arbitrarily close to `beta(p)` when its parameters
   are chosen adversarially — reproduced through the *same* generic solver
   used for question 1, not a closed-form shortcut?
3. **Is the worst case topology-independent in practice**, i.e. does no
   random complex topology in the battery ever beat (get closer to
   `beta(p)` than) the two-link worst case does?

## Background

**Nonatomic routing game.** A directed graph, a single source `s` and sink
`t`, and a continuum of infinitesimal users routing one unit of total flow
from `s` to `t`. Each edge `e` has a latency function `l_e(x)` (nonneg
coefficients here), the delay experienced by a user of that edge when its
flow is `x`.

**Wardrop equilibrium** (Wardrop, 1952; Beckmann-McGuire-Winsten, 1956).
A flow is at equilibrium iff every `s`-`t` path actually used has minimal
and equal latency among all paths. Equivalently, it is the minimizer of the
convex potential

```
Phi(f) = sum_e  integral_0^{f_e}  l_e(x) dx
```

over feasible flows — this is a standard but non-obvious fact (the
first-order condition of this convex program is exactly the equal-latency
condition), and is what `src/solvers.py` exploits: it never enumerates
paths, it minimizes `Phi` directly over edge flows subject to flow
conservation, so it scales to graphs with exponentially many `s`-`t` paths.

**Social optimum.** The flow minimizing total cost `C(f) = sum_e f_e *
l_e(f_e)`, a *different* convex program over the same feasible region.

**PoA bound for polynomials (`src/theory.py`).** For latency functions
`l_e(x) = sum_{k=0}^{p} a_k x^k` with `a_k >= 0`, Roughgarden's variational
argument (via the equilibrium first-order condition) reduces the worst
case to a scalar optimization, giving the closed form

```
beta(p) = 1 / (1 - p * (p+1)^(-(p+1)/p))
```

`beta(1) = 4/3` (the famous linear-latency bound); `beta(4) ≈ 2.151`
(relevant to the degree-4 BPR curves the Bureau of Public Roads actually
uses for real highway latency modeling).

**Independent from-scratch cross-check.** Rather than trust the formula on
citation alone, `src/theory.py` re-derives the same number a second way:
on the two-link network (edge 1: `l(x) = x^p`; edge 2: `l(x) = b`, a
constant), the equilibrium and optimal costs have closed forms in `b`
(worked out in `theory.py`'s docstrings), and numerically maximizing
`PoA(b)` over `b` reproduces `beta(p)` to `~1e-9` for `p = 1..6`
(`tests/test_theory.py::test_worst_case_two_link_matches_closed_form_bound`).
A *third* independent check comes from running that same two-link network
through the generic NLP solver (`src/solvers.py`, no closed-form shortcut)
— see Question 2 below.

## Methodology

- `src/cost_functions.py` — `Polynomial`, nonnegative-coefficient
  latency functions with `l(x)`, `l'(x)`, `integral(x)` (for the
  equilibrium potential), and `x*l(x)` + its gradient (for the social-cost
  objective), all unit-tested against finite differences / numeric
  quadrature.
- `src/network.py` — directed multigraphs via a node-arc incidence matrix
  (no path enumeration); a random **series-parallel** network generator
  (the standard recursive series/parallel-composition construction) that
  is connected and dangling-edge-free by construction; and four textbook
  instances (Pigou's example, Braess's paradox with/without its shortcut
  edge, three parallel links) used purely to validate the solver.
- `src/solvers.py` — `solve_equilibrium` and `solve_optimum`: both convex
  programs (`scipy.optimize.minimize`, SLSQP, analytic gradients, a
  feasible starting flow from an LP phase-1 step) over edge flows subject
  to flow conservation.
- `src/theory.py` — `poa_bound(p)` (the closed form) and the from-scratch
  two-link closed-form re-derivation used to cross-check it.
- `src/experiment.py` — `run_topology_battery` (random networks × degrees,
  empirical PoA vs. bound) and `run_worst_case_convergence` (two-link `b`
  sweep through the generic solver).
- `run_experiment.py` — orchestrates both experiments and all figures:
  `p in {1,2,3,4}`, 300 random series-parallel networks per degree (up to
  20 edges each, 1,200 networks / 2,400 NLP solves), plus an 800-point
  two-link `b`-sweep.

## Key results (all measured by actually running the code — nothing fabricated)

Full run: 1,200 random topologies + 800 two-link sweep points, **9.6s**
total (6.3s battery + 3.4s sweep).

### Question 1 — does any topology ever beat the bound?

| degree p | theoretical bound β(p) | max empirical PoA (1,200 topologies) | violations |
|---|---|---|---|
| 1 | 1.3333 | 1.2382 | **0 / 300** |
| 2 | 1.6258 | 1.4841 | **0 / 300** |
| 3 | 1.8956 | 1.4513 | **0 / 300** |
| 4 | 2.1505 | 2.0987 | **0 / 300** |

Zero violations across all 1,200 random series-parallel networks (3–20
edges each, random nonnegative polynomial coefficients) and all four
degree classes. See `figures/poa_distributions.png` (violin plots — the
entire empirical mass sits below the dashed theoretical line for every
degree) and `figures/poa_vs_degree.png`.

### Question 2 — is the bound actually tight?

| degree p | β(p) | two-link peak PoA (generic solver) | % of bound |
|---|---|---|---|
| 1 | 1.3333 | 1.3282 | 99.6% |
| 2 | 1.6258 | 1.6193 | 99.6% |
| 3 | 1.8956 | 1.8879 | 99.6% |
| 4 | 2.1505 | 2.1416 | 99.6% |

The generic NLP solver — the same code path used for arbitrary random
topologies, with no closed-form shortcut — recovers **99.6% of the
theoretical bound at every tested degree** just by sweeping `b` on the
simplest possible two-edge network. See `figures/convergence_two_link.png`:
each panel's peak sits right at the dashed `beta(p)` line, at `b ≈ 1` in
every case (matching the closed-form worst case derived independently in
`theory.py` — see Background).

### Question 3 — is the worst case topology-independent?

Putting Q1 and Q2 together (`figures/poa_vs_degree.png`): the two-link
network's peak PoA (orange triangles) sits almost exactly on the
theoretical curve at every degree, while the random-topology maxima (blue
circles) fall meaningfully below it — 76–98% of the bound depending on
degree, never above it. More edges, more parallel routes, more nesting of
series/parallel structure bought the random search *nothing*: the
two-edge worst case wasn't beaten by any of the 1,200 more complex
topologies tried. That is Roughgarden's topology-independence theorem,
observed rather than assumed.

### Solver correctness (validated against closed forms before trusting any of the above)

- **Pigou's example** (`l_1(x)=x`, `l_2(x)=1`): solver gives equilibrium
  cost `1.000000`, optimum cost `0.750000`, PoA `1.333333` — the textbook
  4/3, exactly, to `1e-4`.
- **Braess's paradox**: without the free shortcut edge, equilibrium cost
  `1.500000` (already socially optimal); *adding* a zero-cost edge raises
  equilibrium cost to `2.000000` — the paradox, reproduced exactly
  (`figures/braess_paradox.png`).

## Figures

- `figures/poa_vs_degree.png` — theoretical `beta(p)` curve vs. max
  empirical PoA (random topologies) and the two-link worst-case peak, all
  four degrees.
- `figures/poa_distributions.png` — violin plots of the full empirical PoA
  distribution per degree against the theoretical worst case.
- `figures/convergence_two_link.png` — PoA(b) on the two-link network for
  each degree, generic solver, peaking at the theoretical bound.
- `figures/braess_paradox.png` — the classic paradox as a solver sanity
  check.

## Limitations and scope

- Single-commodity routing games only (one source-sink pair); Roughgarden's
  topology-independence theorem is more general (multicommodity), but the
  single-commodity case already exercises the full mechanism and keeps the
  convex program simple and exactly solvable.
- Random topologies are restricted to the series-parallel class. This is
  the standard construction for routing-game examples and keeps every
  generated graph guaranteed acyclic, connected, and free of dangling
  edges by construction; it does not cover every possible DAG topology
  (e.g. non-series-parallel graphs like the Braess network's own
  4-node/5-edge shape — included separately as a named example, not
  generated by the random constructor).
- Degrees tested are `p in {1,2,3,4}` (affine through the degree-4 BPR-style
  curves used in real traffic engineering); the bound formula and both
  solvers work for any `p >= 1`, this range was chosen so the full battery
  runs in well under a minute.
- 300 trials per degree gives a good empirical spread without the random
  search plausibly finding the exact worst case (see Q3) — that's the
  point (the theorem is about the *bound*, not about random search finding
  it), not a limitation of the solver.

## How to reproduce

```bash
cd research-projects/price-of-anarchy-selfish-routing
pip install -r requirements.txt
python3 -m pytest tests -v          # 55 unit + integration tests
python3 run_experiment.py           # full battery (~10s), writes results/ + figures/
python3 run_experiment.py --quick   # smoke-test battery (~3s)
```

## Test plan

- [x] `python3 -m pytest tests -v` — 55 passed
- [x] `python3 run_experiment.py` — completes in ~10s, produces
      `results/topology_battery.csv`, `results/worst_case_convergence.csv`,
      `results/summary.json`, and 4 figures in `figures/`
- [x] `git status` confirms only
      `research-projects/price-of-anarchy-selfish-routing/` is touched — no
      changes to `src/`, `server/`, `package.json`, or any website code
- [x] No `__pycache__`/`.pyc`/venv artifacts committed (project-local
      `.gitignore`)
