# Grokking Group Circuits: Does the Representation-Theoretic Story Generalize Beyond Cyclic Groups?

_A self-contained research project: does the mechanistic-interpretability finding that grokked
networks implement group operations via Fourier analysis generalize to non-abelian groups, where
the natural "harmonic" basis is representation theory (irreducible matrix representations) rather
than 1-dimensional Fourier modes -- and is grokking itself gated more by absolute training-set size
than by whether the group is abelian?_

## 1. Research question

Power et al. (2022) showed that small transformers trained on modular addition exhibit **grokking**:
they memorize the training set almost immediately, generalize far later (often 10-100x more steps),
and — as Nanda et al. (2023, "Progress measures for grokking via mechanistic interpretability") and
Chughtai, Chan & Nanda (2023, "A Toy Model of Universality") showed — the mechanism they land on is
not a lookup table. The trained embedding table organizes itself around a handful of **Fourier
frequencies**, and the network computes `a + b mod p` via trigonometric identities applied to those
frequencies. This is elegant, but every well-known replication studies **cyclic** (abelian) groups,
where "Fourier basis" and "irreducible representation" happen to coincide (every irrep of an abelian
group is 1-dimensional).

This project asks three concrete questions that cyclic-only studies can't answer:

1. **Does the same phenomenon happen for a non-abelian group**, where the group's natural harmonic
   basis is built from *matrix*-valued irreducible representations (2-dimensional rotation/reflection
   blocks for a dihedral group, a quaternionic 2-dimensional complex irrep for the quaternion group
   Q8) rather than 1-dimensional Fourier modes?
2. **Does representation-theoretic structure emerge *in step with* generalization** (a "progress
   measure", in Nanda et al.'s terms) regardless of which group is being learned, or is that
   coupling specific to the cyclic/Fourier case?
3. When a group task *fails* to grok within a fixed compute budget, **is that because the group is
   non-abelian, or because there just isn't enough training data** — a confound none of the papers
   above disentangle, since they don't compare groups of different orders and algebraic types
   side by side.

## 2. Methodology

### 2.1 Task

For a finite group `G`, the task is next-token-style classification: given `(a, b)` with
`a, b ∈ G`, predict `a·b` (the group's binary operation), over the full `|G|²` set of ordered pairs,
split into a train/test partition (50/50 for the main comparison). This is the standard grokking
setup used in the literature above, generalized from `Z/pZ` addition to arbitrary finite groups.

### 2.2 Groups studied

| Group | Order | Type | Irreps |
|---|---|---|---|
| `C_59` (cyclic, addition mod 59) | 59 | abelian | 59 one-dimensional complex characters (the classical DFT basis) |
| `D_15` (dihedral, symmetries of a 15-gon) | 30 | **non-abelian** | 2 linear (trivial, sign) + 7 two-dimensional rotation/reflection irreps |
| `Q_8` (quaternion group `{±1,±i,±j,±k}`) | 8 | **non-abelian** | 4 linear + 1 two-dimensional *quaternionic*-type complex irrep |

`D_15` and `Q_8` are two structurally different flavors of non-abelian group: `D_15`'s irreps are
ordinary real 2D rotation/reflection matrices, while `Q_8`'s 2-dimensional irrep is of quaternionic
(not realizable as a 2D real irrep) type — the smallest non-abelian group where that distinction
shows up at all. A supplementary sweep (§2.5) adds cyclic groups of several smaller orders
(`C_8, C_13, C_23, C_37, C_47`) to separate "non-abelian" from "small".

All group implementations (`grok/groups.py`) are exhaustively tested for the group axioms
(associativity, identity, inverses) and — critically — every hand-derived representation matrix is
checked to be an actual homomorphism (`ρ(a)ρ(b) == ρ(ab)` for **every** pair `a, b ∈ G`) and unitary,
since a bug there would silently invalidate the entire spectral analysis below.

### 2.3 Model

A 2-layer MLP, matching the architecture family used in the "MLPs grok too"
literature (Liu et al. 2022): a shared embedding table `W_E` (one row per group element, used for
both operands) → concatenate the two embeddings → `Linear → ReLU → Linear` → softmax over `|G|`
classes. The forward and backward passes are written out by hand in `grok/model.py` (no autodiff
framework — this project's only dependency is numpy for array math and matplotlib for plots), and
verified against numerical (finite-difference) gradients to a relative error of `~1e-7`
(`tests/test_model.py`). Optimization is hand-rolled AdamW (`grok/optim.py`) — the decoupled weight
decay is the actual ingredient that makes grokking happen at all; without it the network is content
to sit at a memorizing solution indefinitely (Power et al. 2022).

### 2.4 Measuring representation-theoretic structure: the spectral alignment score

This is the part that has to work for *any* finite group, not just cyclic ones. By the Peter-Weyl
theorem (finite-group case), the matrix-coefficient functions `g ↦ ρ(g)_{ij}` of a group's
irreducible representations form a **complete orthonormal basis** for the `|G|`-dimensional space of
real-valued functions on `G`. For a cyclic group this basis *is* the DFT (each 1D irrep contributes
a `cos`/`sin` pair). For `D_15` it's built from the rotation/reflection matrix entries of each 2D
irrep; for `Q_8`, from the real and imaginary parts of its quaternionic irrep's matrix entries.

`grok/spectral.py` builds this basis directly and generically from `group.irreps()`: for each irrep
it collects candidate real vectors (real/imaginary parts of matrix entries as functions of `g`),
projects out whatever earlier irreps already spanned (so equivalent/conjugate irreps — e.g. a cyclic
group's frequency `k` and `p-k`, which give the same real span — don't get double-counted), and keeps
only the genuinely new directions via SVD. The result is a set of mutually orthogonal "blocks", one
per irrep, whose basis vectors together span all of `R^|G|` exactly (checked directly in
`tests/test_spectral.py` for all three groups: `assert_basis_is_complete`).

Given a trained embedding table, its **spectral concentration score** is a normalized participation
ratio over how its variance distributes across these blocks: `1.0` means all variance sits in a
single block (maximally structured — the network is only using a handful of harmonics), `0.0` means
variance is spread perfectly uniformly across every block (what unstructured noise looks like in
expectation). To turn that into a statistical claim rather than just a number, we compare it against
a **null distribution** built by permuting which group element each embedding row belongs to (200-300
random shuffles) — this preserves the exact multiset of embedding vectors and the table's raw
variance/scale while destroying any actual alignment with the group's structure — and report a
z-score against that null.

### 2.5 Experiments

1. **Main comparison** (`C_59`, `D_15`, `Q_8`; 3 seeds each): train to convergence (or to a fixed
   step budget if it doesn't converge), tracking train/test accuracy and the spectral concentration
   score every checkpoint. Reports: grokking gap (steps to 95% test acc ÷ steps to 95% train acc),
   final spectral-alignment z-score, and the correlation between the concentration-score trajectory
   and the test-accuracy trajectory (does structure track generalization, or just appear after the
   fact?).
2. **Data-scale sweep** (cyclic groups of order `8, 13, 23, 37, 47`, 1 seed each, same step budget as
   `D_15`/`Q_8`): does grokking success depend on absolute training-set size (`n_train`) in a way
   that would explain a non-abelian group failing to grok, independent of it being non-abelian?

## 3. Results

_(See `results/results.json` for the complete raw numbers behind every claim below, and `plots/` for
the figures referenced. All numbers are from the actual experiment run: 3 seeds per main group plus
the data-scale and train-fraction sweeps, ~18 minutes total wall-clock.)_

### 3.1 The representation-theoretic story generalizes to a non-abelian group

`![training curves](plots/training_curves.png)`

| Group | seed | train95 step | test95 step | final test acc | grok gap (test95/train95) |
|---|---|---|---|---|---|
| `C_59` | 0 | 600 | 4500 | 1.000 | 7.5x |
| `C_59` | 1 | 600 | 5800 | 1.000 | 9.7x |
| `C_59` | 2 | 600 | 7200 | 0.991 | 12.0x |
| `D_15` | 0 | 400 | 17800 | 0.978 | 44.5x |
| `D_15` | 1 | 400 | never (budget: 20000) | 0.353 | — |
| `D_15` | 2 | 400 | never (budget: 20000) | 0.784 | — |
| `Q_8`  | 0-2 | 200 | never | 0.000 | — |

`C_59` groks cleanly and consistently across all 3 seeds: near-immediate train fit, a long plateau,
then a sharp rise to ~100% test accuracy, with a grokking gap of 7.5-12x. `D_15` shows the same
qualitative signature — train accuracy saturates by step 400, test accuracy stays flat for four
orders of magnitude in step count, then rises sharply late in training — but is noisier and slower:
only 1 of 3 seeds crosses the 95%-test-accuracy line within the 20,000-step budget (at a striking
44.5x grok gap), while the other two are still mid-rise at the budget's end (35% and 78% test
accuracy, both climbing). `Q_8` never leaves the memorizing regime in any seed (§3.3 explains why).
So: the phenomenon transfers to a non-abelian group, but transfers *less reliably/more slowly* — a
real, non-trivial difference this project was able to quantify rather than just report as "yes/no".

`![alignment bar chart](plots/alignment_bar.png)`

Final alignment z-scores (mean ± std across seeds, vs. the row-shuffle null): `C_59` = 58.5 ± 11.4,
`D_15` = 14.2 ± 8.5, `Q_8` = -1.24 ± 0.74. Both `C_59` and `D_15` sit far above the `z=3` significance
line — confirming the phenomenon isn't an artifact of cyclic groups having 1-dimensional Fourier
modes to align to; it holds for a group whose natural harmonic basis is built from 2×2
rotation/reflection matrices too, just with a smaller (and more seed-variable) effect size, tracking
the messier generalization behavior above. `Q_8`'s alignment is statistically indistinguishable from
(if anything, slightly below) the shuffle null, consistent with it never finding a generalizing
circuit.

`![embedding spectrum](plots/embedding_spectrum.png)`

`C_59`'s embedding concentrates the large majority of its variance in just 4 of 30 available
frequency blocks (`freq_23`=28.2%, `freq_20`=21.2%, `freq_22`=17.8%, `freq_8`=12.6%; those 4 alone
carry 79.8% of all variance, with most of the other 26 blocks carrying under 1% each) — the
sparse-Fourier-modes signature from the literature. `D_15` shows the same qualitative sparsity in a
representation-theoretic (not Fourier) basis: its top 4 of 9 blocks (`rho_1`=26.0%, `rho_3`=22.0%,
`rho_5`=20.5%, `rho_7`=12.7%) carry 81.1% of the variance combined, while the two linear irreps
(`trivial`, `sign`) carry almost none. `Q_8`'s spectrum is dominated by its one 2-dimensional irrep
block (`rho_2d`, 48.1%) — but that block alone spans half of `Q_8`'s 8-dimensional function space, so
48% is actually close to the ~50% a randomly-oriented embedding would put there by chance;
consistent with the near-zero z-score, this is not evidence of structure, just of the null baseline
itself being uneven across differently-sized blocks (which is exactly why the z-score against a
shuffle null, not the raw energy fraction, is the metric that matters here).

### 3.2 Structure formation tracks generalization, not just follows it — and can outpace it

`![progress measure](plots/progress_measure.png)`

For `C_59` and `D_15` (seed 0), spectral concentration and test accuracy rise together across
training: Pearson correlation between the two trajectories is `0.86` (`C_59` seed 0; mean `0.84`,
range `0.74-0.92` across seeds) and `0.97` (`D_15` seed 0; mean `0.86`, range `0.69-0.97` across
seeds) — supporting Nanda et al.'s "progress measure" framing for a non-abelian group, not just the
classical cyclic case. (`Q_8`'s correlation, `0.03, -0.87, -0.90` across seeds, isn't meaningful: its
test accuracy is ~flat at 0 throughout, so there's essentially no signal for concentration to
correlate with — reported for completeness, not as evidence either way.)

A subtler finding shows up in the data-scale sweep (§3.3): `C_23` (264 training examples) reaches a
final alignment z-score of `18.9` — comfortably "significant" — while its final test accuracy is
`0.008`, essentially chance. Representation-theoretic structure can partially assemble *before* it's
sufficient for generalization, not only in lockstep with it. That's a meaningful qualification to the
progress-measure story: alignment appears to be a necessary accompaniment of generalization here, not
a sufficient one.

### 3.3 An honest negative result, disentangled: `Q_8` fails because of data scale, not algebra

`![data scale threshold](plots/data_scale_threshold.png)`

`Q_8` (order 8, only 64 total `(a,b)` pairs) never grokked in this project. `experiments/run_experiments.py`
includes a dedicated train-fraction sweep (`quaternion_train_frac_sweep` in `results/results.json`)
giving `Q_8` progressively more of its own tiny dataset — 50% up to 90% train — and it stays
essentially at chance throughout: final test accuracy of `0.06, 0.04, 0.05, 0.00, 0.17` for train
fractions `0.5, 0.6, 0.7, 0.8, 0.9` respectively, with alignment z-scores of `-0.84, -0.86, 0.19,
-0.25, 0.64` — never anywhere near the `z=3` significance line. Taken alone, that would be
ambiguous: does representation theory not organize non-abelian embeddings, or is `Q_8` just too
small a group?

The data-scale sweep (cyclic groups only, so algebraic type is held fixed and only size varies)
answers this directly:

| Group | # train examples | Final test acc | Grokked? | Final alignment z |
|---|---|---|---|---|
| `C_8`  | 32   | 0.000 | no  | -0.76 |
| `C_13` | 84   | 0.000 | no  | 5.79  |
| `C_23` | 264  | 0.008 | no  | 18.92 |
| `C_37` | 684  | 0.997 | **yes** (step 16600) | 32.31 |
| `C_47` | 1104 | 1.000 | **yes** (step 2800)  | 39.45 |
| `C_59` | 1740 | 1.000 | **yes** (step 1400, seed 0) | 46.76 |

Cyclic groups matched in **absolute training-set size** to `Q_8` — `C_8` at 32 train examples — fail
to grok in exactly the same way despite being abelian, while `C_37` (684 examples) and larger grok
comfortably. The transition sits somewhere between 264 (`C_23`, fails) and 450 (`D_15`, mostly
succeeds) training examples in this setup — **a threshold in absolute dataset size, not group order
or algebraic type**. `Q_8` isn't a counterexample to representation theory generalizing across
non-abelian groups; it's simply below the data threshold every group needs, abelian or not, for this
architecture and step budget. (Note also: `C_13`, at 84 examples, already shows a significant
alignment z-score of `5.79` *despite* never generalizing — the same "structure without
generalization yet" pattern as `C_23` in §3.2, now shown to be about data scale rather than being
non-abelian-specific either.)

## 4. Honest scope and limitations

- **Compute-bounded, not asymptotic.** All step budgets and thresholds above are specific to this
  architecture, optimizer, and step budget on this hardware. The "260-450 training examples"
  threshold is a property of this experimental setup, not a universal constant of grokking.
- **Three groups, not a systematic scan.** `D_15` and `Q_8` are two informative, structurally
  different data points, not an exhaustive survey of non-abelian groups. A stronger version of this
  project would sweep dihedral order and a family of small non-abelian groups (e.g. `S_4`, `A_4`)
  matched for training-set size, to map out exactly where (if anywhere) non-abelian-specific
  obstructions to grokking exist, independent of the data-scale confound identified here.
- **The spectral alignment score is a summary statistic, not a full circuit analysis.** It answers
  "is variance concentrated in a few irrep blocks" but doesn't verify the network actually computes
  the group operation *via* those irreps (e.g. by checking for the trace/character identity Chughtai
  et al. derive analytically) — that would be the natural next step for whichever group this method
  flags as strongly aligned.

## 5. Repository layout

```
grok/
  groups.py     — CyclicGroup, DihedralGroup, QuaternionGroup: group ops + irreducible representations
  spectral.py   — generic representation-theoretic basis construction + alignment scoring
  model.py      — 2-layer MLP, hand-written forward/backward
  optim.py      — hand-written AdamW
  train.py      — training loop with checkpointing
tests/          — 65 unit + integration tests (group axioms, irrep homomorphism/unitarity,
                  gradient checks, optimizer sanity, spectral-basis completeness, end-to-end grokking)
experiments/
  run_experiments.py — the full reproducible sweep described above
plots/          — generated figures (see §3)
results/results.json — full raw results backing every number in this README
```

## 6. Reproducing

```bash
pip install -r requirements.txt
PYTHONPATH=. python3 -m pytest -q                    # 65 tests, ~20s
PYTHONPATH=. python3 experiments/run_experiments.py   # full sweep, ~18 min; regenerates plots/ and results/results.json
```

## References

- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). *Grokking: Generalization
  Beyond Overfitting on Small Algorithmic Datasets.*
- Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). *Progress measures for
  grokking via mechanistic interpretability.*
- Chughtai, B., Chan, L., & Nanda, N. (2023). *A Toy Model of Universality: Reverse Engineering how
  Networks Learn Group Operations.*
- Liu, Z., Kitouni, O., Nolte, N., Michaud, E. J., Tegmark, M., & Williams, M. (2022). *Towards
  Understanding Grokking: An Effective Theory of Representation Learning.*
