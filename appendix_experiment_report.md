# The appendix experiment: the §6.1 follower at the individual level

> **Note (2026-10-05):** scripts and output folders not used for the paper were moved to `further_experiments/` (gitignored); folder names below refer to their former location in `scripts/synthetic/`. The E.5 results now in the paper are in `scripts/synthetic/outputs_appendix_anneal_{standard,lingauss}`.

Report for Linus, 4 October 2026. This documents the final form of the experiment proposed for
the appendix, the text changes it needs in §2.1 and §E.1, the training recipe of the neural
follower, and the diagnostics that stay out of the paper. It supersedes Section 8 of
`individual_follower_report.md`; the earlier report is kept for the history (Experiments 1 and
2, the `X4`-only variant, the first pass of this experiment). All numbers are the leader's
deployment MSE; lower is better for the leader.

## 1. What the experiment shows, and how to frame it

§2.1 is an identity between two *optimal* followers: a population of individuals who each
observe `x_O` and best-respond, and the population follower with `A_j = O \ PA(j)`. If both
chose perfectly they would choose the same shift for every person and the curves would
coincide exactly. An experiment therefore cannot "check §2.1"; it compares two *computed*
approximations of one follower. The individuals are computed almost exactly (grid search over
13 shifts, expectations by Monte Carlo); the population follower is a small network trained by
Adam, and nothing guarantees that it ends at the best rule. The gap is thus near-exact search
versus trained network, it has one sign (the individuals are never weaker), and it measures the
network's distance from the best response.

That is the framing to use in the paper: the experiment *certifies the neural follower of
§6.1*. It shows that the conclusions of Fig. 2 do not depend on how well the attack was
optimized, and that the all-variable curve is a lower bound on the harm. It also pre-empts two
referee questions. "Is your neural attacker any good?": yes, to within a few percent, and here
is the gap. "Why a network at all, if search is feasible?": the network is the generic approach
for arbitrary targets and SCMs; search is feasible here only because each step is a
one-dimensional choice in a cheap, known SCM.

Do not write "we validate §2.1 empirically", and do not lean on the warm start (Section 6) in
the main text: it uses the answer to find the answer, and a reader will ask why Fig. 2 is not
warm-started. The paper's training recipe stays as it is; the gap is reported as the result,
and a footnote says what mostly causes it and what reduces it (Section 4).

## 2. The follower

**Why sequential.** The targets are `X1` and `X4` with `X1 → Y → X4`. The common
non-descendants of the targets are `{X2}`, so a one-shot individual as in §2.1 would have
`O ⊆ {X2}` and the assumption `PA(j) ⊆ O ∪ CH(E)` fails for `X4`, whose parent `Y` is neither
observed nor a target. No choice of `O` and no small change of the DAG avoids this while one
target is above `Y` and one below. The individual sets the targets one after the other.

**The individual.** One individual is one draw of all exogenous noises.

1. It observes `x2` and the value `X1` would take without intervention (i.e. `ε1`), and
   chooses `δ1 ∈ [−b, b]`.
2. Nature draws `Y` and `X3`. It observes `y` and the value `X4` would take without
   intervention (`ε4`), recalls `x1, x2`, and chooses `δ4 ∈ [−b, b]`. It does not see `x3` or
   `ε5`; their law given `y` is fixed by the SCM.

Solved backwards: step 2 gives `g4(x1, x2, y, ε4)`, the shift that maximizes the expected
objective for the given history; step 1 gives `g1(x2, ε1)`, maximizing the expected objective
averaged over the future with `g4` applied in each future. Recall of `x1` makes step 2
self-contained: without it the individual would need the law of `X1` given `(y, x2)`, which
depends on its own step-1 rule, and the problem would be a two-agent team problem rather than
one individual. `X3` can be left out because its only parent `Y` is observed (a modeling choice,
not a necessity); `ε1` is not needed in step 2 because it only affected `x1`.

**The appendix follower: `X1` added to `δ4`.** A population follower choosing `g1(X2, ε1)` and
`g4(X1, X2, Y, ε4)` to maximize the average objective must be optimal for almost every history
in step 2 and, given that, for almost every `(x2, ε1)` in step 1; these are the individuals'
two conditions, so both use the same rules and give the leader the same risk. This is the
follower of §2 with `A1 = ∅`, `A4 = {X1}`. The paper's follower lets `δ4` see `(Y, X2, ε4)`
only, which is the optimum over a smaller class, so "coincide up to approximation error" would
be false for it (the gap is below 0.01, but it is not zero by construction). In the appendix
experiment the population follower therefore gets `X1` as an additional input of `δ4`; Fig. 2
and Fig. 5 keep the paper's follower, and the appendix states that the addition changes none of
their curves, with the numbers (Section 5: `PA(Y)` and `SB(Y)` are identical because these
predictors ignore `X4`, and the all-variable curve moves by less than the run-to-run noise).
Lemma 4 covers the new edge `X1 → X4` (`X1` is not a forbidden descendant of `Y` and the edge
creates no cycle), so the stable blanket is unchanged; Fig. 5 (right) already does the same with
`{X1, X3}`.

**Keep equivalence and invariance apart.** The equivalence of §2.1 holds for any observation
sets. Whether the induced population follower keeps the stable blanket is Lemma 4's question:
a sequential individual who observes `y` before setting a target that is *not* a child of `Y`
adds an edge `Y → X_j`, which can shrink the stable blanket, and Lemma 4 excludes exactly that.
The remark in §2.1 should state the equivalence and then only note that §6.1 has
`A4 = {X1}`, which satisfies Lemma 4.

## 3. Text changes for the paper

**§2.1, remark after the one-shot statement** (three sentences; the alternative is the full
replacement in the plan of 4 October, which has the same content):

> *Remark (sequential targets).* If the targets in `CH(E)` are set one after another in a
> topological order of `G_0`, the same argument applies step by step: before setting `X_j` the
> individual observes `x_{O_j}` and the noise `ε_j` of `X_j`'s own assignment (the value `X_j`
> would take without intervention), where `O_j` consists of variables preceding `X_j` in that
> order and contains `PA(j)`, the targets set earlier and their observation sets, and chooses
> `c_j` from a set that may depend on `(x_{O_j}, ε_j)`;
> going through the order backwards, the pointwise optimal choices define functions
> `g_j(x_{O_j}, ε_j)`, and a population of such individuals is a population follower with
> `A_j := O_j \ PA(j)` and `ε_j` as the independent noise of the new assignment. The experiment
> of §6.1 is of this kind: `X1` is set after observing `(x2, ε1)` and `X4` after observing
> `(x1, x2, y, ε4)`, so `A1 = ∅` and `A4 = {X1}`, which satisfies the condition of Lemma 4.

Both the one-shot statement and the remark need the clause that the individual observes the
noise of the target's own assignment and may choose from a per-individual constraint set
(the bounded shifts of §6.1); the current text has neither, and a reader comparing §2.1 with
§6.1 can notice both.

**§E.1 and §E.4: unchanged.** Fig. 2 and both panels of Fig. 5 (the train-size sweep and the
`X1, X3` variant) were produced with the paper's follower and recipe; changing either would
mean regenerating all three for consistency, in exchange for an appendix gap that the
conclusions do not depend on. So §E.1 keeps describing the follower without `X1`, and the
appendix text itself says that its follower additionally sees `X1` and that this changes
nothing in Fig. 2. (If the augmented follower is adopted later, the sentence for §E.1 is: "The
intervention on `X4` may depend on `X1` in addition to its parents `(Y, X2)` and `ε4`, so that
`A1 = ∅` and `A4 = {X1}` in the notation of §2; by Lemma 4 the stable blanket is unchanged.")

**Appendix text.** The current version is in `prediction_intervention_games/body.tex`,
subsection "Individual followers" (`app:individual_followers`), marked `\felix{}` together with
the other insertions (the §2.1 clause and `Remark` environment on sequential targets, a pointer
in §6.1, the appendix intro, one discussion sentence). It is structured as an introduction and three `\paragraph`s: the introduction (what
distinguishes the two followers, the aim of verifying empirically that the population follower
captures the individual followers of §2.1, and where differences can come from); the individual followers (two choices, grid of 13, 16 and 64 Monte Carlo draws);
why the population follower's `X4` intervention gets `X1` here (an individual who set `x1`
knows it; by §2.1 the two followers coincide only with the same information; this changes the
§6.1 results by less than the between-run variability, numbers given); and the results, with
the footnote on the optimization effect and the continuation restart. The paper's own terms
are used throughout: "population follower", "individual followers", "deployment MSE",
"follower's objective"; no "best response", "harm" or "bang-bang". The statement on the
individuals' `X4` shift reads: "the individuals' optimal shift of `X4` always lies at the
bound, `δ4 = ±b`, with the sign depending on whether the expected residual `Y − f_S(X_S)`
given the individual's observations is positive or negative."

The figure (`images/appendix_individual_followers.pdf`, script
`plot_appendix_individual_followers.py`) follows the layout and font sizes of Fig. 2: one row
of four panels, linear-Gaussian SCM left, nonlinear SCM right, both objectives per SCM, with
the caption stating the main observation.

### 3b. Main text reduced to pointers (2026-10-05)

Following Linus's request to keep the main text minimal (and the second review), the main-text
additions were cut from about 260 words to two pointer sentences:

- **§2.1:** the noise clause and the sequential-targets remark were moved out. One sentence
  remains after the assumption `PA(j) ⊆ O ∪ CH(E)`: "App. B.1 extends the argument to followers
  who observe the noise of the targets' own assignments and to targets that are set one after
  another, as in §6.1."
- **Appendix B, new first subsection** "Individual followers with observed noise or sequential
  targets" (`app:individual_followers_theory`), ahead of "Subset-level formulation": one
  lead sentence and two paragraphs, "Observed noise" and "Sequential targets" (no remark
  environment; the label `rem:sequential_targets` is gone and E.5 now refers to the subsection).
  The sequential paragraph ends with the §6.1 instance, `A_4 = {X_1}`.
- **Second review round:** the §2.1 pointer now sits at the end of the subsection (after the
  augmented-graph sentence); the observed-noise paragraph ends with the clause that the target's own
  noise serves as the exogenous noise of its new assignment in the population follower; E.5 says
  "over the noise it does not observe" and "the population follower of §6.1". The reviewer's
  "0.002 is wrong at b = 1.5" refers to the old figure range; the figure stops at b = 1, so 0.002
  stands.
- **§6.1:** one sentence: "App. E.5 repeats this experiment with individual followers as in §2.1,
  who each choose their intervention from their own observations, in place of the population
  follower; consistent with §2.1, the leader's deployment MSE is nearly the same."
- **Discussion:** the sentence on individuals who observe their own outcome noise was dropped.
- **E.5:** the figure now stops at `b = 1` like Fig. 2 (`plot_appendix_individual_followers.py
  --max-bound 1.0`); the caption states the observation qualitatively, the percentages appear
  only in the Results paragraph; the X1-effect sentence reads "differ by at most 0.002 (10 runs)",
  which holds for both SCMs at `b ≤ 1` (nonlinear 0.0021, linear-Gaussian 0.0014; at `b = 1.5` it
  would be 0.0055 and 0.0023).

## 4. The neural follower's training: the gap and what reduces it

**The paper's recipe** (§E.4): two hidden layers of 64 units, output `b · tanh(h)`, Adam with
learning rate 0.001 for 2000 steps on fresh batches of 512 draws, five random restarts, the best
by held-out value. Against the individuals it leaves a gap of 5 to 10 % at `b ≥ 1` for the
all-variable predictor under the MSE objective, with a bimodal pattern across runs: in about one
run in five the best of the five restarts sits at a poor local maximum (0.28 instead of 0.35 at
`b = 1`; 0.63 instead of 0.70 at `b = 1.5`).

**Why there is a gap at all.** Two sources. Optimization: the payoff is a non-convex function of
the weights and Adam stops in local maxima; the bimodality is this. Structure: under the MSE
objective the squared error is close to convex in `δ4`, so the best shift is an endpoint, `+b`
or `−b`, with the side given by the sign of the expected residual; the optimal rule is a step
function of the inputs (the individuals put `δ4` at the bound for essentially everyone), and a
tanh output cannot jump: it interpolates across the boundary, and every person on the wrong side
or short of the bound costs payoff. The boundary is a hyperplane in the linear-Gaussian SCM and
a curved surface in the nonlinear SCM, which is why the part that training cannot remove is near
zero in the first and 1 to 2 % in the second (Section 6).

**Self-contained improvements**, no warm start, tested on three predictor sets (runs 0 to 2,
two of which have stuck runs) against the individuals on the same noise draws
(`tune_attack_training.py`, `outputs_attack_tuning/`; all-variable predictor, MSE objective;
gap = individuals − network, mean over the three runs and worst run):

| recipe | b = 1: gap (worst run) | b = 1.5: gap (worst run) | cost per bound |
|---|---|---|---|
| paper: 5 restarts × 2000 steps | 0.041 (0.091) | 0.086 (0.131) | 1 |
| 20 restarts | 0.012 (0.014) | 0.057 (0.102) | 4 |
| 5 × 5000 steps, cosine-annealed lr | 0.036 (0.091) | 0.083 (0.130) | 2.5 |
| output `b · tanh(3h)` | 0.066 (0.094) | 0.106 (0.133) | 1 |
| 3 hidden layers of 256 | 0.014 (0.018) | 0.041 (0.053) | 2 |
| **continuation in b**: paper + 1 start from the previous bound's network | **0.010 (0.014)** | **0.023 (0.035)** | **1.2** |
| continuation + 5 × 5000 steps, cosine lr | 0.007 (0.013) | 0.017 (0.027) | 3 |

Individuals: 0.371 at `b = 1`, 0.746 at `b = 1.5` (mean over the three runs).

**Reading.** Continuation in `b` is the one change that matters. It is legitimate because the
side of the optimal step does not depend on `b` (exactly in the linear-Gaussian SCM, where
`δ4* = b · sign(E[residual | inputs])`; approximately in the nonlinear SCM), so the network
trained for the previous bound, whose output `b · tanh(h)` rescales automatically, is a good
start for the next one. It removes the stuck runs (run 1 at `b = 1`: 0.276 → 0.354; runs 1 and 2
at `b = 1.5`: 0.636 and 0.626 → 0.702 and 0.731), costs one extra training per bound, and is one
clause in §E.4. Cosine annealing with 5000 steps on top gains another 0.003 to 0.006 at three
times the cost. A steeper output hurts (saturation, vanishing gradients). More restarts or
longer training with a constant learning rate do not rescue the stuck runs: for some predictor
sets random initialization falls into the poor basin every time, and only a start from a related
solved problem avoids it. A wider network helps less than continuation at twice the cost.

**Decision: keep the paper's recipe.** Adopting continuation would make §E.4 describe a new
recipe, and then Fig. 2 and both panels of Fig. 5 would have to be regenerated for consistency,
in exchange for moving an appendix gap from 5 to 10 % to 2 to 3 % that the conclusions do not
depend on. The gap is a result, not a defect: the neural attack in Fig. 2 is a lower bound on
the harm, and `PA(Y)`, `SB(Y)` do not depend on the attack at all. The appendix figure shows the
paper-recipe network against the individuals; the footnote names the one change that mattered
and quotes the ten-run numbers of the continuation network, which the final run computes
alongside (Section 5). If the tighter number is wanted later, the recipe is one clause in §E.4
("for each bound we additionally start from the network trained for the next smaller bound,
whose output rescales with the bound, and keep the best of the six starts by held-out value")
and the code supports it (`run_experiments.py --x4-uses-x1 --attack-continuation`).

### 4b. Follow-up (2026-10-05): can a recipe change remove the gap without continuation?

Question asked: can the network be changed (size, dropout, training procedure) so that it does
not get stuck and the gap needs no justification? Budget 45 minutes, no full rerun. All tests use
the stored predictors and noise draws of the final run (`outputs_appendix_final_standard`), the
all-variable predictor, the MSE objective, ten runs, `b ∈ {1, 1.5}`; reference values are the
stored individuals (`ind_nox3`). Script: `tune_attack_training.py --reference ... --cache-dir ...`
(new variants listed in its docstring); outputs `outputs_attack_tuning2/` and `outputs_attack_tuning3/`.

**What the stuck runs are.** Tracing one collapsing run (run 1, `b = 1.5`) step by step: at step 0
the shifts are small and mixed in sign; by step 100 both networks output `+b` (or `−b`) for every
input (`δ4` mean exactly `±b`, 100 % at the bound), and from then on the objective only creeps
(0.50 → 0.53 over the remaining 1900 steps). The stored stuck runs show the same signature
(`mean_delta_x4 = −1.000` at `b = 1`, `−1.500` at `b = 1.5`, `frac_bound_x4 = 1.000`). The reason
is the starting point, not the step size: with `δ ≈ 0` for everyone, a common shift in the
direction of the mean gradient pushes the individuals with small residual past the point where
the prediction equals the outcome, after which their pointwise gradient also points in the
common direction, and the whole population ends on one side. Lowering the learning rate to
`1e-4` (6000 steps) or adding a 1000-step warmup does not change this (all five restarts still
collapse on run 1). This is why continuation from a smaller bound works: at small `b` the common
shift cannot cross the boundary, the sign pattern is learned, and it is kept as `b` grows.

**Variants, relative gap (individuals − network) / individuals, mean and worst of ten runs:**

| recipe (changes to §E.4) | b = 1: mean (worst) | b = 1.5: mean (worst) |
|---|---|---|
| paper | 7.2 % (24.9 %) | 9.5 % (17.3 %) |
| dropout 0.1 / 0.3 in the hidden layers | 9.2 % (25 %) / 10.1 % (25 %) | 10.4 % (18 %) / 12.6 % (18 %) |
| AdamW, weight decay 0.01 | 7.0 % (24.9 %) | 9.8 % (17.3 %) |
| softsign output `b·h/(1+|h|)` | 8.0 % (10.1 %) | 12.4 % (18.5 %) |
| output `b·tanh(0.3h)` / `b·tanh(0.1h)` | 5.2 % (7.5 %) / 6.2 % (8.2 %) | 7.9 % (16.9 %) / 7.4 % (8.2 %) |
| output layer initialised at zero | 4.6 % (7.2 %) | 7.0 % (16.4 %) |
| **bound annealed from 0 to b over the first 1000 of 2000 steps** | **5.1 % (9.1 %)** | **4.7 % (7.3 %)** |
| continuation in b (Section 4, ten runs of the final run) | 2 to 3 % | 2 to 5 % |

Dropout, weight decay and softsign do nothing or hurt. A flatter output slope or a zero-initialised
output layer reduce the number of collapsed runs but do not remove them (slope 0.3 still collapses
run 1, zero-init still collapses run 4). Annealing the bound within a single training run is the
self-contained version of continuation and removes the collapse in every run and every restart
(all restarts end with mixed-sign `δ4`); the worst run is then 7 % instead of 17 to 25 %. On the
four worst runs it was also combined with 5000 steps, cosine-annealed learning rate, and width 256:
no further gain (width 256 even re-collapses one run). The option is implemented as
`optimize_attack(..., bound_anneal_steps=1000)` but not wired into `run_experiments.py`.

**The residual 3 to 5 % at `b = 1.5` is not a local-maximum effect.** It is present in every run,
including the runs that never collapse, and does not shrink with longer training, cosine schedule,
or width. Swapping one network choice at a time for the individuals' choice on the same noise draws
(run 1, `b = 1.5`, annealed network 0.706, individuals 0.737): network `δ1` with the grid-optimal
`δ4` gives 0.720; individuals' `δ1` with the network's `δ4` rule gives 0.722. So about half the
residual is in each network. The `δ4` network puts 11 % of the individuals on the wrong side of
the boundary (sign disagreement with the grid-optimal choice given the same information), and its
`δ1` is larger and only moderately correlated with the individuals' (mean |δ1| 1.14 vs 0.96,
correlation 0.57). Run 4 gives the same picture (0.700 → 0.714 and 0.719; individuals 0.736).

**Conclusion.** The collapse can be removed by a one-line change to the recipe (bound annealing,
or continuation), which brings the worst run from 17 to 25 % down to about 7 % and the typical
run to 3 to 5 % at `b = 1.5` (2 to 3 % at `b = 1` with continuation). No tested change removes the
remaining few per cent, which is split between the two networks and is an approximation error of
gradient-trained 2 × 64 networks against pointwise grid search. So a sentence explaining a gap is
needed either way; what the recipe change buys is a smaller and uniform gap (no bimodality, nothing
about local maxima to explain) at the price of changing §E.4 and regenerating Fig. 2 and 5.

**The fix in one sentence.** During the first 1000 of the 2000 training steps, the bound is
increased linearly from 0 to `b` (the networks' output is `b_t · tanh(h)` with `b_t = b · min(1, t/1000)`);
nothing else changes. Code: `optimize_attack(..., bound_anneal_steps=1000)`;
`run_experiments.py --attack-bound-anneal-steps 1000`; `run_individual_follower.py --attack-bound-anneal-steps 1000`.

**Would the paper's existing experiments change?** Partial replay (seeds 0 to 2 of the ten; the
paper recipe reproduces the paper's own runs for these seeds) of the four paper configurations
with and without annealing, attacked test MSE
(`outputs_attack_tuning3/paper_configs_paper_vs_anneal_*`, `replay_paper_configs.py`):

- Parents and stable blanket: unchanged (largest difference 0.003, within the run-to-run noise),
  in both SCMs and both objectives.
- Prediction objective: unchanged to within 0.001 for every predictor, bound and SCM, including
  the train-size sweep (1000, 4000) and the `X4 uses X1, X3` configuration. Fig. 5 (both panels
  use the prediction objective) and the prediction panels of Fig. 2 would not change.
- All-variable predictor, MSE objective: the only curves that move. Runs that did not collapse
  under the paper recipe differ by at most 0.013 either way; collapsed runs move up to the level
  of the others (nonlinear SCM: seed 1 at `b = 1` 0.278 → 0.352, seeds 1 and 2 at `b = 1.5`
  0.636 and 0.620 → 0.702 and 0.705; linear-Gaussian SCM: seed 1 at `b = 1` 0.344 → 0.453,
  seed 2 at `b = 1.5` 0.758 → 0.917). In the ten-run figure this raises the all-variable MSE
  curve at `b ≥ 1` by roughly the collapse frequency times the jump (about 0.01 at `b = 1` and
  0.02 to 0.03 at `b = 1.5`) and narrows its confidence band. No qualitative change.
- Appendix E.5: the gap to the individuals becomes uniform, 5.1 % (worst 9.1 %) at `b = 1` and
  4.7 % (worst 7.3 %) at `b = 1.5` for the all-variable predictor under the MSE objective; all
  other cells are unaffected (see above).

## 5. Results

Both SCMs, 10 runs, 4000 individuals per run; three followers on identical noise draws: the
paper's follower with `X1` added to `δ4` and the recipe of §E.4 (`nn_x1`), the same network
with the continuation start (`nn_x1_tuned`, for the footnote only), and the individuals
(`ind_nox3`). Outputs: `outputs_appendix_final_{standard,lingauss}/`.

**Where the followers coincide.** Under the prediction objective all three followers agree to
within 0.0004 for every predictor and bound in both SCMs (the best response is the constant
shift `−b`). For `PA(Y)` and `SB(Y)` under the MSE objective they agree within 0.003 in the
nonlinear SCM, with one exception: `PA(Y)` at `b = 1.5`, where the paper-recipe network gives
0.276 against 0.284 for the individuals and for the continuation network (the same
local-maximum effect as below, in the `δ1` problem). In the linear-Gaussian SCM `PA(Y)` is
0.1595 and `SB(Y)` is 0.0834 at every bound under every follower, exactly.

**Where they differ: the all-variable predictor under the MSE objective.**

Nonlinear SCM:

| b | none | network, recipe of §E.4 | network + continuation start | individuals |
|---|---|---|---|---|
| 0.25 | 0.012 | 0.047 | 0.047 | 0.048 |
| 0.5 | 0.012 | 0.117 | 0.117 | 0.120 |
| 1.0 | 0.012 | 0.345 | 0.362 | 0.371 |
| 1.5 | 0.012 | 0.681 | 0.727 | 0.752 |

Linear-Gaussian SCM:

| b | none | network, recipe of §E.4 | network + continuation start | individuals |
|---|---|---|---|---|
| 0.25 | 0.017 | 0.065 | 0.065 | 0.065 |
| 0.5 | 0.017 | 0.154 | 0.154 | 0.155 |
| 1.0 | 0.017 | 0.431 | 0.456 | 0.458 |
| 1.5 | 0.017 | 0.882 | 0.921 | 0.925 |

Paired differences over the 10 runs (mean ± s.e.m., and in percent of the individuals' value):

| difference | SCM | b = 1 | b = 1.5 |
|---|---|---|---|
| individuals − network (§E.4 recipe) | nonlinear | +0.027 ± 0.008 (7 %) | +0.071 ± 0.011 (10 %) |
| individuals − network (§E.4 recipe) | linear-Gaussian | +0.027 ± 0.014 (6 %) | +0.043 ± 0.021 (5 %) |
| individuals − network with continuation start | nonlinear | +0.009 ± 0.001 (2.5 %) | +0.026 ± 0.004 (3.4 %) |
| individuals − network with continuation start | linear-Gaussian | +0.002 ± 0.000 (0.4 %) | +0.004 ± 0.001 (0.4 %) |

At `b ≤ 0.5` all differences are below 0.004.

**Per run.** The §E.4 network is bimodal. Nonlinear SCM, `b = 1`: 0.34 to 0.36 in nine runs
and 0.276 in one; `b = 1.5`: 0.69 to 0.72 in seven runs and 0.63 to 0.65 in three.
Linear-Gaussian SCM: two runs in ten sit at 0.345 instead of 0.45 at `b = 1` and at 0.76
instead of 0.91 at `b = 1.5`. The individuals vary only between 0.363 and 0.380 (nonlinear,
`b = 1`) and between 0.455 and 0.460 (linear-Gaussian). With the continuation start no run is
stuck: 0.354 to 0.373 and 0.452 to 0.458. This is the basis of the footnote's "about one run
in five" and of its mechanism claim.

**Shifts at the bound.** `δ4` is at `±b` for 100 % of the individuals and for 97 to 99 % of
the network outputs, in both SCMs and at every bound: the best response to the all-variable
predictor is a bang-bang rule on `X4` (the optional sentence of Section 3). The `δ1` fractions
differ between individuals and networks without a matching difference in payoff (Section 6).

**Figures.** `outputs_appendix_final_standard/appendix_figure.png` is the one for the paper:
the §E.4 network against the individuals, one row per SCM, one panel per objective, 95 %
bands over runs. `appendix_figure_with_continuation.png` adds the continuation network and is
for us, not for the paper.

![The appendix figure: the paper's follower with `X1` added to `δ4` (solid, filled markers) against the individual followers with the same information (dashed, open markers), on identical noise draws. Rows: nonlinear and linear-Gaussian SCM; columns: the two follower objectives; bands: 95 % confidence intervals over 10 runs.](further_experiments/outputs_appendix_final_standard/appendix_figure.png)

![Diagnostic version with the continuation network (dash-dotted, squares).](further_experiments/outputs_appendix_final_standard/appendix_figure_with_continuation.png)

**What goes into the text.** The gap in the appendix sentence: "by 5 to 10 % at `b ≥ 1`"
(7 and 10 % nonlinear, 6 and 5 % linear-Gaussian). The footnote: "reduces the difference to
2 to 3 % in the nonlinear and below 1 % in the linear-Gaussian SCM". The `X1` clause: `PA(Y)`,
`SB(Y)` identical, all-variable curve within 0.005 (nonlinear) and 0.014 (linear-Gaussian)
of the paper's follower, from `outputs_appendix_sequential_{standard,lingauss}` (10 runs,
Section 6 of the earlier report).

## 5b. Final results with the annealed recipe (2026-10-05)

Decision (Linus): use the annealed recipe for the appendix experiment, remove the footnote, and
leave a purple todo in the tex: if the recipe is adopted, update §E.4 and regenerate Fig. 2 and 5.
The E.5 networks were rerun with `--attack-bound-anneal-steps 1000` (both SCMs, 10 runs, followers
`nn_standard` and `nn_x1`; outputs `outputs_appendix_anneal_{standard,lingauss}`). The individuals
(`ind_nox3`) are deterministic given predictor, noise draws and seed, which are identical to the
previous run (checked: clean test MSEs agree to the bit), so their rows were copied from
`outputs_appendix_final_*` and the summaries rebuilt with `--plot-only`. The paper figure
`images/appendix_individual_followers.pdf` was regenerated from these folders.

**All-variable predictor, MSE objective, deployment MSE (mean over 10 runs):**

| b | nonlinear: individuals | population (annealed) | gap | linear-Gaussian: individuals | population (annealed) | gap |
|---|---|---|---|---|---|---|
| 0.25 | 0.0479 | 0.0467 | 2.5 % | 0.0651 | 0.0649 | 0.2 % |
| 0.5 | 0.1201 | 0.1143 | 5.1 % | 0.1548 | 0.1541 | 0.4 % |
| 1.0 | 0.3714 | 0.3524 | 5.4 % | 0.4576 | 0.4556 | 0.4 % |
| 1.5 | 0.7523 | 0.7174 | 4.9 % | 0.9249 | 0.9201 | 0.5 % |

Per run, the nonlinear gap ranges from 2 to 10 % (worst: run 9 at `b = 1`, 10.0 %; run 0 at
`b = 1.5`, 7.8 %), with no collapsed run (previously 1 of 10 at `b = 1` and 3 of 10 at `b = 1.5`
with gaps of 14 to 25 %). The linear-Gaussian gap is below 1 % in every run and cell. For
`PA(Y)`, `SB(Y)` and the prediction objective the two followers agree within noise (nonlinear
`PA`/`SB` under MSE: −1.3 to +1.2 %; prediction objective: < 0.1 %).

**Effect of adding X1 to δ4 under the annealed recipe** (`nn_x1 − nn_standard`, mean over runs):
`PA(Y)`, `SB(Y)` identical; all-variable at most 0.0055 (nonlinear) and 0.0023 (linear-Gaussian),
below the between-run standard deviation of the network's result (0.020 and 0.003 at `b = 1.5`).
These replace the previous numbers (0.005, 0.014) in the tex.

**One trade-off worth knowing.** At small bounds the annealed network is slightly worse than the
paper recipe in the nonlinear SCM (`b = 0.5`: 0.1143 vs 0.1164; `b = 0.25`: 0.0467 vs 0.0471),
because half of the 2000 steps are spent at a smaller bound; this is why the gap at `b = 0.5`
rose from 3 % to 5 %. A shorter ramp (for example 500 steps) or 3000 steps in total would likely
remove this; not tested.

**Tex changes (all `\felix{}`):** footnote removed; new paragraph "Training of the population
follower" stating the ramp, followed by the todo
`\felix{[TODO: if we adopt this training recipe, use it for all experiments: update E.4, regenerate Fig. 2 and 5 with it, and delete this paragraph. ...]}`;
numbers in "Matching the information" (0.006 / 0.002), in "Results" and in the caption
("below 1 % in the linear-Gaussian SCM and 2 to 5 % in the nonlinear SCM"). Compiles without errors.

## 6. Diagnostics kept out of the paper

These were run with the paper's recipe (`outputs_appendix_sequential_{standard,lingauss}`,
10 runs, and single-run checks); they explain the gap but do not belong in the main text.

- **Warm start at the individuals' rules** (diagnostic, not a follower). Fit the two networks by
  least squares to the 4000 pairs (inputs, chosen shift) of the individuals, then run the usual
  training from there (2000 steps, one start). All-variable predictor, MSE objective, `b = 1` and
  `1.5`: nonlinear 0.364 and 0.732 against 0.371 and 0.752 for the individuals (2 to 3 %), with
  the fitted networks alone at 0.363 and 0.728; linear-Gaussian 0.455 and 0.920 against 0.458
  and 0.925 (below 1 %), fitted alone 0.455 and 0.918. So most of the paper recipe's gap is a
  bad basin, and what a start in the right basin cannot remove is the continuous network's
  approximation of a step rule, larger where the boundary is curved. The least-squares fit is
  not the payoff-optimal projection onto the network class (near the boundary squared errors
  are large but payoff differences small), so the fit-only numbers overstate the capacity
  limit. Not used in the paper; the footnote on the continuation start (Section 3) takes its place.
- **Width and training length** (run 0, nonlinear). Cold start with 5 × 10 000 steps: 0.367 and
  0.724 against 0.370 and 0.741; width 256 cold: 0.299 and 0.660 (worse, harder optimization);
  warm start with width 64 or 256 and 2000 or 10 000 steps: 0.361 to 0.368 and 0.715 to 0.725.
  At `b = 1` everything trained long enough reaches the individuals within 1 %; at `b = 1.5` no
  width or length tried removes the last 0.017 (2 %). On five runs, 5 × 10 000 steps gain
  0.006 on average over the paper recipe and never rescue a stuck run
  (`outputs_appendix_sequential_long_*`).
- **Monte Carlo size of the individuals.** The draws for the final `δ4` choice matter most
  (4 → 64 draws: +0.002 at `b = 1` and `1.5`); the step-2 draws inside step 1 add at most
  another 0.002; 256 final draws change nothing further. The final runs use 16 draws in step 1
  and in its inner step-2 evaluations, and 64 for the final choice, so the individuals are at
  their optimum within the evaluation noise of 4000 individuals (about 0.003).
- **Fraction of shifts at the bound** (recorded for every follower). `δ4` is at `±b` for all
  individuals (1.00) and nearly all network outputs (0.96 to 0.99), confirming the step
  structure. The `δ1` fraction is not informative about optimality: in the linear-Gaussian SCM
  the individuals have 15 % of `δ1` at the bound and the networks 73 to 86 %, and the payoffs
  still agree within 1 %, so the payoff is flat in `δ1` for most individuals.
- **Omniscient follower** (sees all of its noises, including `ε_Y`, before intervening). Its
  `δ1` depends on `ε_Y`, so after deployment `X1` and `Y` share a noise term: hidden
  confounding, the path `E → X1 ← ε_Y → Y` is open given `X1`, `Y | X_{PA(Y)}` is no longer
  invariant and Lemmas 2, 4 and Theorem 3 do not apply. In the nonlinear SCM it raises the
  `PA(Y)` and `SB(Y)` curves by 0.03 to 0.07. Nuance: with additive noise and the true regression
  function the residual is `ε_Y` whatever the follower does, so the risk of `PA(Y)` and `SB(Y)`
  is unchanged even under this confounding (the linear-Gaussian rows are exactly flat); what the
  omniscient follower exploits in the nonlinear case is the estimation error of the fitted
  predictor. One sentence in the discussion, no experiment; in the tex it reads: "Our follower model
  also excludes individuals who observe the noise of their own outcome before intervening (for
  example, their true creditworthiness): the intervened covariates and `Y` would then share a
  noise term, that is, they would be confounded, and the invariance results would no longer
  apply.""
- **Information ladder** (one-shot `O = {X2}`, sequential without and with `X3`): dropped. The
  ordering is by construction and the magnitudes are small (below 0.01 per added variable
  against 0.2 to 0.4 per half unit of `b`); `outputs_information_ladder_*` is kept.

## 7. Reproduction

All commands run from `scripts/synthetic/` and are listed in the README.

- Final appendix runs (both SCMs, 10 runs, 4000 individuals per run):
  `run_individual_follower.py --followers nn_x1 nn_x1_tuned ind_nox3 --num-runs 10 --n-jobs 5
  --torch-num-threads 3 --grid-size-x1 13 --mc-stage1 16 --mc-stage2 16 --mc-final 64
  [--lineargaussian]`, outputs `outputs_appendix_final_{standard,lingauss}/` (per-run and summary
  CSVs, paired differences, `individual_vs_population.*`, `followers_grid.*`). Two-row figure:
  `plot_appendix_sequential.py outputs_appendix_final_standard outputs_appendix_final_lingauss
  --followers nn_x1 ind_nox3 --out ...` (paper recipe against the individuals; add
  `nn_x1_tuned` for the diagnostic version).
- Training recipes: `tune_attack_training.py --num-runs 3`, outputs `outputs_attack_tuning/`.
- Only if the augmented follower or the recipe is adopted later: Fig. 2 via `run_experiments.py
  --output-dir outputs_x4-uses-x1 --x4-uses-x1 [--attack-continuation]` (and `--lineargaussian`);
  the plotting script expects the output folder names it hard-codes.
- Code: `synthetic_experiments/individual_follower_simple.py` (the individuals; `observe_x3`,
  `n_mc`, `n_mc_step2`, `n_mc_final`), `synthetic_experiments/scm.py` (`AttackMechanisms` with
  `x4_uses_x1`, `depth`, `slope`), `synthetic_experiments/adversary.py` (`optimize_attack` with
  `init_candidates`, `lr_schedule`, `depth`, `slope`, `init_attack`),
  `synthetic_experiments/warm_start.py` (diagnostic only), `run_individual_follower.py`
  (followers `nn_standard`, `nn_x1`, `nn_x1_tuned`, `nn_x1_long`, `nn_x1_warm`, `ind_nox3`,
  `ind_omni`, ...; `--methods`, `--tuned-*` options), `tune_attack_training.py`,
  `plot_appendix_sequential.py`.
- Cost on this machine (14 cores): individuals about 2.5 min per all-variable combination with
  16/16/64 draws (seconds for `PA(Y)`, `SB(Y)`); paper recipe 25 to 60 s per combination,
  continuation recipe 1.2 times that; the final run of both SCMs about 3 hours with 5 workers.
- Nothing is committed; all outputs are untracked. The first pass of this experiment (4 draws in
  step 2) is kept as `outputs_appendix_sequential_*_mc4`.
