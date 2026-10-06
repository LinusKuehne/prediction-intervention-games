# Individual followers in the synthetic experiment (§6.1)

> **Note (2026-10-05):** scripts and output folders not used for the paper were moved to `further_experiments/` (gitignored); folder names below refer to their former location in `scripts/synthetic/`. The E.5 results now in the paper are in `scripts/synthetic/outputs_appendix_anneal_{standard,lingauss}`.

Report for Linus, 2026-10-02, revised 2026-10-04. Everything below refers to the code in `scripts/synthetic/`.
The final form of the appendix experiment, with the improved training of the neural follower
and the text changes for the paper, is documented in `appendix_experiment_report.md`; Section 8
below is its first pass and is kept for the history.
All numbers are the leader's deployment MSE unless stated otherwise. Lower is better for the
leader; a follower that "maximizes MSE" wants these numbers high.

## 0. Where the implemented followers stand relative to §2.1 and §6.1

§2.1 makes five assumptions about the individual follower. Writing them out makes it easy to
say, for every follower in this report, which ones it satisfies.

- (A) *Observation.* The individual observes `x_O` with `O ⊆ ND(CH(E))`: only variables that are
  upstream of every intervention target, so the observation cannot be affected by the choice it
  informs.
- (B) *Parents known.* `PA(j) ⊆ O ∪ CH(E)` for every target `j`: the individual knows at least
  what a §2 population follower may always use.
- (C) *Objective.* `V = E[k(f, X, Y) | X_O = x_O]`, an expectation of a per-individual quantity
  (per-individual constraints are implicitly allowed; §2.1 itself has none).
- (D) *Hard, deterministic choice.* The individual sets each target to a constant `c_j(x_O)`:
  no dependence on the target's own noise `ε_j`, no randomization, no bound.
- (E) *Best response.* The individual actually maximizes `V`.

| follower | targets | observed before each choice | own noise `ε_j` | choice | (A) | (B) | (C) | (D) | (E) | pendant of |
|---|---|---|---|---|---|---|---|---|---|---|
| §2.1 individual (as written) | `CH(E)` | `x_O`, `O ⊆ ND(CH(E))`, one shot | no | constant `c_j` | yes | yes | yes | yes | yes | population follower with `A_j = O \ PA(j)` |
| §6.1 standard follower (NN) | `X1, X4` | `δ1`: `(X2, ε1)`; `δ4`: `(Y, X2, ε4)` | yes | shift `∣δ∣ ≤ b` | no (`δ4` sees `Y`) | stage-wise | yes | no | approx. | two agents with different information (team) |
| §6.1 Fig. 5 follower (NN, `x1x3`) | `X1, X4` | `δ1`: `(X2, ε1)`; `δ4`: `(X1, X2, Y, X3, ε4)` | yes | shift | no | stage-wise | yes | no | approx. | sequential individual, perfect recall |
| `ind_seq` / `ind_simple`, two targets (Exp. 1, 2) | `X1, X4` | as Fig. 5 follower | yes | shift | no | stage-wise | yes | no | yes (grid) | exact pendant of the Fig. 5 follower |
| `nn_x1` (Sec. 8) | `X1, X4` | `δ1`: `(X2, ε1)`; `δ4`: `(X1, X2, Y, ε4)` | yes | shift | no | stage-wise | yes | no | approx. | population follower with `A4 = {X1}` |
| `ind_nox3` (Sec. 8) | `X1, X4` | as `nn_x1` | yes | shift | no | stage-wise | yes | no | yes (grid) | exact pendant of `nn_x1` |
| `ind_pa4`, `X4` only (Sec. 7) | `X4` | `(Y, X2, ε4)` | yes | shift | yes | yes | yes | shift, not constant | yes | exact pendant of the standard follower (`A4 = ∅`) |
| `ind_simple`, `X4` only (Sec. 7) | `X4` | `(X1, X2, X3, Y, ε4)` | yes | shift | yes | yes | yes | shift, not constant | yes | exact pendant of the Fig. 5 follower (`A4 = {X1, X3}`) |
| `ind_greedy` (Sec. 4) | `X1, X4` | as Fig. 5 follower | yes | shift | no | stage-wise | yes | no | **no** (myopic) | none; an approximation |
| `ind_omni` | `X1, X4` | all own noises incl. `ε_Y` | yes | shift | formally yes, but outside §2 | yes | yes | no | yes | none; upper bound |

Three remarks on the table.

- The only way in which the implemented individuals deviate from (D) is that they choose a
  bounded shift around the natural value instead of a free constant. This is a per-individual
  constraint set `[natural − b, natural + b]` that depends on `(x_O, ε_j)`; the argument of §2.1
  goes through unchanged with it (one conditions on `(x_O, ε_j)`). It is forced by §6.1, whose
  follower has exactly this constraint.
- (A) fails for every two-target follower because `δ4` reacts to `Y`, which is downstream of
  `X1`. This is not a defect of the followers but of the one-shot formulation of §2.1; the
  stage-wise version of (A), "`O_j ⊆ ND(j)` for each target", holds for all of them except
  `ind_omni`.
- `ind_omni` shows why (A) must be read as "observed covariates": the noises `ε_Y, ε3, ε5` are
  formally non-descendants of everything, but an intervention depending on `ε_Y` is correlated
  with `Y`'s noise, which is hidden confounding between `X_j` and `Y`, breaks the invariance of
  `Y | X_{PA(Y)}`, and is excluded by §2's "independent exogenous noise".

**Can we make a good comparison at all?** Yes, if one is clear about what it can show. §2.1
is a theorem: the optimal population follower with `A_j = O \ PA(j)` and the optimal individuals
observing `x_O` are the same follower written in two ways. No simulation can make that false. If
the two curves differ, either one side did not reach its optimum or the two sides were not given
the same problem. Here the individuals use exhaustive search, so they are at the optimum up to
grid and Monte Carlo error; the neural follower uses gradient descent and may stop early. So the
experiment measures how close the §6.1 neural follower is to the best response, and illustrates
the construction. Every gap means the neural attack is weaker than the best response, never
stronger. For the gap to mean this, both sides must see the same variables, use the own noise
`ε_j` the same way and act at the same time; otherwise one compares two different followers.
The two-target experiments (Sections 3, 4, 8) align them
with the §6.1 follower and need the sequential reading of §2.1; Section 8 makes the alignment
exact by adding `X1` to the inputs of `δ4`, and that is the experiment I would put in the
appendix. The `X4`-only experiment (Section 7) aligns them with §2.1 as written (it satisfies
(A), (B), (C) and (E) literally and differs from (D) only by the harmless bounded-shift
constraint) and is kept for reference. What no experiment of this kind can test is the fine print of §2.1 (existence of a
measurable maximizer, tie-breaking) or objectives that are not expectations of a per-individual
quantity, which the paper does not use.

---

## 1. How the experiment in §6.1 works, and what the follower observes

**Data.** The SCM has covariates `X1,…,X5`, with `X2 → X1 → Y`, `X2 → Y`, `Y → X3`,
`X2 → X4 ← Y`, `X4 → X5`, Gaussian noises `ε`. Nonlinear and linear-Gaussian versions are in
Appendix E.1. The intervention targets are `X1` and `X4` (children of `E`), so
`PA(Y) = {X1, X2}`, `SB(Y) = {X1, X2, X3}`.

**Leader.** Three predictors: `PA(Y)`, `SB(Y)`, all variables. Nonlinear case: an MLP
(5 hidden layers of width 256) trained on 50 000 observations of the clean SCM. Linear-Gaussian
case: the exact population regression function (no estimation error).

**Follower.** The clean structural assignments of the two targets are kept and a bounded
perturbation is added:

- `X1 = clean_1(X2, ε1) + δ1`, with `δ1 = b · tanh(h1(X2, ε1))`;
- `X4 = clean_4(Y, X2, ε4) + δ4`, with `δ4 = b · tanh(h4(Y, X2, ε4))`;
  in the variant of Fig. 5 (right) `h4` additionally gets `X1` and `X3`.

`h1`, `h4` are small MLPs (two hidden layers of 64). For a fixed predictor they are trained
jointly by Adam for 2000 steps on fresh batches of 512 draws from the SCM, back-propagating through
the SCM, either to maximize the leader's MSE or to minimize the mean prediction `E[f_S(X_S)]`.
Five random restarts; the best on 10 000 held-out draws is kept. The leader's MSE is then measured
on 10 000 fresh draws from the intervened SCM. Bounds `b ∈ {0.25, 0.5, 1, 1.5}`, 10 runs.

**What the follower observes.** This is determined by the network inputs:

- the `X1`-intervention sees `X2` (the parent of `X1`) and `ε1` (the noise of `X1`'s own
  mechanism);
- the `X4`-intervention sees `Y` and `X2` (the parents of `X4`) and `ε4`; in the Fig. 5
  variant also `X1` and `X3`.

The follower also knows the SCM and the deployed `f` (it back-propagates through both); this is
what "best response" means. In the language of §2, the standard experiment has action sets
`A1 = A4 = ∅` (interventions depend on parents only) and the Fig. 5 variant has
`A4 = {X1, X3}`, which satisfies the condition of Lemma 4 (`A4 ⊆ {X1, X2, X3}`), so the stable
blanket is unchanged. That is why the paper can say the Fig. 5 result "is consistent with § 3.3".

**Is this described sufficiently?** Appendix E.1 states the inputs of `Eδ1` and `Eδ4` and the
bound, and E.4 states the training. Three points are implicit and I would make them explicit:

1. *The follower may use the mechanism's own noise.* Because `δ4` depends on `ε4`, the follower
   does not just add an independent shift; it can replace `X4`'s mechanism by any function of
   `(Y, X2, ε4)` within a band of width `b` around the clean one (same for `X1`). This fits the
   §2 formalism ("a measurable function of `X_{PA(j) ∪ S_j}` plus independent exogenous noise",
   if `ε_j` is read as that noise), but a reader will not guess it from "adds perturbations".
2. *The deployment graph.* One sentence saying "`A_j = ∅`; in the variant, `A4 = {X1, X3}`"
   connects the experiment to §2 and Lemma 4 without further explanation.
3. *The individual reading.* §2.1 describes an individual who observes `x_O` with
   `O ⊆ ND(CH(E))` and then sets *all* of `CH(E)` at once, assuming `PA(j) ⊆ O ∪ CH(E)`. Here
   `CH(E) = {X1, X4}` and `X1 → Y → X4`, so `ND(CH(E)) = {X2}` and the assumption fails for `X4`
   (its parent `Y` is a descendant of `X1`). The experiment's follower lets `δ4` react to the
   realized `Y`. Read as individuals, this is a *sequential* follower: choose `δ1` after seeing
   `(x2, ε1)`, then choose `δ4` after seeing the realized `(…, y, …)`. The argument of §2.1
   (the pointwise optimum of the conditional objective is the population optimum) still holds,
   stage by stage, by backward induction; the paper just does not say so. As a population
   follower (§2) the experiment is described correctly; it is only the link to §2.1 that needs
   one sentence.

## 2. Felix's comment

> Isn't this basically the experiment with additional `X1` and `X3` information? The follower
> can choose any conditional distribution (which can also include `X1` and `X3`), which in my
> opinion is the same as individual followers, no?

Yes, for two reasons.

*Which information.* An individual who sets their own `X1` knows the value they set, and when
they set `X4` they have also seen their `y` and, if it is realized by then, their `x3` (the SCM
does not order `X3` and `X4`). A rational individual uses all of this and does not forget `x1`.
So "the individual follower" in this SCM is the one whose `X4`-choice depends on
`(x1, x2, y, x3)`, the input set of the Fig. 5 follower; the standard follower, whose `δ4` sees
only `(y, x2)`, would be an individual who forgot the `x1` they just set. That is why Felix points
to the experiment with `X1` and `X3` added.

*Why one population function equals many individual choices.* The population follower picks one
function `h4(x1, x2, y, x3, ε4)` for everyone and maximizes the average payoff. The average is an
integral over input values and the function is free at each input, so the best function is found
input by input: at each input it takes the value that maximizes the payoff expected given that
input, which is what the individual with that input does. Conversely, a population function that
is suboptimal on a set of inputs of positive probability could be improved there. Both followers
therefore produce the same `(δ1, δ4)` for every individual and the same leader risk. This is the
argument of §2.1. The same holds for `h1(x2, ε1)`; the two steps fit together because one
individual makes both choices, so the jointly optimal `(h1, h4)` is the pair in which `δ1`
anticipates the optimal `δ4`, i.e. backward induction.

Three precisions:

- It is "the same" for the *optimal* follower. The neural follower is an approximation, so the
  curves can differ by its optimization error; this is what the experiments below measure.
- The equivalence needs the objective to be an expectation of a per-individual quantity (true
  for both objectives in the paper) and the constraint to be per individual (true for the bound
  `b`). A budget on the population, e.g. `E[δ²] ≤ B`, would break it.
- "Individual follower" is not one thing: it is defined by what the individual observes. The
  Fig. 5 follower has the largest observation set allowed by Lemma 4 (`A4 ⊆ {X1, X2, X3}`), so
  it is the most informed admissible individual follower. The *standard* follower (`δ4` without
  `X1`) is not literally an individual with perfect recall (an individual who set `X1` would
  remember it); it is a team of two agents with different information. Numerically this makes
  no difference (Section 3).

## 3. Experiment 1: the exact individual follower against the neural follower

**Code.** `synthetic_experiments/individual_follower.py`, `run_individual_follower.py`.
Outputs: `outputs_individual_follower_standard/` (nonlinear, 3 runs, paper-scale predictors,
4000 individuals per run) and `outputs_individual_follower_lingauss/` (5 runs).

**What the individual follower does.** One run draws 4000 individuals (one draw of all noises
each). Every individual

1. observes `(x2, ε1)` and chooses `δ1 ∈ [−b, b]` on a grid of 25 values, maximizing their
   expected objective, where the expectation over the not-yet-realized `(ε_Y, ε3, ε4, ε5)` is a
   Monte Carlo average over 32 draws and already accounts for the optimal second choice
   (backward induction);
2. then observes the realized `(x1, x2, y, x3, ε4)` and chooses `δ4 ∈ [−b, b]` on the grid,
   averaging over 6 draws of `ε5`.

The leader's MSE is then computed on these 4000 individuals. The neural followers are trained as
in the paper and then *replayed on the same 4000 noise draws*, so every comparison is paired.

**How this follower differs from the paper's follower.** Same information as the Fig. 5
follower (`nn_x1x3`), same objective, same bound. The difference is only how the choice is
made: there is no network and no training; each individual's choice is a near-exact optimum
(the search is exhaustive; the only approximations are the grid and the Monte Carlo average,
which I checked do not matter at the third decimal). I also ran an *omniscient* individual who
knows all of their own noises (including `ε_Y`) and optimizes the realized objective. That one is
outside the framework of §2, because its intervention is correlated with `Y`'s noise; it is an
upper bound only.

**Results, nonlinear SCM** (3 runs; `none` = no intervention):

| objective | predictor | b | none | NN standard | NN x1x3 | individual | omniscient |
|---|---|---|---|---|---|---|---|
| max MSE | all variables | 0.5 | 0.012 | 0.116 | 0.117 | 0.124 | 0.130 |
| max MSE | all variables | 1.0 | 0.012 | 0.328 | 0.330 | 0.379 | 0.403 |
| max MSE | all variables | 1.5 | 0.012 | 0.655 | 0.657 | 0.756 | 0.817 |
| max MSE | PA(Y) | 1.0 | 0.160 | 0.184 | 0.184 | 0.182 | 0.224 |
| max MSE | PA(Y) | 1.5 | 0.160 | 0.267 | 0.267 | 0.272 | 0.336 |
| max MSE | SB(Y) | 1.0 | 0.096 | 0.125 | 0.125 | 0.128 | 0.171 |
| max MSE | SB(Y) | 1.5 | 0.096 | 0.143 | 0.143 | 0.145 | 0.209 |
| min E[f] | all variables | 1.0 | 0.012 | 0.253 | 0.253 | 0.253 | 0.254 |
| min E[f] | all variables | 1.5 | 0.012 | 0.557 | 0.557 | 0.557 | 0.561 |
| min E[f] | PA(Y) | 1.5 | 0.160 | 0.192 | 0.192 | 0.192 | 0.192 |
| min E[f] | SB(Y) | 1.5 | 0.096 | 0.125 | 0.125 | 0.125 | 0.125 |

Full table: `outputs_individual_follower_standard/results_summary.csv`; figures
`followers_grid.png` (all followers) and `individual_vs_population.png` (paper layout) in the
same folder.

![Experiment 1, nonlinear SCM: neural population follower (solid) against the exact individual follower with the same information (dashed), on identical individuals.](further_experiments/outputs_individual_follower_standard/individual_vs_population.png)

![Experiment 1, nonlinear SCM, all followers per predictor and objective. Dotted horizontal lines: MSE without intervention.](further_experiments/outputs_individual_follower_standard/followers_grid.png)

**Results, linear-Gaussian SCM** (5 runs, population predictors): `PA(Y)` gives 0.158 and
`SB(Y)` gives 0.084 at every bound under *every* follower including the omniscient one, exactly
as §3.1 predicts for additive noise. Under the prediction objective all followers coincide to
four decimals for all predictors (the optimum is the constant shift `δ = −b`). For the
all-variable predictor under the MSE objective:

| b | NN standard | NN x1x3 | individual | omniscient |
|---|---|---|---|---|
| 0.5 | 0.153 | 0.156 | 0.158 | 0.158 |
| 1.0 | 0.408 | 0.432 | 0.464 | 0.464 |
| 1.5 | 0.882 | 0.851 | 0.934 | 0.934 |

Per run the neural follower is bimodal: at `b = 1` it reaches either about 0.45 (near the
individual optimum 0.464) or about 0.345, and at `b = 1.5` either 0.91 or 0.755, depending on the
run and the input configuration, despite five restarts. That is a local optimum of the neural
follower, not a modeling difference: with the same information the exact individual follower
cannot be beaten, and the omniscient one gains nothing extra here.

**What this shows.** With the same information, individual and population follower agree,
as §2.1 claims. Where they differ (all-variable predictor, MSE objective, large bounds), the
individual follower is the stronger attacker, so the all-variable curve in Fig. 2 is, if anything,
underestimated. `PA(Y)` and `SB(Y)` move by less than 0.004 in the plotted range. More
information than the framework allows (the omniscient follower) does change the numbers,
including for `PA(Y)` and `SB(Y)` in the nonlinear case, by exploiting the finite-sample
regression error of §E.3.

## 4. Experiment 2: a simpler implementation, cheaper settings, and a version of Fig. 2 without any neural follower

**Code.** `synthetic_experiments/individual_follower_simple.py` (self-contained; about 100
lines of code), selected in `run_individual_follower.py` with `--followers ind_simple`.
Validation: `outputs_implementation_comparison/results_run0.csv` (one set of paper-scale
predictors, the same 4000 individuals for every follower).

**What changed.** Only the implementation and the settings, not the follower: same two-step
choice, same information, a grid of 13 instead of 25 values and 16 / 4 Monte Carlo draws instead
of 32 / 6. I also added a *greedy* variant in which `δ1` is chosen as if no intervention on `X4`
followed; that is a different (myopic) follower.

All-variable predictor, MSE objective, run 0:

| b | NN x1x3 | original, fine | original, coarse | compact 16/4 | compact 8/4 | greedy |
|---|---|---|---|---|---|---|
| 0.5 | 0.115 | | 0.124 | 0.124 | 0.124 | 0.122 |
| 1.0 | 0.355 | | 0.378 | 0.378 | 0.378 | 0.366 |
| 1.5 | 0.709 | 0.754 | 0.751 | 0.753 | 0.751 | 0.701 |

Under the prediction objective every column equals 0.0741, 0.2610 and 0.5849 at the three
bounds. `PA(Y)` and `SB(Y)` agree within 0.003 across implementations. Time per combination on
this machine: fine original 18 min, compact 16/4 about 2 min, greedy 3 s, neural follower 2 to
3 min.

So the compact follower with grid 13 and 16 / 4 draws reproduces the fine original to the
third decimal at a tenth of the cost; the greedy follower is exact under the prediction objective
but under the MSE objective loses about as much as the neural follower does, so backward induction
is worth keeping.

**Fig. 2 without a neural follower.** `outputs_individual_only_standard/` (10 runs, 2000
individuals per run) and `outputs_individual_only_lingauss/` (10 runs) repeat the paper's
figure with the compact individual follower as the *only* follower; see
`individual_vs_population.png` in each. Nonlinear case:

| objective | b | PA(Y) | SB(Y) | all variables |
|---|---|---|---|---|
| max MSE | 0.25 | 0.163 | 0.104 | 0.050 |
| max MSE | 0.5 | 0.165 | 0.112 | 0.124 |
| max MSE | 1.0 | 0.187 | 0.126 | 0.378 |
| max MSE | 1.5 | 0.286 | 0.144 | 0.761 |
| min E[f] | 0.25 | 0.163 | 0.100 | 0.026 |
| min E[f] | 0.5 | 0.163 | 0.104 | 0.071 |
| min E[f] | 1.0 | 0.171 | 0.114 | 0.253 |
| min E[f] | 1.5 | 0.200 | 0.129 | 0.561 |

Same picture as Fig. 2, with the all-variable curve 5 to 15 % higher at `b ≥ 1`. This version
needs no attack architecture, optimizer, restarts or held-out selection, and its description is
two sentences (Section 7).

![Fig. 2 of the paper redone with the individual follower as the only follower (nonlinear SCM, 10 runs).](further_experiments/outputs_individual_only_standard/individual_vs_population.png)

(A first version of the implementation comparison accidentally used a 2-layer MLP of width 64 as
predictor; it is kept, labelled, in `outputs_implementation_comparison_mlp2x64/` and shows the
same ordering. The table above is the paper-scale one.)

## 6. Mistakes and caveats

- The first implementation comparison used the wrong predictor size (see Section 4).
- Retraining a predictor is not bitwise reproducible across processes (4th digit of the clean
  MSE), so comparisons are paired only within one process or one run; all tables above are.
- Killing a `joblib` parent with `pkill` left worker processes running for over an hour; they
  wrote nothing but slowed everything down. Cleaned up; the CSVs are unaffected.

## 7. Proposed appendix experiment: validating §2.1 empirically

**Code and outputs.** `run_individual_follower.py --disable-x1-intervention --followers
nn_standard nn_x1x3 ind_pa4 ind_simple ind_omni`; results in
`outputs_appendix_x4only_standard/` (nonlinear, 3 runs, paper-scale predictors, 4000
individuals per run) and `outputs_appendix_x4only_lingauss/` (3 runs). Figures
`individual_vs_population.png` (paper layout) and `followers_grid.png` in each folder.

**Design.** Keep the §6.1 SCM (both versions), the three predictors, and the neural population
follower exactly as in the paper, but let the follower intervene on `X4` only. Then
`CH(E) = {X4}`, `ND(CH(E)) = {X1, X2, Y, X3}`, and §2.1 applies literally: an individual observes
`x_O` with `O ⊆ {X1, X2, X3, Y}` and `PA(X4) = {Y, X2} ⊆ O`, and chooses the single value
`δ4 ∈ [−b, b]`. Two pairs are compared on the same individuals:

- the paper's standard neural follower (`δ4` from `Y, X2, ε4`) against individuals who observe
  only `(y, x2)` and their own `ε4`, i.e. `O = PA(X4)`, `A4 = ∅`;
- the Fig. 5 neural follower (`δ4` from `Y, X2, ε4, X1, X3`) against individuals who observe
  `(x1, x2, x3, y)` and `ε4`, i.e. `A4 = O \ PA(X4) = {X1, X3}`.

Each individual maximizes `E[k | x_O, ε4]` over a grid of 13 values of `δ4`; the expectation over
what they do not observe uses 64 Monte Carlo draws (for the first pair, `X1` is drawn from its
posterior given `(y, x2)` by importance sampling, which is explicit because `Y` has additive
Gaussian noise). The neural followers are trained as in the paper and replayed on the same
individuals.

**Results.** `PA(Y)` and `SB(Y)` are unaffected by any `X4`-intervention, and indeed every
follower gives exactly the clean values (0.160 and 0.096 nonlinear; 0.159 and 0.084
linear-Gaussian) at every bound. For the all-variable predictor:

Nonlinear SCM:

| objective | b | NN standard | individual, O = (Y, X2) | NN x1x3 | individual, O = (X1, X2, X3, Y) |
|---|---|---|---|---|---|
| max MSE | 0.25 | 0.043 | 0.044 | 0.044 | 0.045 |
| max MSE | 0.5 | 0.100 | 0.102 | 0.101 | 0.105 |
| max MSE | 1.0 | 0.289 | 0.298 | 0.270 | 0.304 |
| max MSE | 1.5 | 0.576 | 0.603 | 0.554 | 0.611 |
| min E[f] | 0.25 | 0.0255 | 0.0255 | 0.0255 | 0.0255 |
| min E[f] | 0.5 | 0.0672 | 0.0672 | 0.0672 | 0.0672 |
| min E[f] | 1.0 | 0.2369 | 0.2369 | 0.2369 | 0.2369 |
| min E[f] | 1.5 | 0.5193 | 0.5193 | 0.5193 | 0.5193 |

Linear-Gaussian SCM:

| objective | b | NN standard | individual, O = (Y, X2) | NN x1x3 | individual, O = (X1, X2, X3, Y) |
|---|---|---|---|---|---|
| max MSE | 0.5 | 0.153 | 0.154 | 0.156 | 0.158 |
| max MSE | 1.0 | 0.451 | 0.456 | 0.454 | 0.464 |
| max MSE | 1.5 | 0.913 | 0.922 | 0.916 | 0.935 |
| min E[f] | 1.0 | 0.3454 | 0.3454 | 0.3454 | 0.3454 |
| min E[f] | 1.5 | 0.7565 | 0.7566 | 0.7565 | 0.7566 |

Paired over runs, the individual follower with the same information is never below its neural
counterpart and exceeds it by 0.0005 to 0.027 (standard pair) and by 0.001 to 0.057 (`x1x3`
pair; the larger gaps come from one run in which the neural follower stopped at a worse local
optimum: 0.228 instead of about 0.29 at `b = 1`). Under the prediction objective the two
followers coincide to four decimals. The omniscient individual coincides with the
`(X1, X2, X3, Y)` individual here (the only thing it additionally knows is `ε5`, which hardly
matters), so with `X4` as the only target the admissible individual follower is already the
strongest possible one.

![Appendix experiment, nonlinear SCM: each neural follower (filled markers) and its individual pendant with the same observation set (open markers), on identical individuals.](further_experiments/outputs_appendix_x4only_standard/individual_vs_population.png)

![Appendix experiment, linear-Gaussian SCM with population predictors.](further_experiments/outputs_appendix_x4only_lingauss/individual_vs_population.png)

**Reading.** This is §2.1 in numbers: a population of individuals who each observe `x_O` and
best-respond produces the same deployment distribution, hence the same leader risk, as the
population follower whose intervention may depend on `X_O`; the only differences are the neural
follower's optimization error, which always goes in the direction of a weaker attack. The
`x1x3` pair also shows the augmentation of §2.1 at work: `A4 = {X1, X3}` is exactly
`O \ PA(X4)`.

**Why this and not the two-target version.** With both targets the individual follower is
sequential (Section 1, point 3), which is a mild extension of §2.1 rather than §2.1 itself.
Section 8 takes the other route: extend §2.1 to the sequential case and keep both targets, so
that the appendix experiment is the §6.1 experiment itself with the follower implemented at the
individual level. I now prefer that route; this section is kept for reference.

**Suggested text for the appendix** (two sentences plus the figure):

> To check the equivalence of §2.1 empirically, we repeat the experiment of §6.1 with
> interventions on `X4` only and compare the population follower with a population of individual
> followers: each individual observes `x_O` (with `O = PA(X4) = {Y, X2}` or
> `O = {X1, X2, X3, Y}`, matching the inputs of the population follower's intervention) and
> chooses `δ4 ∈ [−b, b]` by exhaustive search to maximize their conditional expected objective.
> The leader's deployment MSE under the two followers coincides up to the optimization error of
> the neural population follower (Fig. X); `PA(Y)` and `SB(Y)` are unaffected by either.

## 8. The two-target appendix experiment (plan of 4 October)

**The plan.** Keep the SCM and both targets of §6.1 and read §2.1 sequentially: the individual
sets `X1` after observing `(x2, ε1)` and then `X4` after observing `(y, ε4)` and recalling
`(x1, x2)`; `x3` and `ε5` are not observed. Change the §6.1 follower in one place so that it is
the exact population pendant of this individual: `δ4` additionally gets `X1` as input, i.e.
`A1 = ∅`, `A4 = {X1}` (allowed by Lemma 4). Compare the neural follower with the individuals on
the same noise draws, both objectives, both SCMs, the bounds of Fig. 2, 10 runs. Check that a
remaining gap is optimization error by warm-starting the networks at the individuals' rules.

**Implementation.** `AttackMechanisms` and `optimize_attack` got an option `x4_uses_x1` (input
`(Y, X2, ε4, X1)` for `δ4`) and `optimize_attack` an optional initial network. The new module
`synthetic_experiments/warm_start.py` fits the two attack networks (2 hidden layers of 64, as
in the paper) by least squares to the individuals' `(δ1, δ4)` with 3000 full-batch Adam steps.
`run_individual_follower.py` has four new followers:

- `nn_x1`: the paper's follower with `X1` added to `δ4`, trained as in E.4 (5 restarts,
  2000 steps);
- `ind_nox3`: the compact individual follower with `observe_x3=False` (grid of 13 values;
  16 Monte Carlo draws in step 1 and in the step-2 evaluations inside it; 64 draws of
  `(ε3, ε5)` for the final `δ4` choice, common random numbers across the grid);
- `nn_x1_fit`: the networks fitted to `ind_nox3`'s rules, before any training (measures what
  the network class can represent);
- `nn_x1_warm`: `nn_x1_fit` followed by the paper's training (2000 steps, 1 restart);
- `nn_x1_long`: `nn_x1` trained for 10 000 instead of 2000 steps (run separately for the
  all-variable predictor, MSE objective, `b ∈ {1, 1.5}`, 5 runs, both SCMs; outputs
  `outputs_appendix_sequential_long_*`).

`nn_standard` (the paper's follower) and `ind_omni` (the omniscient individual, outside the
framework) are run alongside. Outputs: `outputs_appendix_sequential_standard/` (nonlinear) and
`outputs_appendix_sequential_lingauss/`; commands in the README.

**Results, nonlinear SCM** (10 runs, 4000 individuals per run; mean over runs). Under the
prediction objective all six followers coincide to within 0.0003 for every predictor and bound.
`PA(Y)` and `SB(Y)` under the MSE objective agree within 0.003 between the neural follower and
the individuals (one exception: `PA(Y)` at `b = 1.5`, +0.009 ± 0.006 for the individuals, where
the neural follower is sometimes stuck). All-variable predictor, MSE objective:

| b | none | NN standard | NN +X1 | individual | NN fitted to rules | NN warm-started | omniscient |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.012 | 0.047 | 0.047 | 0.048 | 0.048 | 0.048 | 0.052 |
| 0.5 | 0.012 | 0.116 | 0.116 | 0.120 | 0.118 | 0.119 | 0.130 |
| 1.0 | 0.012 | 0.342 | 0.345 | 0.371 | 0.363 | 0.364 | 0.404 |
| 1.5 | 0.012 | 0.676 | 0.681 | 0.752 | 0.728 | 0.732 | 0.821 |

Paired differences over runs (mean ± s.e.m.), all-variable predictor, MSE objective:

| difference | b = 0.5 | b = 1.0 | b = 1.5 |
|---|---|---|---|
| NN +X1 − NN standard (effect of the one-line change) | +0.001 ± 0.000 | +0.002 ± 0.003 | +0.005 ± 0.011 |
| individual − NN +X1 (same information) | +0.004 ± 0.000 | +0.027 ± 0.008 | +0.071 ± 0.011 |
| NN fitted − individual (before training) | −0.002 ± 0.000 | −0.009 ± 0.000 | −0.025 ± 0.001 |
| NN warm-started − NN +X1 | +0.002 ± 0.000 | +0.019 ± 0.008 | +0.051 ± 0.012 |
| individual − NN warm-started | +0.001 ± 0.000 | +0.008 ± 0.001 | +0.021 ± 0.002 |
| omniscient − individual | +0.010 ± 0.000 | +0.032 ± 0.001 | +0.069 ± 0.002 |

For `PA(Y)` and `SB(Y)` the difference NN +X1 − NN standard is exactly zero (these predictors
ignore `X4`, so the `δ4`-network is irrelevant and the `δ1`-networks are trained identically).
Per run, the cold-started neural follower at `b = 1` is bimodal (0.276 in one run, 0.34 to 0.36
in the others) while the individuals lie between 0.363 and 0.380 and the warm-started network
between 0.353 and 0.377, always within 0.013 of the individuals of the same run. Figures:
`outputs_appendix_sequential_standard/individual_vs_population.png` (below) and
`followers_grid.png`.

![Nonlinear SCM, 10 runs: the paper's follower (dotted), the same with `X1` added to `δ4` (solid), the exact individual follower with the same information (dashed, open markers) and the network warm-started at the individuals' rules (dash-dotted), all on identical individuals.](further_experiments/outputs_appendix_sequential_standard/individual_vs_population.png)

**Two checks on the individuals and on the networks** (run 0, all-variable predictor, MSE
objective).

- *Monte Carlo size.* The draws for the final `δ4` choice matter most: going from 4 to 64 final
  draws raises the individuals' MSE by 0.002 at `b = 1` and at `b = 1.5`; raising the step-2
  draws inside step 1 from 4 to 16 adds at most another 0.002; 256 final draws change nothing
  further. The final runs use 16 inner and 64 final draws, so the individuals are at their
  optimum within the evaluation noise of 4000 individuals (about 0.003).
- *Capacity or optimization?* The least-squares fit to the rules is not the payoff-optimal
  projection onto the network class (near a jump of the rule the squared error is large but the
  payoff difference is small), so fit-only numbers overstate what the class cannot represent;
  with the 64-draw rules the fitted network is within 0.009 (`b = 1`) and 0.023 (`b = 1.5`) of
  the individuals, against 0.026 and 0.052 with the noisier 4-draw rules of the first pass. I
  then varied width and training length:

| network | b = 1 | b = 1.5 |
|---|---|---|
| individuals | 0.370 | 0.741 |
| cold start, width 64, 5 × 2000 steps (the paper's setting) | 0.353 | 0.718 |
| cold start, width 64, 5 × 10 000 steps | 0.367 | 0.724 |
| cold start, width 256, 5 × 2000 steps | 0.299 | 0.660 |
| fitted to the rules only, width 64 | 0.361 | 0.719 |
| warm start, width 64, 2000 steps | 0.364 | 0.715 |
| warm start, width 64, 10 000 steps | 0.367 | 0.725 |
| warm start, width 256, 10 000 steps | 0.368 | 0.724 |

  At `b = 1` every network trained for 10 000 steps, cold or warm, width 64 or 256, reaches
  0.367 against 0.370 for the individuals: the gap there is training budget. At `b = 1.5` the
  best network of any width and length reaches 0.725 against 0.741, a residual of 0.017 (2 %)
  that neither width nor training length removes. The fraction of shifts at the bound (recorded for
  every follower in the final runs) shows where the two differ: `δ4` is at `±b` for all
  individuals (1.00) and nearly all network outputs (0.96 to 0.99), so the difference lies in
  `δ1`, whose optimal rule is interior for most individuals. There the fractions match too
  (individuals 44 % and 30 % at `b = 1` and `1.5`, cold network 47 % and 37 %, warm-started
  41 % and 29 %), so the networks' shortfall is in *which* individuals are pushed to the bound
  and by how much the others are shifted, not in a shrinkage away from the bound. A wider
  network
  cold-started is markedly worse (0.299 and 0.660), so the paper's width 64 is the right
  choice. "Approximation error of the neural follower" covers both findings; the first pass's
  attribution to the continuity of the network class was too strong.
- *Training length on several runs* (`outputs_appendix_sequential_long_*`, 5 runs,
  all-variable predictor, MSE objective, `b ∈ {1, 1.5}`). Training the cold-started network
  for 10 000 instead of 2000 steps gains 0.006 ± 0.002 (nonlinear) and 0.003 to 0.005
  (linear-Gaussian) on average and changes the picture in no run: runs in which the network
  found a good solution move from 0.01 to 0.03 below the individuals to 0.004 to 0.02 below,
  and the one run in five that is stuck at a poor solution (0.276 against 0.37 nonlinear, 0.345
  against 0.46 linear-Gaussian) stays there. The cold-started follower's gap is therefore mostly
  a matter of local optima, which more restarts or the warm start address, not of training
  length.

**Results, linear-Gaussian SCM** (10 runs, population predictors, 4000 individuals per run).
`PA(Y)` gives 0.1595 and `SB(Y)` gives 0.0834 at every bound under every follower, including
the omniscient one, exactly as §3.1 predicts for additive noise and the true regression
function. Under the prediction objective all followers coincide to four decimals for every
predictor. All-variable predictor, MSE objective:

| b | none | NN standard | NN +X1 | individual | NN fitted to rules | NN warm-started | omniscient |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.016 | 0.064 | 0.065 | 0.065 | 0.065 | 0.065 | 0.067 |
| 0.5 | 0.016 | 0.153 | 0.153 | 0.155 | 0.154 | 0.154 | 0.158 |
| 1.0 | 0.016 | 0.429 | 0.430 | 0.458 | 0.455 | 0.455 | 0.463 |
| 1.5 | 0.016 | 0.896 | 0.882 | 0.925 | 0.918 | 0.920 | 0.934 |

Paired over runs: individual − NN +X1 is +0.027 ± 0.014 at `b = 1` and +0.043 ± 0.021 at
`b = 1.5`; the large standard errors come from the bimodality of the neural follower (0.345 in
two runs against 0.45 in the other eight at `b = 1`; 0.76 against 0.91 at `b = 1.5`), while the
individuals vary only between 0.455 and 0.460. NN +X1 − NN standard is zero within this noise
(+0.002 ± 0.016, −0.014 ± 0.015). Here the fitted network alone is within 0.003 and 0.007 of
the individuals, and the warm start closes the gap almost completely: individual − warm-started
is +0.003 ± 0.000 at `b = 1` and +0.005 ± 0.000 at `b = 1.5`, in every run. The omniscient
individual gains only 0.006 and 0.009 over the admissible one. A nuance from the bound
fractions: the individuals' `δ1` is at the bound for 15 % of them at every `b`, the networks'
for 73 to 86 %, and yet the payoffs agree within 1 %; the payoff is flat in `δ1` for most
individuals, so the bound fraction of `δ1` says little about optimality.

![Both SCMs, 10 runs each: the population follower with `X1` as additional input of `δ4` (solid, filled markers) against the exact individual follower with the same information (dashed, open markers), on identical individuals. Rows: nonlinear SCM, linear-Gaussian SCM; columns: the two follower objectives. Bands: 95 % confidence intervals over runs.](further_experiments/outputs_appendix_sequential_standard/appendix_sequential_both.png)

The same figure with the warm-started network added is
`outputs_appendix_sequential_standard/appendix_sequential_both_warm.png`; the script is
`plot_appendix_sequential.py`.


**Reading.** The one-line change to §6.1 (`X1` as input of `δ4`) does not change any result,
so the paper's follower can be made the exact population pendant of the perfect-recall
individual for free. With the same information, individuals and population follower coincide
to the third or fourth decimal wherever the neural follower has an easy problem (prediction
objective, `PA(Y)`, `SB(Y)`, small bounds). Where the problem is hard (all variables, MSE,
`b ≥ 1`) the individuals are 5 to 10 % stronger. A warm start at the individuals' rules
reduces this to 2 to 3 % (nonlinear) and below 1 % (linear-Gaussian); five times longer training alone gains less than 1 %, because the
cold-started network sometimes sits in a poor local optimum; and on run 0 no network width or
training length tried removes the last 1 to 2 % at `b = 1.5`. In the paper I would say "up to the
approximation error of the neural follower" and give the sizes. The conclusions about `PA(Y)`
and `SB(Y)` do not depend on how the follower is implemented at all.

**Assessment of the plan, and what I would do.**

1. *The decision is right.* Adding `X1` to `δ4` costs one sentence in E.1 and makes the §6.1
   follower the exact population pendant of an individual with perfect recall; the sequential
   reading of §2.1 is the honest description of the two-target experiment. The paired table
   shows the change is invisible in the results, and `PA(Y)`, `SB(Y)` are bit-identical.
2. *§2.1 text.* The proposed replacement is correct. I would keep the one-shot statement as the
   main text (it is what most readers need) and add the sequential case as a remark of three
   sentences: targets set in a topological order, the individual recalls the earlier targets and
   their observations, and the same argument applies step by step, the first choice taking the
   optimal later choices into account. The remark must say that §6.1 *is* this sequential case
   (`δ4` reacts to `Y`), otherwise the paper's own experiment appears to violate the one-shot
   assumption. Both versions must say that the individual sees the noise `ε_j` of the target's
   own assignment; §6.1 uses this.
3. *This is the appendix experiment I would include.* It is the §6.1 experiment itself with the
   follower implemented at the individual level, so it needs no new setting; it shows that the
   conclusions of Fig. 2 do not depend on the neural attack and that the all-variable curve is,
   if anything, a lower bound on the harm; and it makes §2.1 concrete. Four sentences and one
   figure with two rows (nonlinear, linear-Gaussian) suffice; the warm start is one more
   sentence and pre-empts the question whether the neural follower is any good. Leaving `X3` out
   of the individual's observation set is a modeling choice, not a necessity (with `X3` the
   individuals gain a further 0.01 at `b = 1`, `outputs_information_ladder_standard`); I would
   leave it out, as the plan does, because it keeps the change to §6.1 at one input.
4. *Omniscient follower: not as a second experiment.* It is outside §2 (the intervention is
   correlated with `Y`'s noise, hidden confounding), and the effect it shows needs a paragraph to
   explain: with additive noise and the true regression function, `Y − f(X_{PA(Y)}) = ε_Y`
   whatever the follower does, so `PA(Y)` is unaffected even by an omniscient follower (the
   linear-Gaussian rows show this exactly); what moves `PA(Y)` and `SB(Y)` in the nonlinear case
   is the follower exploiting the estimation error of the MLP. That is a true and interesting
   caveat, but one sentence in the discussion ("if individuals could condition on their own
   outcome noise, the invariance of §3 fails and an estimated predictor can be exploited") is
   the right amount of text, not an appendix experiment.
5. *Information ladder: drop.* Its ordering is by construction and the magnitudes are small.
6. *The neural follower of §6.1 is limited by local optima, not by training length.* Five
   times more steps gain less than 0.01 and do not rescue the one run in five that is stuck at a
   poor solution; a warm start or more restarts would. I would not regenerate Fig. 2: the
   all-variable curve would move up by at most 10 % and `PA(Y)`, `SB(Y)` not at all, and the
   appendix experiment states the size of the gap, which is what a reader needs.
7. *A second experiment is not needed.* The linear-Gaussian row of the same figure already
   serves as a second check (exact flatness of `PA(Y)` and `SB(Y)` under every follower, exact
   agreement under the prediction objective). If something else is wanted, the only candidate I
   see that adds information is a follower with a *population* constraint (a shared budget
   `E[δ²] ≤ B` instead of a per-individual bound), where §2.1's equivalence fails because the
   constraint is not per individual; that would show the role of the assumption, but it is a
   different follower and needs its own description.

**Suggested appendix text** (four sentences, a footnote and the figure):

> To check the equivalence of §2.1 empirically, we repeat the experiment of §6.1 with the
> follower implemented at the individual level: each individual observes `(x2, ε1)` and chooses
> `δ1 ∈ [−b, b]`, then observes `(y, ε4)`, recalls `(x1, x2)` and chooses `δ4 ∈ [−b, b]`, each
> by grid search,¹ with the first choice taking the optimal second choice into account. The
> population follower of §6.1 and the individuals are evaluated on the same draws of the
> exogenous noise. Figure X shows that the leader's deployment MSE coincides for `PA(Y)`,
> `SB(Y)` and under the prediction objective, and that for the all-variable predictor under the
> MSE objective the individuals are a slightly stronger attacker (by 5 to 10 % at `b ≥ 1`).
> Initializing the networks at the individuals' decision rules and training further removes
> most of this difference; by §2.1 the remaining difference is the approximation error of the
> neural follower, the only possible source.
>
> ¹ Grid of 13 values in `[−b, b]`; expectations over unobserved noise by Monte Carlo with 16
> draws for the first choice and 64 for the second.

## 9. Files

- Library: `synthetic_experiments/individual_follower.py` (first implementation),
  `synthetic_experiments/individual_follower_simple.py` (compact; recommended),
  `synthetic_experiments/warm_start.py` (fit of the attack networks to individual rules).
  `synthetic_experiments/scm.py` gained two static methods (bitwise-identical samples) and the
  `x4_uses_x1` option of `AttackMechanisms`; `adversary.py` passes it through and accepts an
  initial network.
- Scripts: `run_individual_follower.py` (all follower comparisons; `--plot-only` rebuilds
  figures), `run_information_ladder.py`; commands in the README.
- Outputs (all untracked, nothing committed): `outputs_individual_follower_*` (Experiment 1),
  `outputs_implementation_comparison*` and `outputs_individual_only_*` (Experiment 2),
  `outputs_information_ladder_*`, `outputs_appendix_x4only_*`, `outputs_appendix_sequential_*`
  (Section 8).
