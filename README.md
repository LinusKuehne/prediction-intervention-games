# Prediction-Intervention Games and Invariant Sets

Code for the paper [*Prediction-Intervention Games and Invariant Sets*](https://arxiv.org/abs/2605.16828) (Kühne, Schur, and Peters, NeurIPS submission).

## Layout

- `src/stabilized_classification/` — library code for stabilized classification (used by §6.2).
- `scripts/synthetic/` — §6.1 synthetic-SCM experiments.
- `scripts/causal_chambers/` — §6.2 real-data experiments on the Causal Chambers light tunnel.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

## §6.1 — Synthetic experiments

Nonlinear and linear-Gaussian SCMs from §6.1 / Appendix F of the paper. The
package implements three predictors (parents, stable blanket, all variables)
trained on data from a known SCM, then evaluates each under its own best-response
adversary.

### SCM

The default graph is:

- `X2 -> X1 -> Y`
- `X2 -> Y`
- `Y -> X3`
- `X2 -> X4 <- Y`
- `X4 -> X5`

Passing `--X6` adds an extra observed variable with parents `X4` and `Y`:
`X4 -> X6 <- Y`.

Predictor sets:

- Parents: `{X1, X2}`
- Stable blanket: `{X1, X2, X3}`
- All variables: `{X1, X2, X3, X4, X5}` by default, and `{X1, X2, X3, X4, X5, X6}` with `--X6`

Directly intervened variables for the adversary: `{X1, X4}`.

The adversary adds bounded perturbations to the clean mechanisms for `X1` and
`X4` using neural networks of the form

```
b * tanh(h(parents, noise))
```

so the attacked mechanisms become "clean mechanism + bounded deviation". The
bound `b` is swept over multiple values, and `b = 0` recovers the clean SCM
exactly. `X4` now depends on both `Y` and `X2`, and the adversary can therefore
adjust `X4` based on `X2`. `X5` is not directly attacked; it shifts only through
the clean downstream mechanism from the attacked `X4`.

Passing `--disable-x1-intervention` turns off the direct intervention on `X1`,
so only `X4` is directly attacked. By default, the learned `X4` intervention can
depend on `(Y, X2, eps4)`; passing `--x4-uses-x1-x3` additionally lets it depend
on `(X1, X3)`.

There is also a cost-regularized attack mode (`--attack-mode cost`). In that
setting, the adversary optimizes its task objective while paying a penalty
proportional to the average squared perturbation size,
`E[delta_X1^2 + delta_X4^2]`, with regularization weight `c`. The x-axis is then
the cost `c` instead of the perturbation bound. A fixed
`--max-perturbation-bound` still caps the perturbation amplitudes for numerical
stability.

### Adversarial objectives

Three attack objectives are implemented:

1. `signed_error`: minimize `E[Y - f_S(X_S)]`
2. `mse`: maximize `E[(Y - f_S(X_S))^2]`
3. `prediction`: minimize `E[f_S(X_S)]`

Evaluation is always by MSE, and each method is evaluated **only on its own
optimized adversary**.

### Files (inside `scripts/synthetic/`)

- `run_experiments.py` — command-line entry point
- `plot_paper_figures.py` — assembles paper figures from saved CSVs
- `run_individual_follower.py` — compares individual and population followers for Appendix F.5
- `plot_appendix_individual_followers.py` — assembles the Appendix F.5 figure from saved CSVs
- `appendix_numbers.py` — computes the summary numbers quoted in the [Appendix F.5 details](#appendix-f5-of-the-paper-details-of-the-comparison-with-individual-followers) at the end of this README
- `decompose_risk.py` — splits the deployment MSE of PA(Y) and SB(Y) in the nonlinear SCM into population risk and regression error (numbers quoted in Appendix F.3)
- `synthetic_experiments/scm.py` — nonlinear/linear-Gaussian SCMs and bounded adversarial mechanisms
- `synthetic_experiments/models.py` — predictor training
- `synthetic_experiments/adversary.py` — adversary optimization
- `synthetic_experiments/experiment.py` — orchestration and CSV output
- `synthetic_experiments/plotting.py` — per-run plots of own-adversary MSE vs bound
- `unused/` — scripts not used for any paper figure (minimax training, generic per-run plotting)

### Reproducing paper figures

Run from `scripts/synthetic/`:

```bash
cd scripts/synthetic
```

Generate the result CSVs for all four paper configurations (order does not
matter; the train-size sweep takes the longest):

```bash
python run_experiments.py --output-dir outputs_lingauss --lineargaussian
python run_experiments.py --output-dir outputs_standard
python run_experiments.py --output-dir outputs_train-size-sweep --train-size-sweep 1000 4000 50000
python run_experiments.py --output-dir outputs_x4-uses-x1-x3 --x4-uses-x1-x3
```

Then assemble the paper figures (saved to `paper_plots/`):

```bash
python plot_paper_figures.py
```

This produces `lineargaussian_standard_row_grouped.pdf` (Fig. 2, main text) and
`sweep_and_x4_prediction.pdf` (Fig. 5, appendix). `plot_paper_figures.py`
hard-codes the four directory names above, so keep them as-is if you want the
plotting step to find the results.

### Output files (per `--output-dir`)

- `results_per_run.csv`
- `results_summary.csv`
- `mse_vs_bound_signed_error.png`
- `mse_vs_bound_mse.png`
- `mse_vs_cost_signed_error.png` (cost mode)
- `mse_vs_cost_mse.png` (cost mode)
- `config.json`

The per-run plots show mean attacked MSE with 95% confidence intervals across
runs.

### Other useful runs (not used for paper figures)

Quick smoke test:

```bash
python run_experiments.py \
  --output-dir outputs_smoke \
  --torch-num-threads 1 \
  --n-train 600 --n-val 200 --n-test 600 \
  --predictor-max-epochs 40 --predictor-patience 6 \
  --attack-steps 25 --attack-restarts 1 \
  --attack-batch-size 256 --attack-eval-size 1200 \
  --num-runs 1 \
  --disable-x1-intervention \
  --bounds 0.25 0.5 1.0 2.0
```

Fuller run with the standard nonlinear setup:

```bash
python run_experiments.py \
  --output-dir outputs_full \
  --torch-num-threads 1 \
  --n-train 4000 --n-val 1000 --n-test 4000 \
  --attack-steps 400 --attack-restarts 5 \
  --attack-batch-size 512 --attack-eval-size 10000 \
  --num-runs 3 \
  --bounds 0.25 0.5 1.0 2.0 4.0
```

Train-size sweep with Student-t (heavier-tailed) exogenous noise:

```bash
python run_experiments.py \
  --output-dir outputs_train_size_sweep_student_t \
  --noise-distribution student_t --student-t-df 3 \
  --train-size-sweep 1000 4000 20000 \
  --n-val 1000 --n-test 4000 \
  --attack-steps 400 --attack-restarts 5 \
  --num-runs 3 \
  --bounds 0.25 0.5 1.0 2.0
```

Cost-regularized run:

```bash
python run_experiments.py \
  --output-dir outputs_cost \
  --attack-mode cost \
  --torch-num-threads 1 \
  --n-train 4000 --n-val 1000 --n-test 4000 \
  --attack-steps 400 --attack-restarts 5 \
  --attack-batch-size 512 --attack-eval-size 10000 \
  --num-runs 3 \
  --max-perturbation-bound 2.0 \
  --costs 0.0 0.01 0.05 0.1 0.25 0.5 1.0
```

### Individual vs. population follower (Appendix F.5)

Appendix F.5 of the paper shows only the figure and a short summary.
The description of the two followers, the computation, the training of the population
follower, and the results were moved from the paper to the
[end of this README](#appendix-f5-of-the-paper-details-of-the-comparison-with-individual-followers).

`run_individual_follower.py` compares the paper's neural *population* follower
with *individual* followers (App. C.1 of the paper) on identical draws of
individuals. Every individual solves their own problem by grid search, with
Monte Carlo over not-yet-realized noise (backward induction: choose `delta_1`
after observing `(X2, eps1)`, then `delta_4` after observing `(X1, X2, Y, eps4)`).
Follower names: `nn_x1` is the population follower whose `delta_4` additionally
sees `X1` (`A_4 = {X1}`), `nn_standard` the follower of § 6.1, `ind_nox3` the
individual followers.

Final runs used in the paper. The networks are trained with the recipe of App. F.4 plus a bound
ramp (`--attack-bound-anneal-steps 1000`: the bound grows linearly from 0 to `b` over the first
1000 of the 2000 steps), which avoids runs in which the shift of `X4` collapses to one sign for
the whole population:

```bash
python run_individual_follower.py --output-dir outputs_appendix_anneal_standard \
  --followers nn_standard nn_x1 ind_nox3 --attack-bound-anneal-steps 1000 --num-runs 10 --n-jobs 5 \
  --torch-num-threads 3 --grid-size-x1 13 --mc-stage1 16 --mc-stage2 16 --mc-final 64
python run_individual_follower.py --lineargaussian --output-dir outputs_appendix_anneal_lingauss \
  --followers nn_standard nn_x1 ind_nox3 --attack-bound-anneal-steps 1000 --num-runs 10 --n-jobs 5 \
  --torch-num-threads 1 --grid-size-x1 13 --mc-stage1 16 --mc-stage2 16 --mc-final 64
# appendix figure in the style of Fig. 2 (written to paper_plots/)
mkdir -p paper_plots
python plot_appendix_individual_followers.py outputs_appendix_anneal_lingauss outputs_appendix_anneal_standard \
  --out paper_plots/appendix_individual_followers --max-bound 1.0
# numbers quoted in the Appendix F.5 details at the end of this README (effect of adding X1, relative differences, shifts of X4)
python appendix_numbers.py outputs_appendix_anneal_lingauss outputs_appendix_anneal_standard
```

`outputs_appendix_anneal_standard/` is gitignored; generate its results locally
with the command above before running the plotting and summary commands.

Outputs per `--output-dir`: `results_per_run.csv`, `results_summary.csv`,
`paired_differences.csv`, `config.json` and figures; `--plot-only` rebuilds summaries and
figures from a saved `results_per_run.csv`. `appendix_numbers.py` computes the numbers
in the Appendix F.5 details from these files. Library code:
`synthetic_experiments/individual_follower.py` (noise draws, outcomes, first implementation),
`synthetic_experiments/individual_follower_simple.py` (compact implementation used for
`ind_nox3`), `synthetic_experiments/warm_start.py` (diagnostic only).

### Expected qualitative behavior

- On the clean SCM, `all_variables` should usually achieve the lowest test MSE.
- As the intervention bound increases, `all_variables` should worsen under its own adversary.
- `stable_blanket` is typically flatter across bounds because it ignores the mutable descendants while still using the stable child `X3`.
- `parents` is also flat across bounds, but can be worse than `stable_blanket` because it discards `X3`.

### Risk decomposition (Appendix F.3)

`decompose_risk.py` trains the PA(Y) and SB(Y) predictors of Fig. 2 (same seeds as
`run_experiments.py`), optimizes the neural follower for each objective and bound, and
splits the resulting deployment MSE into the population risk `E_e[Var(Y | X_S)]` and the
regression error `E_e[(f_S - fhat_S)^2]`, where `f_S = E[Y | X_S]` is computed exactly
(closed form for PA(Y), numerical integration over `eps_Y` for SB(Y)). It also evaluates
constant shifts of `X1`.

```bash
python decompose_risk.py --output-dir outputs_risk_decomposition --num-runs 10 --n-jobs 5 --torch-num-threads 2
```

Outputs: `results_per_run.csv`, `summary.txt` (the tables behind the numbers quoted in
Appendix F.3) and `config.json`. `--summarize-only` recomputes `summary.txt` from a saved
`results_per_run.csv`; `--attack-bound-anneal-steps 1000` uses the follower of Appendix F.5
instead. The run takes about 15 minutes on an Apple M4 Pro.

## §6.2 — Causal Chambers experiments

Real-data prediction-intervention game on the Causal Chambers light tunnel
(`lt_mk2_standard`). The training data CSVs are already included under
`scripts/causal_chambers/data/`. Run from that directory:

```bash
cd scripts/causal_chambers
```

**Adversarial budget curves** (Fig. 4, `causal_chambers_game.png`):

```bash
python adversarial_follower_dspur.py
```

**Conditional independence test table** (Table in Appendix D.3, hidden
confounding):

```bash
python hidden_confounding_analysis.py
```

The CI test requires R with the [`weightedGCM`](https://cran.r-project.org/package=weightedGCM)
package installed and is called from Python via `rpy2`.

## Development

```bash
pre-commit install
ruff check .          # lint
ruff check --fix .    # auto-fix
ruff format .         # format
```

## Appendix F.5 of the paper: details of the comparison with individual followers

Appendix F.5 of the paper shows only the figure (`paper_plots/appendix_individual_followers.pdf`, Fig. 6 in the paper) and a short summary.
The following text was removed from the paper and is reproduced here verbatim; section, figure, and lemma numbers refer to the paper.

We compare the population follower of § 6.1, whose intervention functions $E_{\delta_1}$ and $E_{\delta_4}$ are modeled with neural networks, with individual followers who each choose shifts given their own realization of the observed variables.
As argued in § 2.1 and App. C.1, a population of individual followers is also modeled by a population follower whose interventions depend on the same variables and noise terms, so the leader's deployment MSE should be similar under both, although equally optimal interventions to minimize the prediction can yield different MSEs.
Differences can arise from the optimization of the population follower's neural networks, or from the grid and Monte Carlo approximations used for the individual followers.

**Individual followers.**
We use the SCMs of App. F.1 with the training DAG $\mathcal{G}_0$ in Fig. 1.
Each individual corresponds to one draw of the exogenous noise and chooses shifts $\delta_1, \delta_4 \in [-b, b]$ for their own $X_1$ and $X_4$.
They choose $\delta_1$ after observing $x_2$ and the current value $\tilde{x}_1$ of $X_1$, and $\delta_4$ after $Y$ has been realized and they have observed $x_1$, $x_2$, $y$, and the current value $\tilde{x}_4$; they observe neither $x_3$ nor $x_5$.
Since the noise in App. F.1 is additive, observing $\tilde{x}_1$ and $\tilde{x}_4$ given the parents is equivalent to observing $\varepsilon_1$ and $\varepsilon_4$, which is how we implement it.
Both followers know the SCM, its noise distributions, and the predictor $f_S$.
The individuals choose each shift to optimize their objective of App. F.2 given their observations, that is, in expectation over the remaining noise variables.

**Information of the two followers.**
In § 6.1, $E_{\delta_4}$ depends only on $(Y, X_2, \varepsilon_4)$.
Here, we also allow it to depend on $X_1$, which the individual knows from their earlier choice, so both followers have the same information.
The dependence on $X_1$ adds the edge $X_1 \to X_4$ to $\mathcal{G}_0$ (one of the dashed edges in Fig. 1).
In the notation of App. C.1, $O_1 = \{2\}$ and $O_4 = \{1, 2, Y\}$, so $A_1 = \varnothing$ and $A_4 = \{1\}$, which satisfies the assumption of Lem. 4.
With the training described below, adding $X_1$ as an input of $E_{\delta_4}$ leaves the population follower's mean deployment MSE for $\mathrm{PA}(Y)$ and $\mathrm{SB}(Y)$ unchanged and changes it for the all-variable predictor by at most about 0.002 over the displayed bounds (10 runs).

**Computation.**
We approximate each individual's optimal shifts $\delta_1$ and $\delta_4$ by searching over 13 equally spaced values in $[-b, b]$, estimating each candidate's expected objective by averaging over draws of the remaining noise variables.
For the final $\delta_4$ decision, we use 64 draws of $(\varepsilon_3, \varepsilon_5)$.
To evaluate a candidate $\delta_1$, we use backward induction as in App. C.1: we draw $(\varepsilon_Y, \varepsilon_4)$ 16 times, select the best $\delta_4$ on the grid in each draw using 16 draws of $(\varepsilon_3, \varepsilon_5)$, and average the resulting estimates.
Grid search is feasible because each individual makes two scalar choices for given inputs; the population follower instead optimizes functions of all inputs, represented by neural networks in App. F.2.
We evaluate the leader's deployment MSE under both followers on the same 4000 fresh draws of the exogenous noise.

**Training of the population follower.**
The networks are trained as in App. F.4, except that the bound is increased linearly from 0 to $b$ over the first 1000 of 2000 training steps and equals $b$ afterwards.[^anneal]

[^anneal]: For the all-variable predictor and a follower maximizing the MSE, the individual followers' shifts of $X_4$ equal $\pm b$, with varying sign.
    With the full bound from the start, the networks in some runs learn shifts of $X_4$ with the same sign for all individuals.
    Such shifts did not occur with the increasing bound.
    A likely reason is that, for a large bound, a shift with the same sign already yields most of the follower's gain, so the networks learn such a shift first and their outputs saturate at $\pm b$.

**Results.**
Fig. 6 shows that the leader's mean deployment MSE under the two followers agrees to within about 1% for $\mathrm{PA}(Y)$ and $\mathrm{SB}(Y)$ under both follower objectives, and for the all-variable predictor when the follower minimizes the expected prediction.
When the follower maximizes the leader's MSE, the individual followers reach a mean deployment MSE for the all-variable predictor that is less than 1% higher than under the population follower in the linear-Gaussian SCM and about 2 to 5% higher in the nonlinear SCM.
A likely reason is that, for the all-variable predictor and the MSE objective, the optimal shift of $X_4$ is $b$ for some individuals and $-b$ for others, which is more difficult to learn than the shift $-b$ that is optimal for almost all individuals under the prediction objective.
In the linear-Gaussian SCM, the sign of the optimal shift of $X_4$ is the sign of a linear function of the individual's observations, which the networks learn well.
In the nonlinear SCM, the sign is a more complicated function of the observations, and the networks' shifts of $X_1$ also differ from those of the individual followers.
Since the population follower of § 6.1 is trained without the increasing bound, these results suggest that, when the follower maximizes the MSE, the all-variable curves in Fig. 2 slightly underestimate the deployment MSE under an optimal follower.
