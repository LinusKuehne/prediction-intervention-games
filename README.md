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

Nonlinear and linear-Gaussian SCMs from §6.1 / Appendix E of the paper. The
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

### Individual vs. population follower (appendix E.5; see appendix_experiment_report.md)

`run_individual_follower.py` compares the paper's neural *population* follower
with *individual* followers (§ 2.1 of the paper) on identical draws of
individuals. Every individual solves their own problem by grid search, with
Monte Carlo over not-yet-realized noise (backward induction: choose `delta_1`
after observing `(X2, eps1)`, then `delta_4` after observing `(X1, X2, Y, eps4)`).
Follower names: `nn_x1` is the population follower whose `delta_4` additionally
sees `X1` (`A_4 = {X1}`), `nn_standard` the follower of § 6.1, `ind_nox3` the
individual followers.

Final runs used in the paper. The networks are trained with the recipe of § E.4 plus a bound
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
# appendix figure in the style of Fig. 2 (written to the paper's images folder)
python plot_appendix_individual_followers.py outputs_appendix_anneal_lingauss outputs_appendix_anneal_standard \
  --out ../../prediction_intervention_games/images/appendix_individual_followers --max-bound 1.0
# numbers quoted in the text of E.5 (effect of adding X1, relative differences, shifts of X4)
python appendix_numbers.py outputs_appendix_anneal_lingauss outputs_appendix_anneal_standard
```

(The stored folders were produced with `--followers nn_standard nn_x1`; the `ind_nox3` rows,
which are deterministic given predictor, noise draws and seed, were copied from the previous
run and the summaries rebuilt with `--plot-only`. The commands above recompute them, which
costs about 50 extra minutes in the nonlinear SCM.)

Outputs per `--output-dir`: `results_per_run.csv`, `results_summary.csv`,
`paired_differences.csv`, `config.json` and figures; `--plot-only` rebuilds summaries and
figures from a saved `results_per_run.csv`. `appendix_numbers.py` computes the numbers in the text from these
files; passing `../../further_experiments/outputs_appendix_final_{lingauss,standard}` (training
without the bound ramp) instead gives the runs in which the shift of `X4` has one sign. Library code:
`synthetic_experiments/individual_follower.py` (noise draws, outcomes, first implementation),
`synthetic_experiments/individual_follower_simple.py` (compact implementation used for
`ind_nox3`), `synthetic_experiments/warm_start.py` (diagnostic only).

Earlier and exploratory runs (first implementation and its validation, one-shot validation of
§ 2.1, sequential runs with warm start and omniscient follower, the training-recipe study, the
information ladder) and all their outputs are in `further_experiments/` (gitignored); see
`further_experiments/README.md`.

### Expected qualitative behavior

- On the clean SCM, `all_variables` should usually achieve the lowest test MSE.
- As the intervention bound increases, `all_variables` should worsen under its own adversary.
- `stable_blanket` is typically flatter across bounds because it ignores the mutable descendants while still using the stable child `X3`.
- `parents` is also flat across bounds, but can be worse than `stable_blanket` because it discards `X3`.

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
