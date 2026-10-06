"""Population (NN) followers vs. individual followers on identical individuals.

For every trained predictor f_S, follower objective and intervention bound b,
four followers are run on the *same* draw of exogenous noises (one noise draw
per run), so that all comparisons are paired:

  none         no intervention (reference)
  nn_standard  paper's population follower: delta_1 = g(X2, eps1),
               delta_4 = g(Y, X2, eps4)
  nn_x1x3      population follower with delta_4 = g(Y, X2, eps4, X1, X3)
               (paper's ``--x4-uses-x1-x3`` setting)
  ind_seq      sequential individual follower (Section 2.1 of the paper, made
               sequential): every individual chooses delta_1 after observing
               (x2, eps1) and delta_4 after observing (x1, x2, y, x3, eps4),
               by grid search + Monte Carlo over not-yet-realized noise.
               Same information as nn_x1x3, but no function class.
  ind_omni     omniscient individual follower (knows all own noises); upper
               bound, outside the paper's framework.

Outputs (in --output-dir): results_per_run.csv, results_summary.csv,
paired_differences.csv, config.json and figures.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import pandas as pd
import torch
from joblib import Parallel, delayed
from synthetic_experiments.adversary import optimize_attack
from synthetic_experiments.experiment import (
    ExperimentConfig,
    _train_all_predictors,
    configure_torch,
)
from synthetic_experiments.individual_follower import (
    IndividualFollowerConfig,
    _outcome,
    draw_noises,
    omniscient_individual_follower,
    population_follower_outcome,
    sequential_individual_follower,
)
from synthetic_experiments.individual_follower_simple import (
    MSE,
    PREDICTION,
    individual_follower,
)
from synthetic_experiments.plotting import (
    LABELS,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
)
from synthetic_experiments.scm import (
    LinearGaussianStableBlanketSCM,
    NonlinearStableBlanketSCM,
)
from synthetic_experiments.warm_start import fit_attack_to_rules

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PLOT_ONLY = False
FOLLOWERS = ("nn_standard", "nn_x1x3", "ind_seq", "ind_omni")
# Compact re-implementation (individual_follower_simple.py); opt-in via --followers.
# ind_simple = exact (backward induction); ind_greedy = delta_1 chosen ignoring delta_4.
# Both use --grid-size-x1 as grid size, --mc-stage1 / --mc-stage2 as Monte Carlo sizes.
OPTIONAL_FOLLOWERS = (
    "ind_simple",
    "ind_greedy",
    "ind_pa4",
    "nn_x1",
    "ind_nox3",
    "nn_x1_warm",
    "nn_x1_fit",
    "nn_x1_long",
    "nn_x1_tuned",
)
# nn_x1_long: nn_x1 trained for --attack-steps-long steps instead of --attack-steps.
# nn_x1_tuned: nn_x1 with the training recipe given by the --tuned-* options (restarts,
# steps, lr schedule, width/depth/slope, continuation in b from the previous bound's
# network); bounds must be given in increasing order for the continuation.
# nn_x1: population NN whose delta_4 additionally sees X1 (A_4 = {X1}); ind_nox3: its
# exact individual pendant (sequential, delta_4 observes (x1, x2, y, eps4), not x3).
# nn_x1_warm: nn_x1 warm-started at ind_nox3's rules and then trained as in the paper;
# it also records nn_x1_fit (the fitted networks before training).
# ind_pa4: individual who observes only PA(X4) = (Y, X2) and eps4 (X1, X3 integrated
# out; X1 by importance sampling). Exact pendant of nn_standard when only X4 is
# intervened on (--disable-x1-intervention).
FOLLOWER_LABELS = {
    "none": "no intervention",
    "nn_standard": r"population NN, $\delta_4(Y,X_2,\varepsilon_4)$",
    "nn_x1x3": r"population NN, $\delta_4(Y,X_2,\varepsilon_4,X_1,X_3)$",
    "ind_seq": "individual (sequential, grid+MC)",
    "ind_omni": "individual (omniscient)",
    "ind_simple": r"individual, observes $(X_1, X_2, X_3, Y)$",
    "ind_greedy": "individual (greedy step 1)",
    "ind_pa4": r"individual, observes $(Y, X_2)$ only",
    "nn_x1": r"population NN, $\delta_4(Y,X_2,\varepsilon_4,X_1)$",
    "ind_nox3": r"individual, $\delta_4$ observes $(X_1,X_2,Y)$",
    "nn_x1_warm": "population NN, warm-started at the individuals' rules",
    "nn_x1_fit": "NN fitted to the individuals' rules (no training)",
    "nn_x1_long": r"population NN, $\delta_4(Y,X_2,\varepsilon_4,X_1)$, 5x longer training",
    "nn_x1_tuned": r"population NN, $\delta_4(Y,X_2,\varepsilon_4,X_1)$, improved training",
}
FOLLOWER_STYLES = {  # (linestyle, marker, fillstyle)
    "nn_standard": (":", "o", "full"),
    "nn_x1x3": ("-", "s", "full"),
    "ind_seq": ("--", "^", "none"),
    "ind_omni": ("-.", "D", "none"),
    "ind_simple": ((0, (1, 1)), "v", "none"),
    "ind_greedy": ((0, (5, 2, 1, 2)), "x", "full"),
    "ind_pa4": ("--", "o", "none"),
    "nn_x1": ("-", "P", "full"),
    "ind_nox3": ("--", "^", "none"),
    "nn_x1_warm": ("-.", "X", "full"),
    "nn_x1_fit": ((0, (1, 1)), "*", "full"),
    "nn_x1_long": ("-", "D", "full"),
    "nn_x1_tuned": ("-", "s", "full"),
}


@dataclass
class RunConfig:
    output_dir: str = "outputs_individual_follower"
    lineargaussian: bool = False
    num_runs: int = 3
    n_jobs: int = 3
    torch_num_threads: int = 4
    n_train: int = 50000
    n_val: int = 3000
    n_test: int = 4000
    predictor_hidden_dim: int = 256
    predictor_depth: int = 5
    predictor_lr: float = 1e-3
    predictor_batch_size: int = 512
    predictor_max_epochs: int = 500
    predictor_patience: int = 20
    predictor_weight_decay: float = 1e-5
    attack_hidden_dim: int = 64
    attack_lr: float = 1e-3
    attack_steps: int = 2000
    attack_steps_long: int = 10000  # for the follower nn_x1_long
    attack_bound_anneal_steps: int = (
        0  # bound grows from 0 to b over these steps (nn_* followers)
    )
    # training recipe of the follower nn_x1_tuned
    tuned_restarts: int = 5
    tuned_steps: int = 2000
    tuned_hidden_dim: int = 64
    tuned_depth: int = 2
    tuned_slope: float = 1.0
    tuned_lr_schedule: str = "constant"
    tuned_continuation: bool = True
    attack_batch_size: int = 512
    attack_restarts: int = 5
    attack_eval_size: int = 10000
    intervene_on_x1: bool = True
    noise_distribution: str = "gaussian"
    student_t_df: int = 3
    bounds: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5)
    objectives: tuple[str, ...] = ("mse", "prediction")
    followers: tuple[str, ...] = FOLLOWERS
    methods: tuple[str, ...] = ("parents", "stable_blanket", "all_variables")
    n_individuals: int = 4000
    grid_size_x1: int = 25
    grid_size_x4: int = 25
    mc_stage1: int = 32
    mc_stage2: int = 6
    mc_final: int = 64  # compact followers: draws for the final delta_4 choice
    omniscient_grid_size: int = 41
    chunk_individuals: int = 64


def parse_args() -> RunConfig:
    d = RunConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--output-dir", type=str, default=d.output_dir)
    p.add_argument("--lineargaussian", action="store_true")
    p.add_argument("--num-runs", type=int, default=d.num_runs)
    p.add_argument("--n-jobs", type=int, default=d.n_jobs)
    p.add_argument("--torch-num-threads", type=int, default=d.torch_num_threads)
    p.add_argument("--n-train", type=int, default=d.n_train)
    p.add_argument("--n-val", type=int, default=d.n_val)
    p.add_argument("--n-test", type=int, default=d.n_test)
    p.add_argument("--predictor-hidden-dim", type=int, default=d.predictor_hidden_dim)
    p.add_argument("--predictor-depth", type=int, default=d.predictor_depth)
    p.add_argument("--predictor-lr", type=float, default=d.predictor_lr)
    p.add_argument("--predictor-batch-size", type=int, default=d.predictor_batch_size)
    p.add_argument("--predictor-max-epochs", type=int, default=d.predictor_max_epochs)
    p.add_argument("--predictor-patience", type=int, default=d.predictor_patience)
    p.add_argument(
        "--predictor-weight-decay", type=float, default=d.predictor_weight_decay
    )
    p.add_argument("--attack-hidden-dim", type=int, default=d.attack_hidden_dim)
    p.add_argument("--attack-lr", type=float, default=d.attack_lr)
    p.add_argument("--attack-steps", type=int, default=d.attack_steps)
    p.add_argument("--attack-steps-long", type=int, default=d.attack_steps_long)
    p.add_argument(
        "--attack-bound-anneal-steps", type=int, default=d.attack_bound_anneal_steps
    )
    p.add_argument("--tuned-restarts", type=int, default=d.tuned_restarts)
    p.add_argument("--tuned-steps", type=int, default=d.tuned_steps)
    p.add_argument("--tuned-hidden-dim", type=int, default=d.tuned_hidden_dim)
    p.add_argument("--tuned-depth", type=int, default=d.tuned_depth)
    p.add_argument("--tuned-slope", type=float, default=d.tuned_slope)
    p.add_argument(
        "--tuned-lr-schedule",
        choices=["constant", "cosine"],
        default=d.tuned_lr_schedule,
    )
    p.add_argument("--no-tuned-continuation", action="store_true")
    p.add_argument("--attack-batch-size", type=int, default=d.attack_batch_size)
    p.add_argument("--attack-restarts", type=int, default=d.attack_restarts)
    p.add_argument("--attack-eval-size", type=int, default=d.attack_eval_size)
    p.add_argument("--disable-x1-intervention", action="store_true")
    p.add_argument(
        "--noise-distribution", choices=["gaussian", "student_t"], default="gaussian"
    )
    p.add_argument("--student-t-df", type=int, default=d.student_t_df)
    p.add_argument("--bounds", type=float, nargs="+", default=list(d.bounds))
    p.add_argument(
        "--objectives",
        nargs="+",
        choices=["signed_error", "mse", "prediction"],
        default=list(d.objectives),
    )
    p.add_argument(
        "--followers",
        nargs="+",
        choices=list(FOLLOWERS) + list(OPTIONAL_FOLLOWERS),
        default=list(FOLLOWERS),
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=list(d.methods),
        default=list(d.methods),
        help="Predictors to evaluate (all three are always trained).",
    )
    p.add_argument("--n-individuals", type=int, default=d.n_individuals)
    p.add_argument("--grid-size-x1", type=int, default=d.grid_size_x1)
    p.add_argument("--grid-size-x4", type=int, default=d.grid_size_x4)
    p.add_argument("--mc-stage1", type=int, default=d.mc_stage1)
    p.add_argument("--mc-stage2", type=int, default=d.mc_stage2)
    p.add_argument("--mc-final", type=int, default=d.mc_final)
    p.add_argument("--omniscient-grid-size", type=int, default=d.omniscient_grid_size)
    p.add_argument("--chunk-individuals", type=int, default=d.chunk_individuals)
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip computation; rebuild summary, paired differences and figures "
        "from results_per_run.csv in --output-dir.",
    )
    a = p.parse_args()
    global PLOT_ONLY
    PLOT_ONLY = a.plot_only
    return RunConfig(
        output_dir=a.output_dir,
        lineargaussian=a.lineargaussian,
        num_runs=a.num_runs,
        n_jobs=a.n_jobs,
        torch_num_threads=a.torch_num_threads,
        n_train=a.n_train,
        n_val=a.n_val,
        n_test=a.n_test,
        predictor_hidden_dim=a.predictor_hidden_dim,
        predictor_depth=a.predictor_depth,
        predictor_lr=a.predictor_lr,
        predictor_batch_size=a.predictor_batch_size,
        predictor_max_epochs=a.predictor_max_epochs,
        predictor_patience=a.predictor_patience,
        predictor_weight_decay=a.predictor_weight_decay,
        attack_hidden_dim=a.attack_hidden_dim,
        attack_lr=a.attack_lr,
        attack_steps=a.attack_steps,
        attack_steps_long=a.attack_steps_long,
        attack_bound_anneal_steps=a.attack_bound_anneal_steps,
        tuned_restarts=a.tuned_restarts,
        tuned_steps=a.tuned_steps,
        tuned_hidden_dim=a.tuned_hidden_dim,
        tuned_depth=a.tuned_depth,
        tuned_slope=a.tuned_slope,
        tuned_lr_schedule=a.tuned_lr_schedule,
        tuned_continuation=not a.no_tuned_continuation,
        attack_batch_size=a.attack_batch_size,
        attack_restarts=a.attack_restarts,
        attack_eval_size=a.attack_eval_size,
        intervene_on_x1=not a.disable_x1_intervention,
        noise_distribution=a.noise_distribution,
        student_t_df=a.student_t_df,
        bounds=tuple(a.bounds),
        objectives=tuple(a.objectives),
        followers=tuple(a.followers),
        methods=tuple(a.methods),
        n_individuals=a.n_individuals,
        grid_size_x1=a.grid_size_x1,
        grid_size_x4=a.grid_size_x4,
        mc_stage1=a.mc_stage1,
        mc_stage2=a.mc_stage2,
        mc_final=a.mc_final,
        omniscient_grid_size=a.omniscient_grid_size,
        chunk_individuals=a.chunk_individuals,
    )


def _experiment_config(cfg: RunConfig) -> ExperimentConfig:
    return ExperimentConfig(
        torch_num_threads=cfg.torch_num_threads,
        n_train=cfg.n_train,
        n_val=cfg.n_val,
        n_test=cfg.n_test,
        predictor_hidden_dim=cfg.predictor_hidden_dim,
        predictor_depth=cfg.predictor_depth,
        predictor_lr=cfg.predictor_lr,
        predictor_batch_size=cfg.predictor_batch_size,
        predictor_max_epochs=cfg.predictor_max_epochs,
        predictor_patience=cfg.predictor_patience,
        predictor_weight_decay=cfg.predictor_weight_decay,
        intervene_on_x1=cfg.intervene_on_x1,
        noise_distribution=cfg.noise_distribution,  # type: ignore[arg-type]
        student_t_df=cfg.student_t_df,
        lineargaussian=cfg.lineargaussian,
    )


def run_single(cfg: RunConfig, run_id: int) -> list[dict]:
    configure_torch(cfg.torch_num_threads)
    scm_cls = (
        LinearGaussianStableBlanketSCM
        if cfg.lineargaussian
        else NonlinearStableBlanketSCM
    )
    scm = scm_cls(
        intervene_on_x1=cfg.intervene_on_x1,
        noise_distribution=cfg.noise_distribution,  # type: ignore[arg-type]
        student_t_df=cfg.student_t_df,
    )
    t0 = time.time()
    predictors = _train_all_predictors(
        _experiment_config(cfg), scm, run_id, cfg.n_train
    )
    print(f"[run {run_id}] predictors trained in {time.time() - t0:.0f}s", flush=True)
    predictors = [p for p in predictors if p.name in cfg.methods]

    # One fixed draw of individuals per run; every follower acts on these.
    generator = torch.Generator(device=scm.device)
    generator.manual_seed(100_000 + run_id)
    noises = draw_noises(scm, cfg.n_individuals, generator)
    ind_cfg = IndividualFollowerConfig(
        grid_size_x1=cfg.grid_size_x1,
        grid_size_x4=cfg.grid_size_x4,
        mc_stage1=cfg.mc_stage1,
        mc_stage2=cfg.mc_stage2,
        omniscient_grid_size=cfg.omniscient_grid_size,
        chunk_individuals=cfg.chunk_individuals,
    )

    rows: list[dict] = []

    def record(predictor, objective, bound, outcome, seconds, **extra) -> None:
        rows.append(
            {
                "run_id": run_id,
                "method": predictor.name,
                "subset_size": len(predictor.subset_indices),
                "objective": objective,
                "bound": float(bound),
                "follower": outcome.follower,
                "clean_test_mse": float(predictor.clean_test_mse),
                "attacked_mse": outcome.attacked_mse,
                "follower_objective": outcome.follower_objective,
                "intervention_strength": outcome.intervention_strength,
                "mean_delta_x1": float(outcome.delta_x1.mean()),
                "mean_abs_delta_x1": float(outcome.delta_x1.abs().mean()),
                "mean_delta_x4": float(outcome.delta_x4.mean()),
                "mean_abs_delta_x4": float(outcome.delta_x4.abs().mean()),
                # fraction of shifts within 5% of the bound (bang-bang diagnostic)
                "frac_bound_x1": float(
                    (outcome.delta_x1.abs() > 0.95 * float(bound)).float().mean()
                ),
                "frac_bound_x4": float(
                    (outcome.delta_x4.abs() > 0.95 * float(bound)).float().mean()
                ),
                "seconds": seconds,
                **extra,
            }
        )

    for predictor in predictors:
        for objective in cfg.objectives:
            prev_tuned = None  # network of the previous bound (continuation)
            for bound in cfg.bounds:
                t0 = time.time()
                clean = population_follower_outcome(
                    scm, predictor, objective, noises, None, name="none"
                )
                record(predictor, objective, bound, clean, time.time() - t0)
                rules = None  # (delta1, delta4) of ind_nox3, for the warm start
                for follower in cfg.followers:
                    t0 = time.time()
                    extra: dict = {}
                    if follower in ("nn_standard", "nn_x1x3", "nn_x1", "nn_x1_long"):
                        result = optimize_attack(
                            scm=scm,
                            predictor=predictor,
                            bound=float(bound),
                            cost=None,
                            intervene_on_x1=cfg.intervene_on_x1,
                            simple=False,
                            x4_uses_x1_x3=(follower == "nn_x1x3"),
                            x4_uses_x1=(follower in ("nn_x1", "nn_x1_long")),
                            objective=objective,  # type: ignore[arg-type]
                            restarts=cfg.attack_restarts,
                            steps=(
                                cfg.attack_steps_long
                                if follower == "nn_x1_long"
                                else cfg.attack_steps
                            ),
                            batch_size=cfg.attack_batch_size,
                            hidden_dim=cfg.attack_hidden_dim,
                            lr=cfg.attack_lr,
                            eval_size=cfg.attack_eval_size,
                            seed=run_id,
                            bound_anneal_steps=cfg.attack_bound_anneal_steps,
                        )
                        outcome = population_follower_outcome(
                            scm, predictor, objective, noises, result.attack, follower
                        )
                        extra = {
                            "nn_attacked_test_mse": result.attacked_test_mse,
                            "nn_attack_objective_value": result.best_objective_value,
                        }
                    elif follower == "ind_seq":
                        outcome = sequential_individual_follower(
                            scm,
                            predictor,
                            objective,  # type: ignore[arg-type]
                            noises,
                            float(bound),
                            ind_cfg,
                            seed=10_000 * run_id + 7,
                            name=follower,
                        )
                    elif follower == "ind_omni":
                        outcome = omniscient_individual_follower(
                            scm,
                            predictor,
                            objective,  # type: ignore[arg-type]
                            noises,
                            float(bound),
                            ind_cfg,
                            name=follower,
                        )
                    elif follower in (
                        "ind_simple",
                        "ind_greedy",
                        "ind_pa4",
                        "ind_nox3",
                    ):
                        x, y, d1, d4 = individual_follower(
                            scm,
                            predictor,
                            MSE if objective == "mse" else PREDICTION,
                            noises,
                            float(bound),
                            grid_size=cfg.grid_size_x1,
                            n_mc=cfg.mc_stage1,
                            n_mc_step2=cfg.mc_stage2,
                            n_mc_final=cfg.mc_final,
                            greedy=(follower == "ind_greedy"),
                            observe_x1=(follower != "ind_pa4"),
                            observe_x3=(follower not in ("ind_pa4", "ind_nox3")),
                            seed=10_000 * run_id + 7,
                        )
                        outcome = _outcome(follower, predictor, objective, x, y, d1, d4)
                        if follower == "ind_nox3":
                            rules = (d1, d4)
                    elif follower == "nn_x1_tuned":
                        result = optimize_attack(
                            scm=scm,
                            predictor=predictor,
                            bound=float(bound),
                            cost=None,
                            intervene_on_x1=cfg.intervene_on_x1,
                            simple=False,
                            x4_uses_x1_x3=False,
                            x4_uses_x1=True,
                            objective=objective,  # type: ignore[arg-type]
                            restarts=cfg.tuned_restarts,
                            steps=cfg.tuned_steps,
                            batch_size=cfg.attack_batch_size,
                            hidden_dim=cfg.tuned_hidden_dim,
                            lr=cfg.attack_lr,
                            eval_size=cfg.attack_eval_size,
                            seed=run_id,
                            depth=cfg.tuned_depth,
                            slope=cfg.tuned_slope,
                            lr_schedule=cfg.tuned_lr_schedule,
                            init_candidates=(
                                [prev_tuned]
                                if cfg.tuned_continuation and prev_tuned is not None
                                else ()
                            ),
                        )
                        prev_tuned = result.attack
                        outcome = population_follower_outcome(
                            scm, predictor, objective, noises, result.attack, follower
                        )
                        extra = {
                            "nn_attacked_test_mse": result.attacked_test_mse,
                            "nn_attack_objective_value": result.best_objective_value,
                        }
                    elif follower == "nn_x1_fit":
                        continue  # recorded together with nn_x1_warm
                    elif follower == "nn_x1_warm":
                        if rules is None:  # ind_nox3 not requested: compute its rules
                            _, _, d1, d4 = individual_follower(
                                scm,
                                predictor,
                                MSE if objective == "mse" else PREDICTION,
                                noises,
                                float(bound),
                                grid_size=cfg.grid_size_x1,
                                n_mc=cfg.mc_stage1,
                                n_mc_step2=cfg.mc_stage2,
                                n_mc_final=cfg.mc_final,
                                observe_x3=False,
                                seed=10_000 * run_id + 7,
                            )
                            rules = (d1, d4)
                        attack0 = fit_attack_to_rules(
                            scm,
                            noises,
                            *rules,
                            float(bound),
                            hidden_dim=cfg.attack_hidden_dim,
                            seed=run_id,
                        )
                        fit = population_follower_outcome(
                            scm, predictor, objective, noises, attack0, "nn_x1_fit"
                        )
                        record(predictor, objective, bound, fit, time.time() - t0)
                        print(
                            f"[run {run_id}] {predictor.name:15s} {objective:10s} "
                            f"b={bound:<5g} {'nn_x1_fit':12s} mse={fit.attacked_mse:.4f} "
                            f"obj={fit.follower_objective:+.4f} "
                            f"({time.time() - t0:.0f}s)",
                            flush=True,
                        )
                        result = optimize_attack(
                            scm=scm,
                            predictor=predictor,
                            bound=float(bound),
                            cost=None,
                            intervene_on_x1=cfg.intervene_on_x1,
                            simple=False,
                            x4_uses_x1_x3=False,
                            x4_uses_x1=True,
                            objective=objective,  # type: ignore[arg-type]
                            restarts=1,
                            steps=cfg.attack_steps,
                            batch_size=cfg.attack_batch_size,
                            hidden_dim=cfg.attack_hidden_dim,
                            lr=cfg.attack_lr,
                            eval_size=cfg.attack_eval_size,
                            seed=run_id,
                            init_attack=attack0,
                        )
                        outcome = population_follower_outcome(
                            scm, predictor, objective, noises, result.attack, follower
                        )
                        extra = {
                            "nn_attacked_test_mse": result.attacked_test_mse,
                            "nn_attack_objective_value": result.best_objective_value,
                        }
                    else:
                        raise ValueError(follower)
                    seconds = time.time() - t0
                    record(predictor, objective, bound, outcome, seconds, **extra)
                    print(
                        f"[run {run_id}] {predictor.name:15s} {objective:10s} b={bound:<5g} "
                        f"{follower:12s} mse={outcome.attacked_mse:.4f} "
                        f"obj={outcome.follower_objective:+.4f} ({seconds:.0f}s)",
                        flush=True,
                    )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "objective", "bound", "follower"]
    summary = (
        results.groupby(keys)
        .agg(
            attacked_mse_mean=("attacked_mse", "mean"),
            attacked_mse_std=("attacked_mse", "std"),
            follower_objective_mean=("follower_objective", "mean"),
            follower_objective_std=("follower_objective", "std"),
            intervention_strength_mean=("intervention_strength", "mean"),
            frac_bound_x1_mean=("frac_bound_x1", "mean"),
            frac_bound_x4_mean=("frac_bound_x4", "mean"),
            n_runs=("run_id", "nunique"),
        )
        .reset_index()
    )
    for prefix in ("attacked_mse", "follower_objective"):
        summary[f"{prefix}_std"] = summary[f"{prefix}_std"].fillna(0.0)
        sem = summary[f"{prefix}_std"] / summary["n_runs"].clip(lower=1).pow(0.5)
        summary[f"{prefix}_sem"] = sem
        summary[f"{prefix}_ci95_radius"] = 1.96 * sem
    return summary.sort_values(keys)


def paired_differences(results: pd.DataFrame) -> pd.DataFrame:
    """Per-run paired differences between followers, then mean / sem over runs."""
    pairs = [
        ("ind_seq", "nn_x1x3"),  # same information: should be ~0 (Section 2.1)
        ("ind_seq", "nn_standard"),
        ("nn_x1x3", "nn_standard"),
        ("ind_omni", "ind_seq"),  # more information: >= 0 in follower objective
        ("ind_simple", "ind_seq"),  # same algorithm, compact implementation
        ("ind_greedy", "ind_seq"),
        ("ind_simple", "nn_x1x3"),
        ("ind_greedy", "nn_x1x3"),
        ("ind_pa4", "nn_standard"),  # same information when only X4 is intervened on
        ("ind_nox3", "nn_x1"),  # same information (A_4 = {X1}): optimization error
        ("nn_x1", "nn_standard"),  # effect of adding X1 to delta_4
        ("nn_x1_warm", "nn_x1"),  # warm start closes the gap?
        ("ind_nox3", "nn_x1_warm"),
        ("nn_x1_fit", "ind_nox3"),  # representation error of the fitted networks
        ("ind_omni", "ind_nox3"),
        ("nn_x1_long", "nn_x1"),  # effect of 5x longer training
        ("ind_nox3", "nn_x1_long"),
        ("nn_x1_tuned", "nn_x1"),  # effect of the improved training recipe
        ("ind_nox3", "nn_x1_tuned"),
    ]
    keys = ["run_id", "method", "objective", "bound"]
    wide = results.pivot_table(
        index=keys, columns="follower", values=["attacked_mse", "follower_objective"]
    )
    out = []
    for a, b in pairs:
        if (
            a not in wide["attacked_mse"].columns
            or b not in wide["attacked_mse"].columns
        ):
            continue
        diff = pd.DataFrame(
            {
                "d_attacked_mse": wide["attacked_mse"][a] - wide["attacked_mse"][b],
                "d_follower_objective": wide["follower_objective"][a]
                - wide["follower_objective"][b],
            }
        ).reset_index()
        diff["pair"] = f"{a} - {b}"
        out.append(diff)
    if not out:
        return pd.DataFrame(
            columns=[
                "pair",
                "method",
                "objective",
                "bound",
                "d_mse_mean",
                "d_mse_sem",
                "d_obj_mean",
                "d_obj_sem",
                "n_runs",
            ]
        )
    diffs = pd.concat(out, ignore_index=True)
    agg = (
        diffs.groupby(["pair", "method", "objective", "bound"])
        .agg(
            d_mse_mean=("d_attacked_mse", "mean"),
            d_mse_sem=("d_attacked_mse", lambda s: s.std(ddof=1) / len(s) ** 0.5),
            d_obj_mean=("d_follower_objective", "mean"),
            d_obj_sem=("d_follower_objective", lambda s: s.std(ddof=1) / len(s) ** 0.5),
            n_runs=("run_id", "nunique"),
        )
        .reset_index()
    )
    return agg.fillna(0.0)


def _plot_lines(ax, stats, method, followers, with_label: bool) -> None:
    color = METHOD_COLORS[method]
    for follower in followers:
        sub = stats[(stats["method"] == method) & (stats["follower"] == follower)]
        sub = sub.sort_values("bound")
        if sub.empty:
            continue
        ls, marker, fill = FOLLOWER_STYLES[follower]
        ax.plot(
            sub["bound"],
            sub["attacked_mse_mean"],
            color=color,
            linestyle=ls,
            marker=marker,
            fillstyle=fill,
            markersize=5,
            linewidth=1.4,
            label=FOLLOWER_LABELS[follower] if with_label else None,
        )
        ax.fill_between(
            sub["bound"],
            sub["attacked_mse_mean"] - sub["attacked_mse_ci95_radius"],
            sub["attacked_mse_mean"] + sub["attacked_mse_ci95_radius"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )


def save_figures(
    results: pd.DataFrame, summary: pd.DataFrame, output_dir: Path
) -> None:
    objectives = [
        o
        for o in ("mse", "prediction", "signed_error")
        if o in set(summary["objective"])
    ]
    followers = [
        f for f in FOLLOWERS + OPTIONAL_FOLLOWERS if f in set(summary["follower"])
    ]

    # Figure 1: grid (rows = objectives, cols = predictors), all followers.
    fig, axes = plt.subplots(
        len(objectives), 3, figsize=(11, 3.2 * len(objectives)), squeeze=False
    )
    for i, objective in enumerate(objectives):
        stats = summary[summary["objective"] == objective]
        for j, method in enumerate(METHOD_ORDER):
            ax = axes[i, j]
            _plot_lines(ax, stats, method, followers, with_label=(i == 0 and j == 0))
            clean = stats[(stats["method"] == method) & (stats["follower"] == "none")]
            if not clean.empty:
                ax.axhline(
                    clean["attacked_mse_mean"].mean(),
                    color=METHOD_COLORS[method],
                    linestyle=":",
                    alpha=0.35,
                )
            ax.set_title(f"{METHOD_LABELS[method]}  |  {LABELS[objective]}", fontsize=9)
            ax.set_xlabel("intervention bound", fontsize=8)
            if j == 0:
                ax.set_ylabel("deployment MSE", fontsize=8)
            ax.tick_params(labelsize=7)
    handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=FOLLOWER_STYLES[f][0],
            marker=FOLLOWER_STYLES[f][1],
            fillstyle=FOLLOWER_STYLES[f][2],
            label=FOLLOWER_LABELS[f],
        )
        for f in followers
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output_dir / "followers_grid.png", dpi=180)
    fig.savefig(output_dir / "followers_grid.pdf")
    plt.close(fig)

    # Figure 2: paper layout (panel per objective), population nn_x1x3 vs ind_seq.
    pair = [
        f
        for f in (
            "nn_standard",
            "ind_pa4",
            "nn_x1x3",
            "ind_seq",
            "ind_simple",
            "nn_x1",
            "ind_nox3",
            "nn_x1_warm",
            "nn_x1_long",
            "nn_x1_tuned",
        )
        if f in followers
    ]
    fig, axes = plt.subplots(
        1, len(objectives), figsize=(4.2 * len(objectives), 3.3), squeeze=False
    )
    for i, objective in enumerate(objectives):
        ax = axes[0, i]
        stats = summary[summary["objective"] == objective]
        for method in METHOD_ORDER:
            _plot_lines(ax, stats, method, pair, with_label=False)
        ax.set_title(LABELS[objective], fontsize=9)
        ax.set_xlabel("intervention bound", fontsize=8)
        if i == 0:
            ax.set_ylabel("deployment MSE", fontsize=8)
        ax.tick_params(labelsize=7)
    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], linestyle="-", label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
    follower_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=FOLLOWER_STYLES[f][0],
            marker=FOLLOWER_STYLES[f][1],
            fillstyle=FOLLOWER_STYLES[f][2],
            label=FOLLOWER_LABELS[f],
        )
        for f in pair
    ]
    handles = method_handles + follower_handles
    ncol = 3 if len(handles) > 5 else len(handles)
    fig.legend(
        handles=handles, loc="lower center", ncol=ncol, fontsize=7, frameon=False
    )
    fig.tight_layout(rect=(0, 0.14 + 0.06 * ((len(handles) - 1) // ncol), 1, 1))
    fig.savefig(output_dir / "individual_vs_population.png", dpi=180)
    fig.savefig(output_dir / "individual_vs_population.pdf")
    plt.close(fig)


def main() -> None:
    cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if PLOT_ONLY:
        results = pd.read_csv(output_dir / "results_per_run.csv")
    else:
        (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        all_rows = Parallel(n_jobs=min(cfg.n_jobs, cfg.num_runs))(
            delayed(run_single)(cfg, run_id) for run_id in range(cfg.num_runs)
        )
        results = pd.DataFrame([row for rows in all_rows for row in rows])
        results.to_csv(output_dir / "results_per_run.csv", index=False)
    summary = summarize(results)
    summary.to_csv(output_dir / "results_summary.csv", index=False)
    diffs = paired_differences(results)
    diffs.to_csv(output_dir / "paired_differences.csv", index=False)
    save_figures(results, summary, output_dir)
    pd.set_option("display.width", 200)
    print("\n=== deployment MSE (mean over runs) ===")
    print(
        summary.pivot_table(
            index=["objective", "method", "bound"],
            columns="follower",
            values="attacked_mse_mean",
        )
        .round(4)
        .to_string()
    )
    print("\n=== paired differences (mean ± sem over runs) ===")
    print(diffs.round(4).to_string(index=False))
    print(f"\nTotal time {time.time() - t0:.0f}s; outputs in {output_dir}")


if __name__ == "__main__":
    main()
