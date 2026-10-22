"""Decompose the deployment MSE of PA(Y) and SB(Y) into population risk and regression error.

Nonlinear SCM of Appendix F.1 with the predictors and the neural follower of Fig. 2
(same seeds as ``run_experiments.py``).  On the follower's test sample, the deployment
MSE of the fitted predictor ``fhat_S`` is split as (Appendix F.3)

    E_e[(Y - fhat_S)^2] = E_e[Var(Y | X_S)] + E_e[(f_S - fhat_S)^2] + cross term,

where ``f_S = E[Y | X_S]`` is computed exactly: in closed form for PA(Y) and by
numerical integration over ``eps_Y`` for SB(Y).  The clean distribution and constant
shifts of ``X1`` are evaluated as well.  ``--summarize-only`` recomputes the summary
from a saved ``results_per_run.csv``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from joblib import Parallel, delayed
from synthetic_experiments.adversary import optimize_attack
from synthetic_experiments.experiment import (
    ExperimentConfig,
    _sample_datasets,
    configure_torch,
)
from synthetic_experiments.models import PredictorBundle, train_predictor
from synthetic_experiments.scm import NonlinearStableBlanketSCM

METHODS = ("parents", "stable_blanket")

# Quadrature grid for eps_Y ~ N(0, 1).
_EPS = torch.linspace(-6.0, 6.0, 241, dtype=torch.float64)
_W = torch.exp(-_EPS.square() / 2)
_W = _W / _W.sum()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--output-dir", type=str, default="outputs_risk_decomposition")
    p.add_argument("--num-runs", type=int, default=10)
    p.add_argument("--n-jobs", type=int, default=5, help="runs in parallel (joblib)")
    p.add_argument("--torch-num-threads", type=int, default=2, help="threads per run")
    p.add_argument("--bounds", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5])
    p.add_argument(
        "--objectives",
        type=str,
        nargs="+",
        choices=["mse", "prediction", "signed_error"],
        default=["mse", "prediction"],
    )
    p.add_argument(
        "--attack-bound-anneal-steps",
        type=int,
        default=0,
        help="0 reproduces the follower of Fig. 2; 1000 the one of Appendix F.5",
    )
    p.add_argument(
        "--const-shifts",
        type=float,
        nargs="*",
        default=[-1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5],
        help="constant shifts delta_1 of X1 (delta_4 = 0) evaluated for reference",
    )
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="only recompute summary.txt from <output-dir>/results_per_run.csv",
    )
    return p.parse_args()


def paper_config(torch_num_threads: int) -> ExperimentConfig:
    """Predictor and follower hyperparameters of Appendix F.4 (run_experiments.py defaults)."""
    return ExperimentConfig(
        torch_num_threads=torch_num_threads,
        n_train=50000,
        n_val=3000,
        n_test=4000,
        predictor_hidden_dim=256,
        predictor_depth=5,
        predictor_lr=1e-3,
        predictor_batch_size=512,
        predictor_max_epochs=500,
        predictor_patience=20,
        predictor_weight_decay=1e-5,
        attack_hidden_dim=64,
        attack_lr=1e-3,
        attack_steps=2000,
        attack_batch_size=512,
        attack_restarts=5,
        attack_eval_size=10000,
    )


def true_conditional(x: torch.Tensor, method: str) -> tuple[torch.Tensor, torch.Tensor]:
    """E[Y | X_S] and Var(Y | X_S) in the nonlinear SCM for S = PA(Y) or SB(Y).

    Y = m(X1, X2) + 0.4 eps_Y with eps_Y ~ N(0, 1), so for PA(Y) the conditional mean is
    m and the conditional variance is 0.16.  For SB(Y), X3 | Y ~ N(g(Y), 0.25) with
    g(y) = tanh(1.2 y) + 0.45 y^2, and the posterior of eps_Y given (X1, X2, X3) is
    integrated numerically on a grid.
    """
    x1, x2, x3 = (x[:, i].double() for i in range(3))
    mean = torch.sin(x1) + 0.8 * x2 + 0.5 * x1 * x2
    if method == "parents":
        return mean, torch.full_like(mean, 0.16)
    if method != "stable_blanket":
        raise ValueError(method)
    y = mean[:, None] + 0.4 * _EPS[None, :]
    g = torch.tanh(1.2 * y) + 0.45 * y.square()
    log_lik = -(x3[:, None] - g).square() / (2 * 0.25)
    post = torch.exp(log_lik - log_lik.max(dim=1, keepdim=True).values) * _W[None, :]
    post = post / post.sum(dim=1, keepdim=True)
    ey = (post * y).sum(dim=1)
    ey2 = (post * y.square()).sum(dim=1)
    return ey, ey2 - ey.square()


class ConstantShift:
    """Follower with delta_1 = constant and delta_4 = 0 (same interface as AttackMechanisms)."""

    def __init__(self, delta: float) -> None:
        self.delta = float(delta)

    def perturb_x1(self, x2: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x2, self.delta)

    def perturb_x4(self, y: torch.Tensor, x2, noise, x1=None, x3=None) -> torch.Tensor:
        return torch.zeros_like(y)


def evaluate(
    scm: NonlinearStableBlanketSCM,
    predictor: PredictorBundle,
    attack,
    *,
    run_id: int,
    objective: str,
    bound: float,
    eval_size: int,
) -> dict:
    """Decompose the deployment MSE on the follower's test sample (seed as in optimize_attack)."""
    generator = torch.Generator(device=scm.device)
    generator.manual_seed(run_id + 99999)
    with torch.no_grad():
        x, y, delta_x1, delta_x4 = scm.sample_with_intervention_info(
            eval_size, attack=attack, generator=generator
        )
        fhat = predictor.predict(x).squeeze(1).double()
    f, cond_var = true_conditional(x, predictor.name)
    y = y.squeeze(1).double()
    return {
        "run_id": run_id,
        "method": predictor.name,
        "objective": objective,
        "bound": float(bound),
        "clean_test_mse": float(predictor.clean_test_mse),
        "attacked_mse": (y - fhat).square().mean().item(),
        "pop_risk": (y - f).square().mean().item(),
        "pop_risk_condvar": cond_var.mean().item(),
        "reg_error": (f - fhat).square().mean().item(),
        "cross": (2 * (y - f) * (f - fhat)).mean().item(),
        "mean_delta_x1": delta_x1.mean().item(),
        "mean_abs_delta_x1": delta_x1.abs().mean().item(),
        "frac_bound_x1": (
            (delta_x1.abs() > 0.95 * bound).float().mean().item() if bound > 0 else 0.0
        ),
        "frac_neg_x1": (delta_x1 < 0).float().mean().item(),
        "mean_delta_x4": delta_x4.mean().item(),
        "mean_abs_delta_x4": delta_x4.abs().mean().item(),
        "mean_x1": x[:, 0].mean().item(),
        "mean_y": y.mean().item(),
    }


def run_single(args: argparse.Namespace, run_id: int) -> list[dict]:
    config = paper_config(args.torch_num_threads)
    configure_torch(config.torch_num_threads)
    scm = NonlinearStableBlanketSCM()
    # Same seeding and training order as _train_all_predictors in experiment.py.
    torch.manual_seed(run_id)
    x_tr, y_tr, x_va, y_va, x_te, y_te = _sample_datasets(
        config, scm, run_id, config.n_train
    )
    subsets = {
        "parents": scm.graph_sets.parents,
        "stable_blanket": scm.graph_sets.stable_blanket,
    }
    predictors: list[PredictorBundle] = []
    for name in METHODS:
        t0 = time.time()
        predictors.append(
            train_predictor(
                name=name,
                subset_indices=subsets[name],
                x_train_full=x_tr,
                y_train=y_tr,
                x_val_full=x_va,
                y_val=y_va,
                x_test_full=x_te,
                y_test=y_te,
                hidden_dim=config.predictor_hidden_dim,
                depth=config.predictor_depth,
                lr=config.predictor_lr,
                batch_size=config.predictor_batch_size,
                max_epochs=config.predictor_max_epochs,
                patience=config.predictor_patience,
                weight_decay=config.predictor_weight_decay,
            )
        )
        print(f"[run {run_id}] {name} trained in {time.time() - t0:.0f}s", flush=True)

    rows: list[dict] = []
    for predictor in predictors:
        common = dict(run_id=run_id, eval_size=config.attack_eval_size)
        rows.append(
            evaluate(scm, predictor, None, objective="none", bound=0.0, **common)
        )
        for delta in args.const_shifts:
            row = evaluate(
                scm,
                predictor,
                ConstantShift(delta),
                objective="const_shift",
                bound=abs(delta),
                **common,
            )
            row["const_shift"] = float(delta)
            rows.append(row)
        for objective in args.objectives:
            for bound in args.bounds:
                t0 = time.time()
                result = optimize_attack(
                    scm=scm,
                    predictor=predictor,
                    bound=float(bound),
                    cost=None,
                    intervene_on_x1=True,
                    simple=False,
                    x4_uses_x1_x3=False,
                    objective=objective,  # type: ignore[arg-type]
                    restarts=config.attack_restarts,
                    steps=config.attack_steps,
                    batch_size=config.attack_batch_size,
                    hidden_dim=config.attack_hidden_dim,
                    lr=config.attack_lr,
                    eval_size=config.attack_eval_size,
                    seed=run_id,
                    bound_anneal_steps=args.attack_bound_anneal_steps,
                )
                row = evaluate(
                    scm,
                    predictor,
                    result.attack,
                    objective=objective,
                    bound=bound,
                    **common,
                )
                row["nn_attacked_test_mse"] = float(result.attacked_test_mse)
                row["attack_objective_value"] = float(result.best_objective_value)
                row["seconds"] = time.time() - t0
                rows.append(row)
                print(
                    f"[run {run_id}] {predictor.name:14s} {objective:10s} b={bound:<4} "
                    f"mse={row['attacked_mse']:.4f} pop={row['pop_risk_condvar']:.4f} "
                    f"reg={row['reg_error']:.4f} mean_delta_x1={row['mean_delta_x1']:+.3f}",
                    flush=True,
                )
    return rows


def _sem(s: pd.Series) -> float:
    return float(s.std() / len(s) ** 0.5)


def summarize(df: pd.DataFrame) -> str:
    """Tables behind the numbers of Appendix F.3."""
    pd.set_option("display.width", 250)
    out: list[str] = [f"runs: {df.run_id.nunique()}"]
    nn = df.dropna(subset=["nn_attacked_test_mse"])
    out.append(
        "max |attacked_mse - nn_attacked_test_mse| = "
        f"{(nn.attacked_mse - nn.nn_attacked_test_mse).abs().max():.2e}"
    )
    out.append(
        "max |attacked_mse - pop_risk - reg_error - cross| = "
        f"{(df.attacked_mse - df.pop_risk - df.reg_error - df.cross).abs().max():.2e}"
    )
    out.append(f"max |cross| = {df.cross.abs().max():.4f}")

    follower = df[~df.objective.isin(["none", "const_shift"])]
    cols = [
        "attacked_mse",
        "pop_risk_condvar",
        "reg_error",
        "mean_delta_x1",
        "mean_abs_delta_x1",
        "frac_bound_x1",
        "frac_neg_x1",
        "mean_y",
    ]
    grouped = follower.groupby(["objective", "method", "bound"])
    table = grouped[cols].mean()
    table.insert(1, "attacked_mse_sem", grouped["attacked_mse"].apply(_sem))
    out += ["", "=== neural follower: means over runs ===", table.round(4).to_string()]

    clean = df[df.objective == "none"].set_index(["run_id", "method"])
    out += [
        "",
        "=== no follower ===",
        clean.groupby("method")[["attacked_mse", "pop_risk_condvar", "reg_error"]]
        .mean()
        .round(4)
        .to_string(),
    ]

    # Increase of each term relative to a reference, paired over runs.  The first
    # reference (the smallest bound, i.e. the first plotted one) is the one quoted in
    # Appendix F.3; ``pop_risk_share`` is the share of the increase due to the change
    # in population risk.
    keys = ["attacked_mse", "pop_risk_condvar", "reg_error"]
    objectives = sorted(follower.objective.unique())
    b0 = float(follower.bound.min())
    clean_by_method = {
        method: g.set_index("run_id")[keys]
        for method, g in df[df.objective == "none"].groupby("method")
    }
    references = [
        (
            f"b={b0} (first plotted bound, as in Appendix F.3)",
            {
                key: g.set_index("run_id")[keys]
                for key, g in follower[follower.bound == b0].groupby(
                    ["objective", "method"]
                )
            },
            b0,
        ),
        (
            "no intervention",
            {
                (objective, method): base
                for objective in objectives
                for method, base in clean_by_method.items()
            },
            0.0,
        ),
    ]
    for label, bases, min_bound in references:
        rows = []
        for (objective, method, bound), sub in follower.groupby(
            ["objective", "method", "bound"]
        ):
            if bound <= min_bound:
                continue
            sub = sub.set_index("run_id")
            d = sub[keys] - bases[(objective, method)].loc[sub.index]
            rows.append(
                {
                    "objective": objective,
                    "method": method,
                    "bound": bound,
                    "d_mse": d.attacked_mse.mean(),
                    "d_mse_sem": _sem(d.attacked_mse),
                    "d_pop_risk": d.pop_risk_condvar.mean(),
                    "d_pop_risk_sem": _sem(d.pop_risk_condvar),
                    "d_reg_error": d.reg_error.mean(),
                    "d_reg_error_sem": _sem(d.reg_error),
                    "pop_risk_share": d.pop_risk_condvar.mean() / d.attacked_mse.mean(),
                }
            )
        out += [
            "",
            f"=== increase relative to {label}, split into population risk and "
            "regression error (paired over runs) ===",
            pd.DataFrame(rows).round(4).to_string(index=False),
        ]

    pivot = follower.pivot_table(
        index=["objective", "run_id"],
        columns=["method", "bound"],
        values=["attacked_mse", "pop_risk_condvar", "reg_error"],
    )
    bounds = sorted(follower.bound.unique())
    out += [
        "",
        "=== increase from the smallest bound, PA(Y) vs SB(Y) (paired over runs) ===",
    ]
    for objective in follower.objective.unique():
        p = pivot.loc[objective]
        for bound in bounds[1:]:
            for key in ["attacked_mse", "pop_risk_condvar", "reg_error"]:
                d_pa = p[key][("parents", bound)] - p[key][("parents", bounds[0])]
                d_sb = (
                    p[key][("stable_blanket", bound)]
                    - p[key][("stable_blanket", bounds[0])]
                )
                diff = d_sb - d_pa
                out.append(
                    f"{objective:10s} {key:16s} {bounds[0]}->{bound}: PA {d_pa.mean():+.4f}  "
                    f"SB {d_sb.mean():+.4f}  SB-PA {diff.mean():+.4f} (sem {_sem(diff):.4f})"
                )

    const = df[df.objective == "const_shift"]
    if not const.empty:
        out += [
            "",
            "=== constant shift of X1 (delta_4 = 0): means over runs ===",
            const.groupby(["method", "const_shift"])[
                ["attacked_mse", "pop_risk_condvar", "reg_error"]
            ]
            .mean()
            .round(4)
            .to_string(),
        ]
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(run_single)(args, run_id) for run_id in range(args.num_runs)
        )
        df = pd.DataFrame([row for rows in results for row in rows])
        df.to_csv(output_dir / "results_per_run.csv", index=False)
        config = {
            **vars(args),
            "experiment_config": asdict(paper_config(args.torch_num_threads)),
        }
        (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    df = pd.read_csv(output_dir / "results_per_run.csv")
    summary = summarize(df)
    (output_dir / "summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
