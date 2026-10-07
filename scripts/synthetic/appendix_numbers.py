"""Numbers quoted in the text of Appendix E.5 (individual vs population follower).

Reads the saved outputs of run_individual_follower.py; nothing is recomputed.

    python appendix_numbers.py outputs_appendix_anneal_lingauss outputs_appendix_anneal_standard

Only bounds up to --max-bound (default 1.0, the bounds shown in the appendix figure) are
used. Followers: nn_standard (population follower of Section 6.1), nn_x1 (its delta_4 also
depends on X1), ind_nox3 (individual followers).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEYS = ["method", "objective", "bound"]


def _mean_mse(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.pivot_table(
        index=KEYS, columns="follower", values="attacked_mse_mean"
    )


def x1_effect(summary: pd.DataFrame) -> None:
    """Effect of adding X1 to the inputs of delta_4, same training recipe."""
    mse = _mean_mse(summary)
    if not {"nn_x1", "nn_standard"} <= set(mse.columns):
        return
    diff = (mse["nn_x1"] - mse["nn_standard"]).abs()
    print("max |mean MSE(nn_x1) - mean MSE(nn_standard)| over bounds and objectives:")
    print(diff.groupby(level="method").max().round(4).to_string(), "\n")


def individuals_vs_population(summary: pd.DataFrame) -> None:
    """Relative difference (individuals - population) / population of the mean MSE, b > 0."""
    mse = _mean_mse(summary)
    if not {"ind_nox3", "nn_x1"} <= set(mse.columns):
        return
    rel = 100 * (mse["ind_nox3"] - mse["nn_x1"]) / mse["nn_x1"]
    rel = rel[rel.index.get_level_values("bound") > 0]
    table = rel.groupby(level=["method", "objective"]).agg(["min", "max"]).round(2)
    print("relative difference of mean MSE, individuals vs population (%), b > 0:")
    print(table.to_string(), "\n")


def shifts(per_run: pd.DataFrame) -> None:
    """Individuals' shifts of X4 and runs in which a network's shift of X4 has one sign."""
    rows = per_run[(per_run["method"] == "all_variables") & (per_run["bound"] > 0)]
    rows = rows.assign(
        rel_mean_delta_x4=rows["mean_delta_x4"] / rows["bound"],
        one_sign=(rows["mean_delta_x4"].abs() > 0.9 * rows["bound"]),
    )
    ind = rows[rows["follower"] == "ind_nox3"].groupby(["objective", "bound"])
    print("individuals, all-variable predictor: fraction with |delta_4| at the bound,")
    print("and mean delta_4 / b (-1: shift -b for every individual):")
    print(ind[["frac_bound_x4", "rel_mean_delta_x4"]].mean().round(4).to_string(), "\n")
    nn = rows[rows["follower"].str.startswith("nn") & (rows["objective"] == "mse")]
    one_sign = nn.groupby(["follower", "bound"])["one_sign"].sum().unstack()
    print("population follower, all-variable predictor, MSE objective: runs in which")
    print("the shift of X4 has one sign for the whole population (|mean| > 0.9 b):")
    print(one_sign.to_string(), "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("output_dirs", nargs="+", type=Path)
    p.add_argument("--max-bound", type=float, default=1.0)
    a = p.parse_args()
    for d in a.output_dirs:
        summary = pd.read_csv(d / "results_summary.csv")
        per_run = pd.read_csv(d / "results_per_run.csv")
        summary = summary[summary["bound"] <= a.max_bound]
        per_run = per_run[per_run["bound"] <= a.max_bound]
        print(f"=== {d} ===")
        x1_effect(summary)
        individuals_vs_population(summary)
        shifts(per_run)


if __name__ == "__main__":
    main()
