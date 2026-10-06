"""Appendix figure in the style of Fig. 2: population follower (neural networks, delta_4 also
sees X1) against individual followers (grid search), one row of four panels.

    python plot_appendix_individual_followers.py outputs_appendix_final_lingauss \
        outputs_appendix_final_standard --out ../../prediction_intervention_games/images/appendix_individual_followers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from synthetic_experiments.plotting import METHOD_COLORS, METHOD_LABELS, METHOD_ORDER

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

try:
    from synthetic_experiments.plotting import METHOD_MARKERS
except ImportError:  # pragma: no cover
    from plot_paper_figures import METHOD_MARKERS

FOLLOWERS = {  # follower -> (label, linestyle, fillstyle)
    "nn_x1": ("population follower", "-", "full"),
    "ind_nox3": ("individual followers", "--", "none"),
}
TITLES = {"mse": "maximize MSE", "prediction": "minimize prediction"}


def _panel(ax, summary: pd.DataFrame, objective: str, followers: list[str]) -> None:
    stats = summary[summary["objective"] == objective]
    for method in METHOD_ORDER:
        for follower in followers:
            _, ls, fill = FOLLOWERS[follower]
            sub = stats[(stats["method"] == method) & (stats["follower"] == follower)]
            sub = sub.sort_values("bound")
            ax.plot(
                sub["bound"],
                sub["attacked_mse_mean"],
                color=METHOD_COLORS[method],
                linestyle=ls,
                marker=METHOD_MARKERS[method],
                fillstyle=fill,
                markersize=4,
                linewidth=1.4,
            )
            ax.fill_between(
                sub["bound"],
                sub["attacked_mse_mean"] - sub["attacked_mse_ci95_radius"],
                sub["attacked_mse_mean"] + sub["attacked_mse_ci95_radius"],
                color=METHOD_COLORS[method],
                alpha=0.14,
                linewidth=0,
            )
    ax.set_title(TITLES[objective], fontsize=11)
    ax.set_xlabel("intervention bound", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("lingauss_dir")
    p.add_argument("nonlinear_dir")
    p.add_argument("--followers", nargs="+", default=["nn_x1", "ind_nox3"])
    p.add_argument("--out", default="appendix_individual_followers")
    p.add_argument(
        "--max-bound", type=float, default=None, help="drop bounds above this"
    )
    a = p.parse_args()
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.45), sharey=False)
    for k, d in enumerate((a.lingauss_dir, a.nonlinear_dir)):
        summary = pd.read_csv(Path(d) / "results_summary.csv")
        if a.max_bound is not None:
            summary = summary[summary["bound"] <= a.max_bound]
        for m, objective in enumerate(("mse", "prediction")):
            _panel(axes[2 * k + m], summary, objective, a.followers)
    axes[0].set_ylabel("deployment MSE", fontsize=10)

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[m],
            marker=METHOD_MARKERS[m],
            linestyle="-",
            markersize=5,
            linewidth=1.6,
            label=METHOD_LABELS[m],
        )
        for m in METHOD_ORDER
    ] + [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=FOLLOWERS[f][1],
            marker="o",
            fillstyle=FOLLOWERS[f][2],
            markersize=5,
            linewidth=1.6,
            label=FOLLOWERS[f][0],
        )
        for f in a.followers
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))

    fig.canvas.draw()
    inv = fig.transFigure.inverted()

    def _pair_center(ax_a, ax_b) -> float:
        boxes = [
            b for b in (ax_a.get_tightbbox(), ax_b.get_tightbbox()) if b is not None
        ]
        union = Bbox.union(boxes).transformed(inv)
        return (union.x0 + union.x1) / 2

    for label, x in [
        ("Linear-Gaussian SCM", _pair_center(axes[0], axes[1])),
        ("Nonlinear SCM", _pair_center(axes[2], axes[3])),
    ]:
        fig.text(x, 0.965, label, ha="center", va="top", fontsize=11, fontweight="bold")
    out = Path(a.out)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
