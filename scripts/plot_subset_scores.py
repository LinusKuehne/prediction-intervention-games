"""Scatter plots of invariance score vs each predictiveness score.

Loads the frozen per-subset results (one row per scored covariate subset) and,
for every predictiveness-type score column, makes a scatter plot with one dot
per subset:

    x = invariance score        (variance of regrets; lower = more invariant)
    y = the predictiveness score

Dots are coloured by the number of covariates in the subset.  One PNG is
written per score, plus a combined grid, into results/plots/.
"""

import math
import os

import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("Agg")  # headless: only write PNG files, never open a window

SCORES_CSV = "results/stabilized_regression_scores_freeze.csv"
OUT_DIR = "results/plots"
INVARIANCE_COL = "invariance_score"

# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

df = pd.read_csv(SCORES_CSV)
df = df.dropna(subset=[INVARIANCE_COL])
print(f"Loaded {len(df)} subsets from {SCORES_CSV}")

# every score column except the invariance score, the subset name and the
# per-cluster regrets -> these are the y-axes, one plot each
score_cols = [
    c
    for c in df.columns
    if c not in ("subset", INVARIANCE_COL) and not c.startswith("regret_cluster_")
]
print(f"Predictiveness score columns ({len(score_cols)}): {score_cols}")


def n_covariates(name):
    """Number of logical covariates in a subset name like '{TA, VPD, PFT}'."""
    inner = str(name).strip().strip("{}").strip()
    return 0 if not inner else len([p for p in inner.split(",") if p.strip()])


df["n_covariates"] = df["subset"].map(n_covariates)


def direction(col):
    """Whether higher or lower is better for a score column (for the label)."""
    return "higher = better" if "skill" in col else "lower = better"


x = df[INVARIANCE_COL].to_numpy()
use_logx = bool((x > 0).all())  # invariance spans orders of magnitude
xlabel = "invariance score  (variance of regrets; lower = more invariant)"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# one scatter plot per predictiveness score
# ---------------------------------------------------------------------------

for col in score_cols:
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    sc = ax.scatter(
        x,
        df[col],
        c=df["n_covariates"],
        cmap="viridis",
        s=14,
        alpha=0.6,
        edgecolors="none",
    )
    if use_logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{col}  ({direction(col)})")
    ax.set_title(f"{col}  vs  invariance score   --   {len(df)} subsets")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="number of covariates in subset")
    out = os.path.join(OUT_DIR, f"invariance_vs_{col}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")

# ---------------------------------------------------------------------------
# combined grid (quick overview)
# ---------------------------------------------------------------------------

ncols = 2
nrows = math.ceil(len(score_cols) / ncols)
fig, axes = plt.subplots(
    nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False, layout="constrained"
)
flat = axes.ravel()
sc = None
for ax, col in zip(flat, score_cols, strict=False):
    sc = ax.scatter(
        x,
        df[col],
        c=df["n_covariates"],
        cmap="viridis",
        s=10,
        alpha=0.6,
        edgecolors="none",
    )
    if use_logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{col}  ({direction(col)})")
    ax.set_title(col)
    ax.grid(True, alpha=0.3)
for ax in flat[len(score_cols) :]:  # hide unused cells
    ax.set_visible(False)
if sc is not None:
    fig.colorbar(sc, ax=flat.tolist(), label="number of covariates in subset")
fig.suptitle(f"Invariance vs predictiveness -- {len(df)} subsets", fontsize=14)
combined = os.path.join(OUT_DIR, "invariance_vs_all.png")
fig.savefig(combined, dpi=150)
plt.close(fig)
print(f"  saved {combined}")
print("Done.")
