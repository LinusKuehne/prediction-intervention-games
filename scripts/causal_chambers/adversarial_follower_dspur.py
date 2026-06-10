"""Adversarial follower experiment for D-spur, over 10 repetitions.

Each repetition trains the predictors on its own disjoint, env-stratified fold of
the D-spur training data and evaluates them against an independently lab-collected
action cube (one per rep; see `generate_adversarial_cubes.py`). The budget curves
are then plotted with t-based 95% confidence bands across the 10 reps.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestClassifier

from adversarial_common import (
    ACTION_NAMES,
    ALL_FEATURES,
    BUDGETS,
    DATA_DIR,
    M,
    N_REPS,
    PRED_COLORS,
    PRED_LABELS,
    SB_B_FEATURES,
    SB_FEATURES,
    SB_IR2_FEATURES,
    SEED,
    THETA_REF,
    build_X,
    cube_path,
    eligible_by_budget,
    eval_metrics,
    restore_from_cube,
)
from stabilized_classification import StabilizedClassificationClassifier

DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_ENVS = [0, 1, 2]

# predictor name -> feature list (predictors themselves are fit per rep)
PRED_FEATURES = {
    "f_sb": SB_FEATURES,
    "f_sb_ir2": SB_IR2_FEATURES,
    "f_sb_b": SB_B_FEATURES,
    "f_all": ALL_FEATURES,
    "f_sc": ALL_FEATURES,
}


# ─── budget action menu (printed once) ────────────────────────────────────────

print(f"Action menu: {M} actions")
for budget in BUDGETS:
    eligible_idx = eligible_by_budget(budget)
    print(f"Budget {budget}: {len(eligible_idx)}/{M} actions eligible")


# ─── training-data folds (disjoint, env-stratified) ──────────────────────────


def make_train_folds(df_train, n_reps=N_REPS, seed=SEED):
    """Partition rows into n_reps disjoint folds, stratified by environment.

    For each env in TRAIN_ENVS the row indices are shuffled with a seeded RNG and
    split into n_reps near-equal chunks; fold r is the concatenation of chunk r
    across envs. Returns a list of integer-position index arrays into df_train.
    """
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_reps)]
    E = np.asarray(df_train["E"].values, dtype=int)
    for env in TRAIN_ENVS:
        pos = np.where(E == env)[0]
        rng.shuffle(pos)
        for r, chunk in enumerate(np.array_split(pos, n_reps)):
            folds[r].append(chunk)
    return [np.concatenate(parts) for parts in folds]


# ─── per-rep predictor training ──────────────────────────────────────────────


def train_predictors(df_fold, seed):
    """Fit the 5 predictors on one training fold; return {name: (model, features)}."""
    y = np.asarray(df_fold["Y"].values, dtype=int)
    E = np.asarray(df_fold["E"].values, dtype=int)

    predictors = {}
    for name in ["f_sb", "f_sb_ir2", "f_sb_b", "f_all"]:
        feats = PRED_FEATURES[name]
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        clf.fit(df_fold[feats].values, y)
        predictors[name] = (clf, feats)

    f_sc = StabilizedClassificationClassifier(
        invariance_test="tram_gcm",
        test_classifier_type="RF",
        pred_classifier_type="RF",
        random_state=seed,
        n_jobs=8,
    )
    f_sc.fit(df_fold[ALL_FEATURES].values, y, E)
    predictors["f_sc"] = (f_sc, ALL_FEATURES)
    return predictors


def inspect_sc(f_sc):
    """Print the SC ensemble composition (diagnostic, rep 0 only)."""
    print(
        f"\nf_sc ensemble: {f_sc.n_predictive_subsets_} active / "
        f"{f_sc.n_invariant_subsets_} invariant / {f_sc.n_subsets_total_} total subsets"
    )
    sb_b_idx_set = frozenset(ALL_FEATURES.index(f) for f in SB_B_FEATURES)
    for stat in sorted(f_sc.active_subsets_, key=lambda s: -s["score"]):
        names = [ALL_FEATURES[i] for i in stat["subset"]]
        marker = "  <-- == SB_B" if frozenset(stat["subset"]) == sb_b_idx_set else ""
        print(
            f"  w={stat['weight']:.3f}  score={stat['score']:+.4f}  "
            f"p={stat['p_value']:.3f}  {names}{marker}"
        )

    print("\nAll invariant subsets (sorted by score):")
    for stat in sorted(f_sc._all_invariant_fitted_, key=lambda s: -s["score"]):
        names = [ALL_FEATURES[i] for i in stat["subset"]]
        in_active = any(stat["subset"] == a["subset"] for a in f_sc.active_subsets_)
        marker = "  *active*" if in_active else ""
        marker += "  <-- == SB_B" if frozenset(stat["subset"]) == sb_b_idx_set else ""
        print(f"  score={stat['score']:+.4f}  p={stat['p_value']:.3f}  {names}{marker}")


# ─── per-rep evaluation ──────────────────────────────────────────────────────


def evaluate_rep(predictors, cube_df, seed, fixed_action_csv=None):
    """Run probe/eval split + best-response by budget for one rep.

    Returns a list of per-(predictor, budget) result dicts.
    """
    R_all, Y_all, B_by_action, Z_by_action = restore_from_cube(cube_df)

    rng = np.random.default_rng(seed)
    N = len(Y_all)
    perm = rng.permutation(N)
    n_probe = N // 2
    probe_idx = perm[:n_probe]
    eval_idx = perm[n_probe:]

    R_probe = R_all[probe_idx]
    R_eval = R_all[eval_idx]
    Y_eval = Y_all[eval_idx]

    # probe mean scores: probe_mean_score[pred][m] = mean E[f(X^m)] on probe split
    probe_mean_score = {}
    for pred_name, (predictor, features) in predictors.items():
        ms = np.full(M, np.nan)
        for m, action_name in enumerate(ACTION_NAMES):
            X_m = build_X(
                R_probe,
                B_by_action[action_name][probe_idx],
                Z_by_action[action_name][probe_idx],
                features,
            )
            ms[m] = predictor.predict_proba(X_m)[:, 1].mean()
        probe_mean_score[pred_name] = ms

    # optional fixed-action diagnostics (eval split) -> CSV
    if fixed_action_csv is not None:
        fixed_rows = []
        for pred_name, (predictor, features) in predictors.items():
            for action_name in ACTION_NAMES:
                X_m = build_X(
                    R_eval,
                    B_by_action[action_name][eval_idx],
                    Z_by_action[action_name][eval_idx],
                    features,
                )
                p_m = predictor.predict_proba(X_m)[:, 1]
                met = eval_metrics(Y_eval, p_m)
                fixed_rows.append(
                    {"predictor": pred_name, "action": action_name, **met}
                )
        pd.DataFrame(fixed_rows).to_csv(fixed_action_csv, index=False)
        print(f"Saved {os.path.basename(fixed_action_csv)}")

    # best-response by budget
    rows = []
    for budget in BUDGETS:
        eligible_idx = eligible_by_budget(budget)
        for pred_name, (predictor, features) in predictors.items():
            ms_eligible = probe_mean_score[pred_name][eligible_idx]
            best_m = eligible_idx[int(np.argmin(ms_eligible))]
            best_action = ACTION_NAMES[best_m]

            # clean: always under theta_ref reference
            X_clean = build_X(
                R_eval,
                B_by_action[THETA_REF][eval_idx],
                Z_by_action[THETA_REF][eval_idx],
                features,
            )
            p_clean = predictor.predict_proba(X_clean)[:, 1]
            met_clean = eval_metrics(Y_eval, p_clean)

            # adversarial: single best-response action applied to all eval samples
            X_adv = build_X(
                R_eval,
                B_by_action[best_action][eval_idx],
                Z_by_action[best_action][eval_idx],
                features,
            )
            p_adv = predictor.predict_proba(X_adv)[:, 1]
            met_adv = eval_metrics(Y_eval, p_adv)

            rows.append(
                {
                    "predictor": pred_name,
                    "budget": budget,
                    **{f"clean_{k}": v for k, v in met_clean.items()},
                    **{f"adv_{k}": v for k, v in met_adv.items()},
                    "best_action": best_action,
                }
            )
    return rows


# ─── aggregation (t-based 95% CI across reps) ────────────────────────────────


def summarize(df, quantities):
    """Per (predictor, budget) mean + t-based 95% CI for each named quantity."""
    out = []
    for (pred_name, budget), g in df.groupby(["predictor", "budget"]):
        row = {"predictor": pred_name, "budget": budget, "n_reps": len(g)}
        n = len(g)
        tcrit = student_t.ppf(0.975, n - 1) if n > 1 else float("nan")
        for q in quantities:
            vals = g[q].to_numpy()
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            sem = std / np.sqrt(n) if n > 1 else 0.0
            ci95 = tcrit * sem if n > 1 else 0.0
            row[f"{q}_mean"] = mean
            row[f"{q}_std"] = std
            row[f"{q}_sem"] = sem
            row[f"{q}_ci95"] = ci95
            row[f"{q}_ci_low"] = mean - ci95
            row[f"{q}_ci_high"] = mean + ci95
        out.append(row)
    return pd.DataFrame(out)


# ─── main ─────────────────────────────────────────────────────────────────────


def main():
    df_train_full = pd.read_csv(os.path.join(DATA_DIR, "d_spur_train.csv"))
    df_train_full = df_train_full[df_train_full["E"].isin(TRAIN_ENVS)].reset_index(
        drop=True
    )
    print(f"Training pool (envs {TRAIN_ENVS}): {len(df_train_full)} obs")

    folds = make_train_folds(df_train_full)
    for r, idx in enumerate(folds):
        env_counts = df_train_full.iloc[idx]["E"].value_counts().sort_index().to_dict()
        print(f"  fold {r}: n={len(idx)}  per-env {env_counts}")

    all_rows = []
    for rep in range(N_REPS):
        cube_file = os.path.join(DATA_DIR, cube_path(rep))
        if not os.path.exists(cube_file):
            raise FileNotFoundError(
                f"Missing action cube for rep {rep}: {cube_file}\n"
                f"Run generate_adversarial_cubes.py first to collect cubes 0..{N_REPS - 1}."
            )
        print(f"\n═══ rep {rep} ═══  (cube: {os.path.basename(cube_file)})")
        cube_df = pd.read_csv(cube_file)

        predictors = train_predictors(df_train_full.iloc[folds[rep]], seed=SEED + rep)
        if rep == 0:
            inspect_sc(predictors["f_sc"][0])

        fixed_csv = (
            os.path.join(DATA_DIR, "fixed_action_results.csv") if rep == 0 else None
        )
        rows = evaluate_rep(
            predictors, cube_df, seed=SEED + rep, fixed_action_csv=fixed_csv
        )
        for row in rows:
            row["rep"] = rep
        all_rows.extend(rows)

    df_reps = pd.DataFrame(all_rows)
    df_reps["delta_ef"] = df_reps["adv_ef"] - df_reps["clean_ef"]
    reps_csv = os.path.join(DATA_DIR, "adversarial_results_by_budget_reps.csv")
    df_reps.to_csv(reps_csv, index=False)
    print(f"\nSaved {os.path.basename(reps_csv)}  ({len(df_reps)} rows)")

    quantities = ["delta_ef", "adv_brier", "clean_brier"]
    df_summary = summarize(df_reps, quantities)
    summary_csv = os.path.join(DATA_DIR, "adversarial_results_by_budget_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved {os.path.basename(summary_csv)}")

    plot_budget_curves(df_summary)


# ─── plot ─────────────────────────────────────────────────────────────────────


def plot_budget_curves(df_summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), sharey=False)

    df_plot = df_summary[df_summary["budget"] <= 0.5]

    for pred_name in PRED_FEATURES:
        sub = df_plot[df_plot["predictor"] == pred_name].sort_values("budget")
        color = PRED_COLORS[pred_name]
        label = PRED_LABELS[pred_name]
        b = sub["budget"].to_numpy()

        # left panel: delta E[f] with band
        axes[0].plot(b, sub["delta_ef_mean"], marker="o", color=color, label=label)
        axes[0].fill_between(
            b,
            sub["delta_ef_ci_low"],
            sub["delta_ef_ci_high"],
            color=color,
            alpha=0.2,
            linewidth=0,
        )

        # right panel: adversarial Brier with band + clean baseline (mean)
        axes[1].plot(b, sub["adv_brier_mean"], marker="o", color=color, label=label)
        axes[1].fill_between(
            b,
            sub["adv_brier_ci_low"],
            sub["adv_brier_ci_high"],
            color=color,
            alpha=0.2,
            linewidth=0,
        )
        axes[1].axhline(
            sub["clean_brier_mean"].iloc[0], color=color, linestyle=":", alpha=0.35
        )

    axes[0].axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    axes[0].set_xlabel("intervention bound", fontsize=12)
    axes[0].set_ylabel(
        r"$\mathbb{E}_{e^*(\hat{f})}[\hat{f}(X)] - \mathbb{E}_{e_{\mathrm{ref}}}[\hat{f}(X)]$",
        fontsize=12,
    )
    axes[1].set_xlabel("intervention bound", fontsize=12)
    axes[1].set_ylabel("deployment MSE\n" + r"under $e^*(\hat{f})$", fontsize=12)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=14)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=len(PRED_FEATURES),
        fontsize=14,
        frameon=False,
    )
    plt.tight_layout(rect=(0, 0.17, 1, 1))

    plot_base = os.path.join(DATA_DIR, "adversarial_budget_curves")
    plt.savefig(plot_base + ".png", dpi=300)
    plt.savefig(plot_base + ".pdf")
    plt.close()
    print(f"Saved plot to {plot_base}.{{png,pdf}}")


if __name__ == "__main__":
    main()
