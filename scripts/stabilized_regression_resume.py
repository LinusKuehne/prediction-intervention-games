"""Stabilized Regression -- resume the per-subset score sweep.

Continuation of ``stabilized_regression.py``: it scores subsets from a seeded
shuffle of all 8192 covariate subsets and writes them to SCORES_CSV; if it is
interrupted, only some of them are done.

This script reconstructs the *exact same* seeded subset order and the *exact
same* seeded training subsample, reads SCORES_CSV to see which subsets are
already there, and scores every subset still missing -- appending them to the
same CSV in the seeded shuffle order.  There is no subset-count limit: it runs
until all 8192 subsets are done or you stop it with Ctrl-C (run it again to
keep going).  Like the original it works batch-wise and is Ctrl-C-safe:
completed batches are flushed to disk, only the in-progress batch is lost.

Everything else (data handling, subsampling, scoring) is identical to
stabilized_regression.py, so a subset scored here gets exactly the score it
would have gotten in the original run.
"""

import contextlib
import csv
import io
import os
import warnings
from itertools import chain, combinations
from typing import cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from stab_reg_eval import skill_score

# ---------------------------------------------------------------------------
# configuration  (data / SEED / subsample / forest params match the original)
# ---------------------------------------------------------------------------

DATA_PATH = (
    "/Users/linuskuehne/git/prediction-intervention-games/scripts/sr_TA40_ET.csv"
)
SCORES_CSV = "results/stabilized_regression_scores.csv"

TARGET = "ET"
PFT_PREFIX = "PFT_"  # one-hot columns bundled into a single logical covariate

N_ESTIMATORS = 250
N_JOBS = 12  # subsets evaluated in parallel
SEED = 42
N_PER_CLUSTER = 15000  # training rows kept per cluster (seeded subsample)
CHUNK_SIZE = 4 * N_JOBS  # subsets per parallel batch; CSV is flushed after each

# ---------------------------------------------------------------------------
# load data and build the logical covariates ("feature groups")
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH, low_memory=False)

# 12 continuous covariates (each its own group) + the PFT_* one-hot bundle
pft_cols = [c for c in df.columns if c.startswith(PFT_PREFIX)]
continuous_cols = [
    c
    for c in df.columns
    if c not in (TARGET, "env", "time", "cluster", "split")
    and not c.startswith(PFT_PREFIX)
]
feature_cols = continuous_cols + pft_cols  # flat column order of the X matrices
feature_groups = [[c] for c in continuous_cols] + [pft_cols]  # 13 logical covariates
group_names = continuous_cols + ["PFT"]

# numpy column indices spanned by each logical covariate
group_col_idx = []
_offset = 0
for _grp in feature_groups:
    group_col_idx.append(list(range(_offset, _offset + len(_grp))))
    _offset += len(_grp)

# splits: training environment = cluster, val/test environment = env.
# the training set is subsampled to N_PER_CLUSTER rows per cluster (seeded).
df_train = (
    df[df["split"] == "train"]
    .groupby("cluster")
    .sample(n=N_PER_CLUSTER, random_state=SEED)
)
df_val = df[df["split"] == "val"]
df_test = df[df["split"] == "test"]

X_train = df_train[feature_cols].to_numpy(dtype=float)
y_train = df_train[TARGET].to_numpy(dtype=float)
E_train = df_train["cluster"].astype(int).to_numpy()
env_values = np.unique(E_train)

X_val = df_val[feature_cols].to_numpy(dtype=float)
y_val = df_val[TARGET].to_numpy(dtype=float)
E_val = df_val["env"].to_numpy()
env_values_val = np.unique(E_val)

X_test = df_test[feature_cols].to_numpy(dtype=float)
y_test = df_test[TARGET].to_numpy(dtype=float)
E_test = df_test["env"].to_numpy()
env_values_test = np.unique(E_test)
time_test = pd.to_datetime(df_test["time"]).to_numpy()  # datetime64; for skill_score

print(
    f"Loaded {len(df)} rows -> "
    f"train {len(X_train)} (subsampled, {N_PER_CLUSTER}/cluster) / "
    f"val {len(X_val)} / test {len(X_test)}\n"
    f"{len(group_names)} logical covariates "
    f"({len(continuous_cols)} continuous + 1 PFT bundle of {len(pft_cols)} columns)\n"
    f"{len(env_values)} training environments (clusters): {env_values.tolist()}; "
    f"{len(env_values_val)} val / {len(env_values_test)} test environments"
)

del df, df_train, df_val, df_test  # free the dataframe; numpy arrays are enough

# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------


def rmse(y_true, y_pred):
    """Root-mean-squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def median_over_envs(y_true, y_pred, environment, env_values_, score_fn):
    """Apply ``score_fn`` within each environment, return the median across envs."""
    per_env = [
        score_fn(y_true[environment == e], y_pred[environment == e])
        for e in env_values_
    ]
    return float(np.median(per_env))


class MeanRegressor:
    """Constant predictor (training mean of y) for the empty covariate subset."""

    def fit(self, X_sub, y_sub):
        self.mean_ = float(np.mean(y_sub))
        return self

    def predict(self, X_sub):
        return np.full(X_sub.shape[0], self.mean_)


def fit_forest(X_sub, y_sub, n_estimators, seed, rf_n_jobs=1):
    """Fit an out-of-bag random-forest regressor on X_sub.

    For the empty subset (0 columns) there is nothing to fit, so the
    constant-mean predictor is returned instead.  ``rf_n_jobs`` controls the
    forest's internal parallelism: 1 while subsets are swept in parallel.
    """
    if X_sub.shape[1] == 0:
        return MeanRegressor().fit(X_sub, y_sub)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        oob_score=True,
        random_state=seed,
        n_jobs=rf_n_jobs,
    )
    with warnings.catch_warnings():
        # small environments can leave a few samples without an OOB score
        warnings.simplefilter("ignore", UserWarning)
        model.fit(X_sub, y_sub)
    return model


def oob_predictions(model, X_sub):
    """Out-of-bag predictions of a fitted model (in-bag fallback for NaNs)."""
    if isinstance(model, MeanRegressor):
        return model.predict(X_sub)
    oob = np.asarray(model.oob_prediction_, dtype=float).ravel()
    if np.isnan(oob).any():
        fallback = model.predict(X_sub)
        oob = np.where(np.isnan(oob), fallback, oob)
    return oob


def subset_columns(subset):
    """Numpy column indices for a subset given as a tuple of group indices."""
    return [c for g in subset for c in group_col_idx[g]]


def subset_name(subset):
    """Human-readable name of a subset given as a tuple of group indices."""
    return "{" + ", ".join(group_names[g] for g in subset) + "}"


def evaluate_subset(subset, cols, train, val, test, n_estimators, seed):
    """Score one covariate subset.  Worker for the parallel sweep.

    ``cols`` are the numpy column indices of the subset; ``train`` / ``val``
    are tuples ``(X, y, environment, env_values)`` and ``test`` additionally
    carries ``time``.  Returns the invariance score, the per-environment
    regrets, the three predictiveness scores and the test-set skill score --
    but no fitted forests, so the sweep stays light.
    """
    X_train, y_train, E_train, envs_train = train
    X_val, y_val, E_val, envs_val = val
    X_test, y_test, E_test, envs_test, time_test = test

    X_tr = X_train[:, cols]

    # all-environment model: fitted on the pooled training set
    all_model = fit_forest(X_tr, y_train, n_estimators, seed)
    oob_all = oob_predictions(all_model, X_tr)

    # predictiveness on train: pooled out-of-bag RMSE
    predictiveness_train = rmse(y_train, oob_all)

    # predictiveness on val / test: RMSE scored within each environment, then
    # the median across environments
    val_pred = all_model.predict(X_val[:, cols])
    test_pred = all_model.predict(X_test[:, cols])
    predictiveness_val = median_over_envs(y_val, val_pred, E_val, envs_val, rmse)
    predictiveness_test = median_over_envs(y_test, test_pred, E_test, envs_test, rmse)

    # skill score of this subset's test-set predictions; skill_score's debug
    # prints are silenced -- many workers would otherwise flood stdout
    predictions_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": test_pred,
            "env": E_test,
            "site_id": E_test,
            "time": time_test,
        }
    )
    with contextlib.redirect_stdout(io.StringIO()):
        skill_test = skill_score(
            predictions_df, target="ET", setting="time-split", agg="median"
        )

    # regret of S in each training environment (cluster)
    regrets = {}
    for e in envs_train:
        mask = E_train == e

        # g_all(S, e): RMSE on env e of the all-environment model
        g_all_e = rmse(y_train[mask], oob_all[mask])

        # g_e(S, e): RMSE on env e of a model fitted on env e only
        env_model = fit_forest(X_tr[mask], y_train[mask], n_estimators, seed)
        oob_e = oob_predictions(env_model, X_tr[mask])
        g_e = rmse(y_train[mask], oob_e)

        regrets[e] = g_all_e - g_e

    # invariance score: variance of the regret across environments
    return {
        "subset": subset,
        "regrets": regrets,
        "invariance_score": float(np.var(list(regrets.values()))),
        "predictiveness_train": predictiveness_train,
        "predictiveness_val": predictiveness_val,
        "predictiveness_test": predictiveness_test,
        "skill_score_test": skill_test,
    }


# ---------------------------------------------------------------------------
# rebuild the seeded subset order -- all 8192 subsets, no count limit
# ---------------------------------------------------------------------------

all_subsets = list(
    chain.from_iterable(
        combinations(range(len(feature_groups)), r)
        for r in range(len(feature_groups) + 1)
    )
)
order = np.random.default_rng(SEED).permutation(len(all_subsets))
subsets = [all_subsets[i] for i in order]

# ---------------------------------------------------------------------------
# resume: keep only the target subsets not already present in the CSV
# ---------------------------------------------------------------------------

csv_exists = os.path.exists(SCORES_CSV)
if csv_exists:
    done_names = set(pd.read_csv(SCORES_CSV)["subset"].dropna().astype(str))
else:
    done_names = set()

remaining = [s for s in subsets if subset_name(s) not in done_names]
print(
    f"\n{len(done_names)} subsets already in {SCORES_CSV}; "
    f"{len(remaining)} of the {len(subsets)} target subsets still to do."
)

# ---------------------------------------------------------------------------
# score the remaining subsets, appending each finished batch to the CSV
# ---------------------------------------------------------------------------

fieldnames = [
    "subset",
    "invariance_score",
    "predictiveness_train",
    "predictiveness_val",
    "predictiveness_test",
    "skill_score_test",
] + [f"regret_cluster_{e}" for e in env_values]

train_data = (X_train, y_train, E_train, env_values)
val_data = (X_val, y_val, E_val, env_values_val)
test_data = (X_test, y_test, E_test, env_values_test, time_test)

os.makedirs(os.path.dirname(SCORES_CSV) or ".", exist_ok=True)
print(
    f"\nScoring {len(remaining)} subsets on {N_JOBS} workers, in batches of "
    f"{CHUNK_SIZE}; appending to {SCORES_CSV}\n"
    "Safe to interrupt with Ctrl-C -- completed batches are already on disk."
)

n_done = 0
# append to the existing CSV; create it with a header only if it is missing
with open(SCORES_CSV, "a" if csv_exists else "w", newline="") as f:
    writer = csv.writer(f)
    if not csv_exists:
        writer.writerow(fieldnames)
        f.flush()
    try:
        # subsets are scored one batch at a time with a plain (non-generator)
        # Parallel call so that Ctrl-C interrupts cleanly; after each batch its
        # rows are flushed.  Only the in-progress batch is lost on Ctrl-C.
        for start in range(0, len(remaining), CHUNK_SIZE):
            batch = remaining[start : start + CHUNK_SIZE]
            batch_results = cast(
                "list[dict]",
                Parallel(n_jobs=N_JOBS, verbose=10)(
                    delayed(evaluate_subset)(
                        subset,
                        subset_columns(subset),
                        train_data,
                        val_data,
                        test_data,
                        N_ESTIMATORS,
                        SEED,
                    )
                    for subset in batch
                ),
            )
            for res in batch_results:
                s = res["subset"]
                writer.writerow(
                    [
                        subset_name(s),
                        res["invariance_score"],
                        res["predictiveness_train"],
                        res["predictiveness_val"],
                        res["predictiveness_test"],
                        res["skill_score_test"],
                    ]
                    + [res["regrets"][e] for e in env_values]
                )
            f.flush()  # the whole finished batch is durable
            n_done += len(batch)
            print(f"  {n_done}/{len(remaining)} remaining subsets written")
    except KeyboardInterrupt:
        print(
            f"\nInterrupted -- {n_done} more subsets written; the in-progress "
            f"batch of up to {CHUNK_SIZE} was dropped."
        )
    else:
        print(f"\nDone -- all {n_done} remaining subsets written.")
