"""SC vs. IMP as a function of training sample size (per environment).

Second causal-chambers experiment (no data generation). SC (the stabilized-classification
*ensemble* over invariant + predictive subsets) and IMP (the single invariant subset with
the highest predictiveness score) coincide when data is abundant, but can differ when subset
selection / scoring is noisy. Here we vary the number of training observations PER
ENVIRONMENT and compare both under the STRONGEST intervention bound (budget = 1.0: the
follower may pick any of the 49 actions).

For each training size and each random repetition we:
  1. subsample ``n`` observations per environment from ``d_spur_train.csv``,
  2. fit ONE StabilizedClassification model,
  3. read off SC (``method="ensemble"``) and IMP (``method="best"``) from that same fit,
  4. let the follower pick the worst action over all 49 (probe split) and score the
     predictor on the eval split (deployment MSE / Brier),
and average over repetitions with 95% CIs.

Uses only the cached action cube + the training CSV — no chamber API, no torch. The action
menu / cube helpers are kept in sync with ``adversarial_follower_dspur.py``.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stabilized_classification import StabilizedClassificationClassifier

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "data")

ALL_FEATURES = ["red", "green", "blue", "ir_2", "vis_2", "ir_3", "vis_3"]
TRAIN_ENVS = [0, 1, 2]

SIZES = [50, 100, 200, 400, 800, 1600]  # training observations PER ENVIRONMENT
N_REPS = 5  # random repetitions per size (for 95% CIs)
SPLIT_SEED = 123  # cube probe/eval split (fixed across everything)
SUBSAMPLE_SEED = 1000  # base seed for per-env subsampling
SC_SEED = 2000  # base seed for the SC fit


# ─── action menu + cube helpers (in sync with adversarial_follower_dspur.py) ──


def make_actions(strengths=(0.25, 0.5, 0.75, 1.0, 1.25, 1.4)):
    actions = [{"name": "s0_zero", "coef_led": 0, "coef_pol": 0}]
    seen = {(0, 0, 0, 0)}
    for s in strengths:
        L = int(round(25 * s))
        P = int(round(60 * s))
        led = {"off": (0, 0), "pos": (0, L), "rev": (L, 0)}
        pol = {"off": (0, 0), "pos": (0, P), "rev": (P, 0)}
        for led_name, (l0, l1) in led.items():
            for pol_name, (p0, p1) in pol.items():
                key = (l0, l1, p0, p1)
                if key in seen:
                    continue
                seen.add(key)
                actions.append(
                    {
                        "name": f"s{s:g}_led_{led_name}_pol_{pol_name}",
                        "coef_led": l1 - l0,
                        "coef_pol": p1 - p0,
                    }
                )
    return actions


ACTION_NAMES = [a["name"] for a in make_actions()]
THETA_REF = "s0.5_led_pos_pol_pos"  # clean training mechanism (reference action)


def build_X(R, B, Z, features):
    cols = {
        "red": R[:, 0],
        "green": R[:, 1],
        "blue": R[:, 2],
        "ir_2": B[:, 0],
        "vis_2": B[:, 1],
        "ir_3": Z[:, 0],
        "vis_3": Z[:, 1],
    }
    return np.column_stack([cols[f] for f in features])


def restore_from_cube(df):
    R = df[["red", "green", "blue"]].values
    Y = df["Y"].values.astype(int)
    B = {n: df[[f"ir_2_{n}", f"vis_2_{n}"]].values for n in ACTION_NAMES}
    Z = {n: df[[f"ir_3_{n}", f"vis_3_{n}"]].values for n in ACTION_NAMES}
    return R, Y, B, Z


def brier(y_true, p1):
    return float(np.mean((p1 - y_true) ** 2))


# ─── load data ────────────────────────────────────────────────────────────────

df_train = pd.read_csv(os.path.join(DATA_DIR, "d_spur_train.csv"))
df_train = df_train[df_train["E"].isin(TRAIN_ENVS)].reset_index(drop=True)
X_tr_all = df_train[ALL_FEATURES].values
y_tr_all = df_train["Y"].values.astype(int)
E_tr_all = df_train["E"].values.astype(int)
env_idx = {e: np.where(E_tr_all == e)[0] for e in TRAIN_ENVS}
min_per_env = min(len(v) for v in env_idx.values())
print(
    f"train: {len(df_train)} obs; per-env counts={[len(env_idx[e]) for e in TRAIN_ENVS]}"
)
SIZES = [n for n in SIZES if n <= min_per_env]
print(f"sizes per env: {SIZES}; reps={N_REPS}")

df_cube = pd.read_csv(
    os.path.join(DATA_DIR, "adversarial_action_cube_grid49_N1000.csv")
)
R_all, Y_all, B_by_action, Z_by_action = restore_from_cube(df_cube)

# fixed probe/eval split of the cube (same convention as the main experiment)
split_rng = np.random.default_rng(SPLIT_SEED)
perm = split_rng.permutation(len(Y_all))
n_probe = len(Y_all) // 2
probe_idx, eval_idx = perm[:n_probe], perm[n_probe:]
R_probe, R_eval, Y_eval = R_all[probe_idx], R_all[eval_idx], Y_all[eval_idx]


def evaluate_strongest(predict_proba_fn):
    """Follower picks the worst action over ALL 49 (probe), score predictor on eval."""
    ms = np.array(
        [
            predict_proba_fn(
                build_X(
                    R_probe,
                    B_by_action[a][probe_idx],
                    Z_by_action[a][probe_idx],
                    ALL_FEATURES,
                )
            )[:, 1].mean()
            for a in ACTION_NAMES
        ]
    )
    best = ACTION_NAMES[int(np.argmin(ms))]
    p_adv = predict_proba_fn(
        build_X(
            R_eval,
            B_by_action[best][eval_idx],
            Z_by_action[best][eval_idx],
            ALL_FEATURES,
        )
    )[:, 1]
    p_clean = predict_proba_fn(
        build_X(
            R_eval,
            B_by_action[THETA_REF][eval_idx],
            Z_by_action[THETA_REF][eval_idx],
            ALL_FEATURES,
        )
    )[:, 1]
    return brier(Y_eval, p_adv), brier(Y_eval, p_clean), float(p_adv.mean())


# ─── sweep over training size × repetitions ───────────────────────────────────

rows = []
for n in SIZES:
    for rep in range(N_REPS):
        sub_rng = np.random.default_rng(SUBSAMPLE_SEED + rep)
        idx = np.concatenate(
            [sub_rng.choice(env_idx[e], size=n, replace=False) for e in TRAIN_ENVS]
        )
        sc = StabilizedClassificationClassifier(
            invariance_test="tram_gcm",
            test_classifier_type="RF",
            pred_classifier_type="RF",
            random_state=SC_SEED + rep,
            n_jobs=8,
        )
        sc.fit(X_tr_all[idx], y_tr_all[idx], E_tr_all[idx])

        for method_name, method in [("SC", "ensemble"), ("IMP", "best")]:
            adv, clean, adv_ef = evaluate_strongest(
                lambda X, m=method, model=sc: model.predict_proba(X, method=m)
            )
            rows.append(
                {
                    "n_per_env": n,
                    "rep": rep,
                    "method": method_name,
                    "adv_brier": adv,
                    "clean_brier": clean,
                    "adv_ef": adv_ef,
                }
            )
        print(
            f"n={n:5d} rep={rep}: SC adv={rows[-2]['adv_brier']:.4f}  "
            f"IMP adv={rows[-1]['adv_brier']:.4f}",
            flush=True,
        )

res = pd.DataFrame(rows)
res.to_csv(os.path.join(DATA_DIR, "sc_vs_imp_samplesize.csv"), index=False)
print("Saved sc_vs_imp_samplesize.csv")


# ─── plot: deployment MSE at the strongest bound vs. n, SC vs. IMP (mean ± 95% CI) ──

COL = {"SC": "#E69F00", "IMP": "#44AA99"}
fig, ax = plt.subplots(figsize=(5.5, 4.0))
for name in ["SC", "IMP"]:
    g = res[res["method"] == name].groupby("n_per_env")["adv_brier"]
    ns = g.mean().index.values
    mean = g.mean().values
    ci = 1.96 * g.sem().values
    ax.plot(ns, mean, marker="o", color=COL[name], label=name)
    ax.fill_between(ns, mean - ci, mean + ci, color=COL[name], alpha=0.2)

ax.set_xscale("log")
ax.set_xticks(SIZES)
ax.set_xticklabels([str(n) for n in SIZES])
ax.set_xlabel("training observations per environment", fontsize=12)
ax.set_ylabel("deployment MSE under strongest\nintervention (bound = 1.0)", fontsize=12)
ax.legend(frameon=False, fontsize=12, title=f"mean ± 95% CI ({N_REPS} reps)")
plt.tight_layout()

out = os.path.join(DATA_DIR, "sc_vs_imp_samplesize")
plt.savefig(out + ".png", dpi=300)
plt.savefig(out + ".pdf")
plt.close()
print(f"Saved {out}.{{png,pdf}}")
