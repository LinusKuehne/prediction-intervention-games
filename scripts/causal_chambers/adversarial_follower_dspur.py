"""Adversarial follower experiment for D-spur."""

import os
import time

import causalchamber.lab as lab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from csc_baseline import CausalStrategicClassifier
from nn_baselines import NNBaselineClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from utils import sample_truncnorm_integers, wait_for_completion

from stabilized_classification import StabilizedClassificationClassifier

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SEED = 123
N_SAMPLES = 1000  # split equally into probe and eval
THRESHOLD = 12500  # Y = 1{ir_1 > THRESHOLD}, same as D-spur
BUDGETS = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

SB_FEATURES = ["red", "green", "blue"]
SB_IR2_FEATURES = ["red", "green", "blue", "ir_2"]
SB_B_FEATURES = ["red", "green", "blue", "ir_2", "vis_2"]
ALL_FEATURES = ["red", "green", "blue", "ir_2", "vis_2", "ir_3", "vis_3"]

PRED_COLORS = {
    "f_sb": "#0072B2",
    "f_sb_ir2": "#CC79A7",
    "f_sb_b": "#D55E00",
    "f_all": "#009E73",
    "f_sc": "#E69F00",
    "f_cserm": "#984EA3",
    "f_cserm_lo": "#A65628",
    "f_cserm_001": "#E7298A",
    "f_cserm_0": "#666666",
    "f_imp": "#44AA99",
    "f_erm_lin": "#332288",
    "f_erm_deep": "#332288",
    "f_irm_lin": "#882255",
    "f_irm_deep": "#882255",
    "f_dro_lin": "#999933",
    "f_dro_deep": "#999933",
}
PRED_LABELS = {
    "f_sb": r"$\hat{f}_{\rm RGB}$ (wrong stable blanket)",
    "f_sb_ir2": r"$\hat{f}_{\rm RGB + ir\_2}$",
    "f_sb_b": r"$\hat{f}_{\rm RGB + ir\_2 + vis\_2}$",
    "f_all": r"$\hat{f}_{\rm all}$",
    "f_sc": r"$\hat{f}^{\;\mathrm{SC}}$",
    "f_cserm": r"$\hat{f}^{\,\mathrm{CSERM}}_{0.05}$",
    "f_cserm_lo": r"$\hat{f}^{\,\mathrm{CSERM}}_{0.005}$",
    "f_cserm_001": r"$\hat{f}^{\,\mathrm{CSERM}}_{0.001}$",
    "f_cserm_0": r"$\hat{f}^{\,\mathrm{CSERM}}_{0}$",
    "f_imp": r"$\hat{f}^{\;\mathrm{IMP}}$",
    "f_erm_lin": r"ERM (lin)",
    "f_erm_deep": r"ERM (deep)",
    "f_irm_lin": r"IRM (lin)",
    "f_irm_deep": r"IRM (deep)",
    "f_dro_lin": r"GroupDRO (lin)",
    "f_dro_deep": r"GroupDRO (deep)",
}
# Linestyle per predictor (NN pairs share a color; linear=dashed, deep=solid).
PRED_STYLES = {
    "f_erm_lin": "--",
    "f_erm_deep": "-",
    "f_irm_lin": "--",
    "f_irm_deep": "-",
    "f_dro_lin": "--",
    "f_dro_deep": "-",
}


# ─── action menu ──────────────────────────────────────────────────────────────


def make_actions(strengths=(0.25, 0.5, 0.75, 1.0, 1.25, 1.4)):
    """Build the finite action menu.

    Each entry records (base_led, coef_led, base_pol, coef_pol) and the
    (strength, led_pattern, pol_pattern) metadata used for budget filtering.
    """
    actions = [
        {
            "name": "s0_zero",
            "base_led": 0,
            "coef_led": 0,
            "base_pol": 0,
            "coef_pol": 0,
            "strength": 0.0,
            "led_pattern": "off",
            "pol_pattern": "off",
        }
    ]
    seen = {(0, 0, 0, 0)}  # keys are (l0, l1, p0, p1)

    for s in strengths:
        L = int(round(25 * s))
        P = int(round(60 * s))
        led_pairs = {"off": (0, 0), "pos": (0, L), "rev": (L, 0)}
        pol_pairs = {"off": (0, 0), "pos": (0, P), "rev": (P, 0)}

        for led_name, (l0, l1) in led_pairs.items():
            for pol_name, (p0, p1) in pol_pairs.items():
                key = (l0, l1, p0, p1)
                if key in seen:
                    continue
                seen.add(key)
                actions.append(
                    {
                        "name": f"s{s:g}_led_{led_name}_pol_{pol_name}",
                        "base_led": l0,
                        "coef_led": l1 - l0,
                        "base_pol": p0,
                        "coef_pol": p1 - p0,
                        "strength": float(s),
                        "led_pattern": led_name,
                        "pol_pattern": pol_name,
                    }
                )

    return actions


ACTIONS = make_actions()
M = len(ACTIONS)
ACTION_NAMES = [a["name"] for a in ACTIONS]
print(f"Action menu: {M} actions  ({[a['name'] for a in ACTIONS]})")


# ─── helpers ──────────────────────────────────────────────────────────────────


def restore_from_cube(df):
    """Reconstruct R_all, Y_all, B_by_action, Z_by_action from the saved action cube."""
    R = df[["red", "green", "blue"]].values
    Y = df["Y"].values.astype(int)
    B = {n: df[[f"ir_2_{n}", f"vis_2_{n}"]].values for n in ACTION_NAMES}
    Z = {n: df[[f"ir_3_{n}", f"vis_3_{n}"]].values for n in ACTION_NAMES}
    return R, Y, B, Z


# ─── reference mechanism and distance ─────────────────────────────────────────

# θ_ref = training environment: coef_led=12 (led_3_ir = 12·Y), coef_pol=30 (pol_2 = 30·Y).
THETA_REF = "s0.5_led_pos_pol_pos"
REF_IDX = ACTION_NAMES.index(THETA_REF)
_REF_COEF_LED = ACTIONS[REF_IDX]["coef_led"]  # 12
_REF_COEF_POL = ACTIONS[REF_IDX]["coef_pol"]  # 30


def mechanism_distance(action):
    """Distance from θ_ref in coefficient space: max(|Δcoef_led|/25, |Δcoef_pol|/60).

    Reversal actions (coef < 0) land at d ≥ 0.72 and are only reachable at high budget.
    """
    return float(
        max(
            abs(action["coef_led"] - _REF_COEF_LED) / 25.0,
            abs(action["coef_pol"] - _REF_COEF_POL) / 60.0,
        )
    )


# Precomputed distances; budget b selects the round(b*M) closest actions.
_action_distances = np.array([mechanism_distance(a) for a in ACTIONS])
_dist_rank = list(np.argsort(_action_distances))


def eligible_by_budget(b: float) -> list[int]:
    """Return indices of the round(b*M) actions closest to θ_ref (min 1)."""
    k = max(1, round(b * M))
    return _dist_rank[:k]


for budget in BUDGETS:
    eligible_idx = eligible_by_budget(budget)
    thr = _action_distances[eligible_idx[-1]]
    print(f"\nBudget {budget} ({len(eligible_idx)}/{M} actions, max d={thr:.3f}):")
    for i in eligible_idx:
        print(f"  {ACTION_NAMES[i]:35s} d={mechanism_distance(ACTIONS[i]):.3f}")


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


def eval_metrics(y_true, p, threshold=0.5):
    y1 = y_true == 1
    y0 = y_true == 0
    return {
        "brier": float(np.mean((p - y_true) ** 2)),
        "acc": float(np.mean((p >= threshold) == y_true)),
        "bce": float(log_loss(y_true, p)),
        "ef": float(p.mean()),
        "ef_y1": float(p[y1].mean()) if y1.any() else float("nan"),
        "ef_y0": float(p[y0].mean()) if y0.any() else float("nan"),
        "fnr": float(np.mean(p[y1] < threshold)) if y1.any() else float("nan"),
    }


# ─── train predictors on existing D-spur training data ────────────────────────

rlab = lab.Lab(os.path.join(DIR, ".credentials"))

df_train_full = pd.read_csv(os.path.join(DATA_DIR, "d_spur_train.csv"))
TRAIN_ENVS = [0, 1, 2]
df_train_full = df_train_full[df_train_full["E"].isin(TRAIN_ENVS)].reset_index(
    drop=True
)
E_full = np.asarray(df_train_full["E"].values, dtype=int)
y_full = np.asarray(df_train_full["Y"].values, dtype=int)
print(f"Training on environments {TRAIN_ENVS}: {len(y_full)} obs")

X_train_sb = df_train_full[SB_FEATURES].values
X_train_sb_ir2 = df_train_full[SB_IR2_FEATURES].values
X_train_sb_b = df_train_full[SB_B_FEATURES].values
X_train_all = df_train_full[ALL_FEATURES].values

f_sb = RandomForestClassifier(n_estimators=100, random_state=SEED)
f_sb.fit(X_train_sb, y_full)
print(f"Trained f_sb     on {SB_FEATURES}  (n={len(y_full)})")

f_sb_ir2 = RandomForestClassifier(n_estimators=100, random_state=SEED)
f_sb_ir2.fit(X_train_sb_ir2, y_full)
print(f"Trained f_sb_ir2 on {SB_IR2_FEATURES}  (n={len(y_full)})")

f_sb_b = RandomForestClassifier(n_estimators=100, random_state=SEED)
f_sb_b.fit(X_train_sb_b, y_full)
print(f"Trained f_sb_b   on {SB_B_FEATURES}  (n={len(y_full)})")

f_all = RandomForestClassifier(n_estimators=100, random_state=SEED)
f_all.fit(X_train_all, y_full)
print(f"Trained f_all   on {ALL_FEATURES}  (n={len(y_full)})")

f_sc = StabilizedClassificationClassifier(
    invariance_test="tram_gcm",
    test_classifier_type="RF",
    pred_classifier_type="RF",
    random_state=SEED,
    n_jobs=8,
)

f_sc.fit(X_train_all, y_full, E_full)
print(f"Trained f_sc   (SC TramGCM RF, n={len(y_full)})")

# ─── inspect f_sc ensemble ────────────────────────────────────────────────────
print(
    f"\nf_sc ensemble: {f_sc.n_predictive_subsets_} active / "
    f"{f_sc.n_invariant_subsets_} invariant / {f_sc.n_subsets_total_} total subsets"
)
sb_b_idx_set = frozenset(ALL_FEATURES.index(f) for f in SB_B_FEATURES)
active_subsets_named = []
for stat in sorted(f_sc.active_subsets_, key=lambda s: -s["score"]):
    names = [ALL_FEATURES[i] for i in stat["subset"]]
    marker = "  <-- == SB_B" if frozenset(stat["subset"]) == sb_b_idx_set else ""
    print(
        f"  w={stat['weight']:.3f}  score={stat['score']:+.4f}  "
        f"p={stat['p_value']:.3f}  {names}{marker}"
    )
    active_subsets_named.append(names)

print("\nAll invariant subsets (sorted by score):")
for stat in sorted(f_sc._all_invariant_fitted_, key=lambda s: -s["score"]):
    names = [ALL_FEATURES[i] for i in stat["subset"]]
    in_active = any(stat["subset"] == a["subset"] for a in f_sc.active_subsets_)
    marker = "  *active*" if in_active else ""
    marker += "  <-- == SB_B" if frozenset(stat["subset"]) == sb_b_idx_set else ""
    print(f"  score={stat['score']:+.4f}  p={stat['p_value']:.3f}  {names}{marker}")

sb_b_in_invariant = any(
    frozenset(stat["subset"]) == sb_b_idx_set for stat in f_sc._all_invariant_fitted_
)
sb_b_in_active = any(
    frozenset(stat["subset"]) == sb_b_idx_set for stat in f_sc.active_subsets_
)
print(
    f"\nSB_B = {SB_B_FEATURES}: in invariant? {sb_b_in_invariant}; "
    f"in active ensemble? {sb_b_in_active}"
)

# ─── CSERM (clean-data) baselines ────────────────────────────────────────────
# Linear strategic classifiers (Horowitz & Rosenfeld, 2023) that anticipate
# manipulation of the only movable sensors (ir_3, vis_3). See csc_baseline.py.
# The cost-grid floor (smallest multiple of the calibrated alpha_0) sets how
# aggressively CSERM down-weights the manipulable features: worst-env Brier decreases
# monotonically as alpha shrinks, so the floor controls where f_cserm lands on the
# continuum from f_all-like (gameable) to f_sb_b-like (stable blanket). We report
# two floors: 0.05 (moderate, ~log-symmetric around alpha_0) and 0.005 (aggressive).
CSERM_MULTS_05 = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
CSERM_MULTS_005 = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

f_cserm = CausalStrategicClassifier(
    features=ALL_FEATURES,
    causal_features=["red", "green", "blue"],
    movable_features=["ir_3", "vis_3"],
    alpha_mults=CSERM_MULTS_05,
    random_state=SEED,
)
f_cserm.fit(X_train_all, y_full, E_full)
print(
    f"Trained f_cserm     (CSERM clean-data, floor 0.05; "
    f"alpha0={f_cserm.alpha0_:.3g}, alpha*={f_cserm.alpha_:.3g})"
)

f_cserm_lo = CausalStrategicClassifier(
    features=ALL_FEATURES,
    causal_features=["red", "green", "blue"],
    movable_features=["ir_3", "vis_3"],
    alpha_mults=CSERM_MULTS_005,
    random_state=SEED,
)
f_cserm_lo.fit(X_train_all, y_full, E_full)
print(
    f"Trained f_cserm_lo  (CSERM clean-data, floor 0.005; "
    f"alpha0={f_cserm_lo.alpha0_:.3g}, alpha*={f_cserm_lo.alpha_:.3g})"
)

# Two further floors down the continuum: 0.001, and 0. alpha=0 forces ||w_movable||=0,
# i.e. CSERM drops ir_3/vis_3 entirely -> a linear classifier on the stable blanket. We
# pin f_cserm_0 to alpha=0 (single-point grid): worst-env Brier is actually slightly worse
# at exactly 0 than at tiny alpha>0, so a grid including 0 would never select it, yet we
# want to display this endpoint explicitly.
CSERM_MULTS_001 = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
)
CSERM_MULTS_0 = (0.0,)

f_cserm_001 = CausalStrategicClassifier(
    features=ALL_FEATURES,
    causal_features=["red", "green", "blue"],
    movable_features=["ir_3", "vis_3"],
    alpha_mults=CSERM_MULTS_001,
    random_state=SEED,
)
f_cserm_001.fit(X_train_all, y_full, E_full)
print(
    f"Trained f_cserm_001 (CSERM clean-data, floor 0.001; "
    f"alpha0={f_cserm_001.alpha0_:.3g}, alpha*={f_cserm_001.alpha_:.3g})"
)

f_cserm_0 = CausalStrategicClassifier(
    features=ALL_FEATURES,
    causal_features=["red", "green", "blue"],
    movable_features=["ir_3", "vis_3"],
    alpha_mults=CSERM_MULTS_0,
    random_state=SEED,
)
f_cserm_0.fit(X_train_all, y_full, E_full)
print(
    f"Trained f_cserm_0   (CSERM clean-data, floor 0 = drops ir_3/vis_3; "
    f"alpha0={f_cserm_0.alpha0_:.3g}, alpha*={f_cserm_0.alpha_:.3g})"
)


# ─── IMP (invariant most-predictive): reuse the fitted SC model ───────────────
# IMP predicts with the single highest-predictiveness invariant subset, which the
# StabilizedClassification model already exposes via predict_proba(method="best").
# No extra training: f_imp wraps the already-fitted f_sc.
class _SCMethodAdapter:
    """Expose a fitted StabilizedClassification under a fixed predict method."""

    def __init__(self, sc_model, method):
        self._sc = sc_model
        self._method = method

    def predict_proba(self, X):
        return self._sc.predict_proba(X, method=self._method)


f_imp = _SCMethodAdapter(f_sc, "best")
print("Built f_imp   (IMP = SC single best invariant subset; reuses f_sc)")


# ─── NN baselines: ERM / IRM / GroupDRO (deep) ────────────────────────────────
# ERM-NN (all features) isolates RandomForest-vs-NN against f_all. IRM uses the bounded
# convex penalty (lambda in [0, 1]); all select hyperparameters by worst-environment
# Brier and are seed-ensembled over 3 seeds. See nn_baselines.py. The linear variants are
# implemented there but not fit here (they nearly coincide with their deep/RF counterparts:
# linear ERM/IRM are gamed like f_all, and linear IRM selects lambda~0.1 -> approx ERM).
#
# Why IRM (deep) is ~flat AND high (~0.20) in the deployment-MSE (right) panel: selection
# picks the largest lambda (~0.99; "heavy invariance" -- see the printed cfg below). It is
# NOT a constant predictor (outputs are bimodal, std~0.31, and it does respond to ir_3).
# Rather, with the train->cube shift it is biased high and over-predicts the positive class
# (cube E[f|Y=0]~0.57, should be 0). The chamber feedback scales with Y (led_3_ir, pol_2
# proportional to Y), so the follower can only manipulate the Y=1 samples -- the Y=0 samples
# are fixed. (HIGH) the over-prediction on the unmovable Y=0 class pins Brier ~0.20 and the
# follower cannot touch it; (FLAT) on the Y=1 samples IRM-deep keeps predicting positive
# under attack (ir_3 ~1275->67 yet it does not flip them, unlike the RF), so E[f] barely
# moves. Miscalibrated-but-unfoolable, not constant.
NN_METHODS = [
    ("f_erm_deep", "erm", "deep"),
    ("f_irm_deep", "irm", "deep"),
    ("f_dro_deep", "groupdro", "deep"),
]
nn_models = {}
for _nn_name, _nn_method, _nn_arch in NN_METHODS:
    _clf = NNBaselineClassifier(
        method=_nn_method,
        architecture=_nn_arch,
        n_seeds=3,
        n_jobs=8,
        random_state=SEED,
    )
    _clf.fit(X_train_all, y_full, E_full)
    nn_models[_nn_name] = _clf
    print(
        f"Trained {_nn_name:11s} ({_nn_method}-{_nn_arch}; "
        f"epochs*={_clf.best_n_epochs_}, worstValBrier={_clf.best_worst_val_brier_:.4f}; "
        f"cfg={_clf.best_config_})"
    )

PREDICTORS = {
    "f_sb": (f_sb, SB_FEATURES),
    "f_sb_ir2": (f_sb_ir2, SB_IR2_FEATURES),
    "f_sb_b": (f_sb_b, SB_B_FEATURES),
    "f_all": (f_all, ALL_FEATURES),
    "f_sc": (f_sc, ALL_FEATURES),
    "f_cserm": (f_cserm, ALL_FEATURES),
    "f_cserm_lo": (f_cserm_lo, ALL_FEATURES),
    "f_cserm_001": (f_cserm_001, ALL_FEATURES),
    "f_cserm_0": (f_cserm_0, ALL_FEATURES),
    "f_imp": (f_imp, ALL_FEATURES),
    **{name: (nn_models[name], ALL_FEATURES) for name, _, _ in NN_METHODS},
}


# ─── data collection ──────────────────────────────────────────────────────────

CUBE_PATH = os.path.join(DATA_DIR, "adversarial_action_cube_grid49_N1000.csv")

if os.path.exists(CUBE_PATH):
    print(f"\nLoading action cube from {CUBE_PATH} ...")
    df_cube = pd.read_csv(CUBE_PATH)
    R_all, Y_all, B_by_action, Z_by_action = restore_from_cube(df_cube)
    print(f"Restored: N={len(Y_all)}, M={M}, P(Y=1)={Y_all.mean():.3f}")
else:
    rgb_inputs = {
        "red": sample_truncnorm_integers(
            N_SAMPLES, mean=64, std=20, low=0, high=255, random_state=SEED + 11
        ),
        "green": sample_truncnorm_integers(
            N_SAMPLES, mean=32, std=30, low=0, high=255, random_state=SEED + 12
        ),
        "blue": sample_truncnorm_integers(
            N_SAMPLES, mean=90, std=12, low=0, high=255, random_state=SEED + 13
        ),
    }
    R_all = np.column_stack(
        [rgb_inputs["red"], rgb_inputs["green"], rgb_inputs["blue"]]
    )

    # phase 1: set RGB, measure ir_1 -> Y
    print("\nSubmitting phase 1...")
    exp_p1 = rlab.new_experiment(chamber_id="lt-test-b0ni", config="standard")
    exp_p1.from_df(pd.DataFrame(rgb_inputs))
    eid_p1 = exp_p1.submit(tag="adv_follower_phase1")
    time.sleep(2)

    print("Waiting for phase 1...")
    wait_for_completion(rlab)

    df_p1 = rlab.download_data(eid_p1, root=os.path.join(DIR, "tmp")).dataframe
    Y_all = np.where(df_p1["ir_1"].values > THRESHOLD, 1, 0)
    print(f"Phase 1 complete. N={N_SAMPLES}, P(Y=1)={Y_all.mean():.3f}")

    # phase 2: one experiment per action; same RGB, Y-dependent led_3_ir / pol_2
    print(f"\nSubmitting {M} phase 2 experiments...")
    eids_p2 = {}
    for action in ACTIONS:
        inputs_p2 = {k: v.copy() for k, v in rgb_inputs.items()}
        inputs_p2["led_3_ir"] = np.clip(
            action["base_led"] + Y_all * action["coef_led"], 0, None
        ).astype(int)
        inputs_p2["pol_2"] = np.clip(
            action["base_pol"] + Y_all * action["coef_pol"], 0, None
        ).astype(int)

        exp_p2 = rlab.new_experiment(chamber_id="lt-test-b0ni", config="standard")
        exp_p2.from_df(pd.DataFrame(inputs_p2))
        safe_name = action["name"].replace(".", "p")
        eid = exp_p2.submit(tag=f"adv_follower_phase2_{safe_name}")
        eids_p2[action["name"]] = eid
        time.sleep(2)

    print("Waiting for phase 2...")
    wait_for_completion(rlab)

    B_by_action = {}
    Z_by_action = {}
    for action_name, eid in eids_p2.items():
        df_p2 = rlab.download_data(eid, root=os.path.join(DIR, "tmp")).dataframe
        B_by_action[action_name] = df_p2[["ir_2", "vis_2"]].values
        Z_by_action[action_name] = df_p2[["ir_3", "vis_3"]].values

    print(f"Phase 2 complete. Action cube: {N_SAMPLES} units x {M} actions.")

    # save action cube
    df_cube = pd.DataFrame(
        {
            "Y": Y_all,
            "red": R_all[:, 0],
            "green": R_all[:, 1],
            "blue": R_all[:, 2],
            **{f"ir_2_{n}": B_by_action[n][:, 0] for n in ACTION_NAMES},
            **{f"vis_2_{n}": B_by_action[n][:, 1] for n in ACTION_NAMES},
            **{f"ir_3_{n}": Z_by_action[n][:, 0] for n in ACTION_NAMES},
            **{f"vis_3_{n}": Z_by_action[n][:, 1] for n in ACTION_NAMES},
        }
    )
    df_cube.to_csv(CUBE_PATH, index=False)
    print("Saved action cube.")


# ─── probe / eval split ───────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)

N = len(Y_all)
perm = rng.permutation(N)
n_probe = N // 2
probe_idx = perm[:n_probe]
eval_idx = perm[n_probe:]

R_probe = R_all[probe_idx]


# ─── probe mean scores for all predictors x all actions ──────────────────────

# probe_mean_score[pred_name] has shape (M,):
#   probe_mean_score[pred_name][m] = mean E[f(X^m)] on probe split.

print("\n─── Computing probe mean scores ───")
probe_mean_score = {}
for pred_name, (predictor, features) in PREDICTORS.items():
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
    print(f"  {pred_name}: done")


# ─── fixed-action diagnostics (eval split) ───────────────────────────────────

print("\n─── Fixed-action diagnostics (eval split) ───")
print(
    f"{'predictor':<10}  {'action':<32}  {'Brier':>8}  {'Acc':>8}  {'BCE':>8}  "
    f"{'E[f]':>8}  {'E[f|Y=1]':>10}  {'E[f|Y=0]':>10}  {'FNR':>8}"
)
print("─" * 100)

R_eval = R_all[eval_idx]
Y_eval = Y_all[eval_idx]

fixed_rows = []
for pred_name, (predictor, features) in PREDICTORS.items():
    for action_name in ACTION_NAMES:
        X_m = build_X(
            R_eval,
            B_by_action[action_name][eval_idx],
            Z_by_action[action_name][eval_idx],
            features,
        )
        p_m = predictor.predict_proba(X_m)[:, 1]
        met = eval_metrics(Y_eval, p_m)
        print(
            f"{pred_name:<10}  {action_name:<32}  {met['brier']:>8.4f}  {met['acc']:>8.4f}  {met['bce']:>8.4f}  "
            f"{met['ef']:>8.4f}  {met['ef_y1']:>10.4f}  {met['ef_y0']:>10.4f}  {met['fnr']:>8.4f}"
        )
        fixed_rows.append({"predictor": pred_name, "action": action_name, **met})
    print()

pd.DataFrame(fixed_rows).to_csv(
    os.path.join(DATA_DIR, "fixed_action_results.csv"), index=False
)
print("Saved fixed_action_results.csv")


# ─── best-response by budget (eval split) ────────────────────────────────────

print("\n─── Best-response by budget (eval split) ───")
print(
    f"{'pred':<10}  {'budget':>7}  {'clean Brier':>12}  {'adv Brier':>12}  "
    f"{'clean E[f]':>12}  {'adv E[f]':>12}  {'adv FNR':>10}"
)
print("─" * 90)

result_rows = []

for budget in BUDGETS:
    eligible_idx = eligible_by_budget(budget)

    for pred_name, (predictor, features) in PREDICTORS.items():
        ms_eligible = probe_mean_score[pred_name][eligible_idx]
        best_m = eligible_idx[int(np.argmin(ms_eligible))]  # global action index
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

        print(
            f"{pred_name:<10}  {budget:>7.1f}  {met_clean['brier']:>12.4f}  {met_adv['brier']:>12.4f}  "
            f"{met_clean['ef']:>12.4f}  {met_adv['ef']:>12.4f}  {met_adv['fnr']:>10.4f}"
        )
        print(f"    selected action: {best_action}")
        result_rows.append(
            {
                "predictor": pred_name,
                "budget": budget,
                **{f"clean_{k}": v for k, v in met_clean.items()},
                **{f"adv_{k}": v for k, v in met_adv.items()},
                "best_action": best_action,
            }
        )

df_budget = pd.DataFrame(result_rows)
df_budget.to_csv(
    os.path.join(DATA_DIR, "adversarial_results_by_budget.csv"), index=False
)
print("\nSaved adversarial_results_by_budget.csv")


# ─── plots ────────────────────────────────────────────────────────────────────

df_plot = df_budget[df_budget["budget"] <= 0.5]


def make_budget_plot(pred_names, plot_base):
    """Budget-curve figure (Δ E[f] and adversarial Brier) for a subset of predictors."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), sharey=False)

    for pred_name in pred_names:
        sub = df_plot[df_plot["predictor"] == pred_name].sort_values("budget")
        color = PRED_COLORS[pred_name]
        label = PRED_LABELS[pred_name]
        style = PRED_STYLES.get(pred_name, "-")
        delta_ef = sub["adv_ef"] - sub["clean_ef"]
        axes[0].plot(
            sub["budget"],
            delta_ef,
            marker="o",
            color=color,
            linestyle=style,
            label=label,
        )
        axes[1].plot(
            sub["budget"],
            sub["adv_brier"],
            marker="o",
            color=color,
            linestyle=style,
            label=label,
        )
        axes[1].axhline(
            sub["clean_brier"].iloc[0], color=color, linestyle=":", alpha=0.35
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
    n = len(pred_names)
    ncol = n if n <= 5 else int(np.ceil(n / 2))  # wrap to 2 rows when crowded
    n_rows = int(np.ceil(n / ncol))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=ncol,
        fontsize=14,
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0.10 + 0.07 * n_rows, 1, 1])

    plt.savefig(plot_base + ".png", dpi=300)
    plt.savefig(plot_base + ".pdf")
    plt.close()
    print(f"Saved plot to {plot_base}.{{png,pdf}}")


# Figure 1: the methods already in the paper (unchanged).
make_budget_plot(
    ["f_sb", "f_sb_ir2", "f_sb_b", "f_all", "f_sc"],
    os.path.join(DATA_DIR, "adversarial_budget_curves"),
)

# Figure 2: causal / invariance baselines (CSERM variants + IMP) vs. stabilized classification.
make_budget_plot(
    ["f_cserm", "f_cserm_lo", "f_cserm_001", "f_cserm_0", "f_imp", "f_sc"],
    os.path.join(DATA_DIR, "adversarial_budget_curves_baselines"),
)

# Figure 3: NN baselines (ERM / IRM / GroupDRO, deep) vs. stabilized classification.
make_budget_plot(
    ["f_erm_deep", "f_irm_deep", "f_dro_deep", "f_sc"],
    os.path.join(DATA_DIR, "adversarial_budget_curves_nn"),
)
