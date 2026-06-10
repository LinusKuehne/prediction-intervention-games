"""Why does the RandomForest f_all get gamed far harder than the deep ERM NN?

Both use all features and BOTH rely heavily on the manipulable sensor ir_3 (it is the
RF's top feature, and mean-imputing ir_3/vis_3 spikes the Brier of both). So the gap is
NOT a difference in which features they use -- it is the response *geometry* to ir_3:

  * f_all (RandomForest) is piecewise-constant: it places a sharp threshold on ir_3 and
    CLAMPS to a constant outside the training range. Nudging ir_3 across the threshold
    collapses the prediction to its low floor, and any further/extreme manipulation does
    nothing more (it is clamped). The follower therefore games it fully with a small,
    cheap push.

  * the deep NN is a smooth, saturating function: its prediction slides gradually with
    ir_3 and saturates at the ends. A bounded manipulation moves it only part-way, and
    extreme out-of-distribution values hit saturation (and can even reverse). The follower
    can push it only part-way down.

Consequently f_erm_deep's lower adversarial error is an artifact of NN smoothness / OOD
saturation, NOT principled invariance: both predictors would collapse to roughly the same
low value if the follower could drive ir_3 all the way to its floor. This is a statement
about the RandomForest-vs-NN function class under feature manipulation, independent of the
causal story. Running this script prints the ir_3 response curves that show the
step-and-clamp (RF) vs smooth-and-saturating (NN) shapes.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn_baselines import NNBaselineClassifier  # noqa: E402

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "data")
SEED = 123
ALL_FEATURES = ["red", "green", "blue", "ir_2", "vis_2", "ir_3", "vis_3"]
THETA_REF = "s0.5_led_pos_pol_pos"  # clean reference action
IR3_GRID = [0, 100, 300, 500, 740, 1000, 1500, 2000, 3000, 5000]

# ─── train f_all (RF) and f_erm_deep (NN) on the pooled training environments ──
df = pd.read_csv(os.path.join(DATA_DIR, "d_spur_train.csv"))
df = df[df["E"].isin([0, 1, 2])].reset_index(drop=True)
X = df[ALL_FEATURES].values
y = df["Y"].values.astype(int)
E = df["E"].values.astype(int)
print(
    "training ir_3: min=%.0f p5=%.0f median=%.0f p95=%.0f max=%.0f"
    % (
        df.ir_3.min(),
        df.ir_3.quantile(0.05),
        df.ir_3.median(),
        df.ir_3.quantile(0.95),
        df.ir_3.max(),
    )
)

f_all = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X, y)
f_erm_deep = NNBaselineClassifier(
    method="erm", architecture="deep", n_seeds=3, n_jobs=8, random_state=SEED
).fit(X, y, E)

# ─── base = clean (THETA_REF) features for the held-out eval half of the cube ──
cube = pd.read_csv(os.path.join(DATA_DIR, "adversarial_action_cube_grid49_N1000.csv"))
rng = np.random.default_rng(SEED)
eval_idx = rng.permutation(len(cube))[len(cube) // 2 :]
base = np.column_stack(
    [
        cube["red"].values,
        cube["green"].values,
        cube["blue"].values,
        cube[f"ir_2_{THETA_REF}"].values,
        cube[f"vis_2_{THETA_REF}"].values,
        cube[f"ir_3_{THETA_REF}"].values,
        cube[f"vis_3_{THETA_REF}"].values,
    ]
)[eval_idx]

# ─── sweep ir_3 (all eval samples set to v; other features held at clean) ──────
print("\nSweep ir_3 (all eval samples set to v, other features = clean), mean P(Y=1):")
print(f"{'ir_3=v':>8} {'f_all (RF)':>12} {'f_erm_deep (NN)':>16}")
ir3_col = ALL_FEATURES.index("ir_3")
for v in IR3_GRID:
    Xv = base.copy()
    Xv[:, ir3_col] = v
    p_rf = f_all.predict_proba(Xv)[:, 1].mean()
    p_nn = f_erm_deep.predict_proba(Xv)[:, 1].mean()
    print(f"{v:>8} {p_rf:>12.3f} {p_nn:>16.3f}")
