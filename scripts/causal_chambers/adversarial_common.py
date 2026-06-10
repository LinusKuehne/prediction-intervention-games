"""Shared constants and helpers for the adversarial-follower D-spur experiment.

Imported by both the cube-generation script (`generate_adversarial_cubes.py`)
and the analysis script (`adversarial_follower_dspur.py`).
"""

import os

import numpy as np
from sklearn.metrics import log_loss

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "data")

SEED = 123
N_SAMPLES = 1000  # split equally into probe and eval
THRESHOLD = 12500  # Y = 1{ir_1 > THRESHOLD}, same as D-spur
N_REPS = 10
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
}
PRED_LABELS = {
    "f_sb": r"$\hat{f}_{\rm RGB}$ (wrong stable blanket)",
    "f_sb_ir2": r"$\hat{f}_{\rm RGB + ir\_2}$",
    "f_sb_b": r"$\hat{f}_{\rm RGB + ir\_2 + vis\_2}$",
    "f_all": r"$\hat{f}_{\rm all}$",
    "f_sc": r"$\hat{f}^{\;\mathrm{SC}}$",
}


# ─── per-rep cube path / seeds ────────────────────────────────────────────────


def cube_path(rep: int) -> str:
    """Filename of the action cube for a given repetition.

    rep 0 is the cube originally collected with seeds (SEED+11, SEED+12, SEED+13);
    the historical file was renamed to ..._rep0.csv so all reps share one scheme.
    """
    return f"adversarial_action_cube_grid49_N1000_rep{rep}.csv"


def rgb_seeds(rep: int) -> tuple[int, int, int]:
    """RGB truncnorm seeds for a rep. rep 0 -> (SEED+11, SEED+12, SEED+13)."""
    base = SEED + 1000 * rep
    return base + 11, base + 12, base + 13


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


# ─── cube / feature helpers ───────────────────────────────────────────────────


def restore_from_cube(df):
    """Reconstruct R_all, Y_all, B_by_action, Z_by_action from a saved action cube."""
    R = df[["red", "green", "blue"]].values
    Y = df["Y"].values.astype(int)
    B = {n: df[[f"ir_2_{n}", f"vis_2_{n}"]].values for n in ACTION_NAMES}
    Z = {n: df[[f"ir_3_{n}", f"vis_3_{n}"]].values for n in ACTION_NAMES}
    return R, Y, B, Z


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
