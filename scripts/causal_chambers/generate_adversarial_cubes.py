"""Generate the lab-collected action cubes for the adversarial-follower experiment.

One cube per repetition (N=1000 units x M actions). Each cube takes hours of lab
time, so cubes are saved as individual CSVs and the script is restartable: a rep
whose CSV already exists is skipped, so rerunning after a crash resumes at the
first missing cube. rep 0 is the historical cube (renamed to ..._rep0.csv).

Run this offline before running `adversarial_follower_dspur.py`.
"""

import os
import time

import causalchamber.lab as lab
import numpy as np
import pandas as pd
from adversarial_common import (
    ACTION_NAMES,
    ACTIONS,
    DATA_DIR,
    N_REPS,
    N_SAMPLES,
    THRESHOLD,
    M,
    cube_path,
    rgb_seeds,
)
from utils import sample_truncnorm_integers, wait_for_completion

DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(DIR, "tmp")  # download_data requires this dir to already exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

print(f"Action menu: {M} actions")


def generate_cube(rlab, rep):
    """Collect one action cube for a repetition and save it as its own CSV.

    Phase 1: set RGB, measure ir_1 -> Y. Phase 2: one experiment per action with
    the same RGB and Y-dependent led_3_ir / pol_2; record (ir_2, vis_2, ir_3,
    vis_3). The CSV is written only after the full cube is collected.
    """
    s_red, s_green, s_blue = rgb_seeds(rep)
    rgb_inputs = {
        "red": sample_truncnorm_integers(
            N_SAMPLES, mean=64, std=20, low=0, high=255, random_state=s_red
        ),
        "green": sample_truncnorm_integers(
            N_SAMPLES, mean=32, std=30, low=0, high=255, random_state=s_green
        ),
        "blue": sample_truncnorm_integers(
            N_SAMPLES, mean=90, std=12, low=0, high=255, random_state=s_blue
        ),
    }
    R_all = np.column_stack(
        [rgb_inputs["red"], rgb_inputs["green"], rgb_inputs["blue"]]
    )

    # phase 1: set RGB, measure ir_1 -> Y
    print(f"\n[rep {rep}] Submitting phase 1...")
    exp_p1 = rlab.new_experiment(chamber_id="lt-test-b0ni", config="standard")
    exp_p1.from_df(pd.DataFrame(rgb_inputs))
    eid_p1 = exp_p1.submit(tag=f"adv_follower_rep{rep}_phase1")
    time.sleep(2)

    print(f"[rep {rep}] Waiting for phase 1...")
    wait_for_completion(rlab)

    df_p1 = rlab.download_data(eid_p1, root=TMP_DIR).dataframe
    Y_all = np.where(df_p1["ir_1"].values > THRESHOLD, 1, 0)
    print(f"[rep {rep}] Phase 1 complete. N={N_SAMPLES}, P(Y=1)={Y_all.mean():.3f}")

    # phase 2: one experiment per action; same RGB, Y-dependent led_3_ir / pol_2
    print(f"[rep {rep}] Submitting {M} phase 2 experiments...")
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
        eid = exp_p2.submit(tag=f"adv_follower_rep{rep}_phase2_{safe_name}")
        eids_p2[action["name"]] = eid
        time.sleep(2)

    print(f"[rep {rep}] Waiting for phase 2...")
    wait_for_completion(rlab)

    B_by_action = {}
    Z_by_action = {}
    for action_name, eid in eids_p2.items():
        df_p2 = rlab.download_data(eid, root=TMP_DIR).dataframe
        B_by_action[action_name] = df_p2[["ir_2", "vis_2"]].values
        Z_by_action[action_name] = df_p2[["ir_3", "vis_3"]].values

    print(
        f"[rep {rep}] Phase 2 complete. Action cube: {N_SAMPLES} units x {M} actions."
    )

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
    out_path = os.path.join(DATA_DIR, cube_path(rep))
    df_cube.to_csv(out_path, index=False)
    print(f"[rep {rep}] Saved {out_path}")


def main():
    rlab = lab.Lab(os.path.join(DIR, ".credentials"))
    for rep in range(N_REPS):
        out_path = os.path.join(DATA_DIR, cube_path(rep))
        if os.path.exists(out_path):
            print(f"rep {rep}: {os.path.basename(out_path)} exists, skipping")
            continue
        generate_cube(rlab, rep)
    print("\nAll cubes present.")


if __name__ == "__main__":
    main()
