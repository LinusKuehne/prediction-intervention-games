import numpy as np
import pandas as pd
from aggregation import AGGREGATIONS


def rmse(ytrue, ypred):
    """Root Mean Squared Error."""
    mask = np.isfinite(ytrue) & np.isfinite(ypred)
    if not mask.any():
        return np.nan
    return np.sqrt(np.nanmean((ypred - ytrue) ** 2))


# Published LR baselines (E^0_{b,c} in the paper), used as denominators in
# the skill score. Indexed as LR_BASELINE_RMSE[target][agg][setting][scale].
# Settings use code names; scales use the table/display names (eval.py
# renames the 'spatial' aggregation to 'site-mean' before the leaderboard).
# Scales daily/monthly are intentionally absent: eval.py drops them.
LR_BASELINE_RMSE = {
    "ET": {
        "median": {
            "time-split": {
                "hourly": 6.0,
                "weekly": 2.8,
                "seasonal": 2.6,
                "anom": 1.9,
                "iav": 0.48,
                "site-mean": 1.2,
            },
            "spatial-easy40": {
                "hourly": 6.1,
                "weekly": 2.9,
                "seasonal": 2.6,
                "anom": 2.1,
                "iav": 0.48,
                "site-mean": 1.4,
            },
            "TA40": {
                "hourly": 7.5,
                "weekly": 3.1,
                "seasonal": 2.5,
                "anom": 2.3,
                "iav": 0.43,
                "site-mean": 1.5,
            },
        },
        "q90": {
            "time-split": {
                "hourly": 8.1,
                "weekly": 4.3,
                "seasonal": 4.2,
                "anom": 2.6,
                "iav": 0.76,
                "site-mean": 2.9,
            },
            "spatial-easy40": {
                "hourly": 8.6,
                "weekly": 5.5,
                "seasonal": 5.4,
                "anom": 2.8,
                "iav": 0.94,
                "site-mean": 3.9,
            },
            "TA40": {
                "hourly": 10.0,
                "weekly": 4.8,
                "seasonal": 4.3,
                "anom": 2.8,
                "iav": 0.88,
                "site-mean": 3.8,
            },
        },
    },
    "GPP": {
        "median": {
            "time-split": {
                "hourly": 4.3,
                "weekly": 2.1,
                "seasonal": 1.9,
                "anom": 1.1,
                "iav": 0.20,
                "site-mean": 0.89,
            },
            "spatial-easy40": {
                "hourly": 4.5,
                "weekly": 2.1,
                "seasonal": 1.7,
                "anom": 1.4,
                "iav": 0.40,
                "site-mean": 0.82,
            },
            "TA40": {
                "hourly": 4.6,
                "weekly": 1.9,
                "seasonal": 1.5,
                "anom": 0.86,
                "iav": 0.20,
                "site-mean": 1.2,
            },
        },
        "q90": {
            "time-split": {
                "hourly": 6.3,
                "weekly": 3.5,
                "seasonal": 2.9,
                "anom": 2.2,
                "iav": 0.69,
                "site-mean": 2.5,
            },
            "spatial-easy40": {
                "hourly": 6.0,
                "weekly": 3.5,
                "seasonal": 3.0,
                "anom": 2.2,
                "iav": 0.87,
                "site-mean": 2.4,
            },
            "TA40": {
                "hourly": 6.1,
                "weekly": 3.6,
                "seasonal": 3.3,
                "anom": 1.5,
                "iav": 0.60,
                "site-mean": 2.8,
            },
        },
    },
    "NEE": {
        "median": {
            "time-split": {
                "hourly": 4.0,
                "weekly": 1.7,
                "seasonal": 1.6,
                "anom": 0.93,
                "iav": 0.16,
                "site-mean": 0.90,
            },
            "spatial-easy40": {
                "hourly": 4.1,
                "weekly": 1.6,
                "seasonal": 1.4,
                "anom": 1.2,
                "iav": 0.22,
                "site-mean": 0.68,
            },
            "TA40": {
                "hourly": 4.5,
                "weekly": 1.8,
                "seasonal": 1.7,
                "anom": 0.81,
                "iav": 0.13,
                "site-mean": 1.4,
            },
        },
        "q90": {
            "time-split": {
                "hourly": 5.9,
                "weekly": 3.0,
                "seasonal": 2.5,
                "anom": 1.9,
                "iav": 0.42,
                "site-mean": 2.1,
            },
            "spatial-easy40": {
                "hourly": 5.8,
                "weekly": 3.1,
                "seasonal": 3.0,
                "anom": 1.8,
                "iav": 0.60,
                "site-mean": 2.3,
            },
            "TA40": {
                "hourly": 5.6,
                "weekly": 3.0,
                "seasonal": 2.8,
                "anom": 1.2,
                "iav": 0.37,
                "site-mean": 2.5,
            },
        },
    },
}

# eval.py renames 'spatial' to 'site-mean' for the leaderboard; mirror that
# when looking up baselines.
_SCALE_ALIASES = {"spatial": "site-mean"}


def skill_score(
    predictions_df=None,
    target="ET",
    setting="TA40",
    agg="median",
    return_per_scale=False,
):
    """
    Skill score relative to linear regression, averaged across temporal scales.

    For each scale c, aggregates predictions with AGGREGATIONS[c], computes
    per-domain RMSE, summarises across domains with `agg` (median or 90th
    quantile) to get E^m_{b,c}, then averages 1 - E^m_{b,c} / E^0_{b,c}
    over the scales available in LR_BASELINE_RMSE. This matches the per-setting
    analog of S^m_overall in the paper (equal weights, same scales as
    get_weighted_skill_scores).

    Args:
        y_true, y_pred: 1D arrays of true and predicted values.
        env: 1D array of domain (site or site-year) ids, aligned with y_true.
        time: 1D array of timestamps (datetime-like), aligned with y_true.
        target: 'ET', 'GPP', or 'NEE'.
        setting: 'time-split', 'spatial-easy40', or 'TA40'.
        agg: 'median' or 'q90'.
        return_per_scale: if True, also return a {scale: skill} dict.

    Returns:
        float skill score (mean across scales), or (float, dict) if
        return_per_scale=True.
    """
    df = predictions_df[["y_true", "y_pred", "env", "time"]].copy()
    df["time"] = pd.to_datetime(df["time"])
    print(df.head())

    try:
        baselines = LR_BASELINE_RMSE[target][agg][setting]
    except KeyError as err:
        raise KeyError(
            f"No LR baseline for target={target!r}, agg={agg!r}, setting={setting!r}."
        ) from err

    summarise = np.median if agg == "median" else (lambda x: np.quantile(x, 0.9))

    per_scale = {}
    for fn_name, agg_fn in AGGREGATIONS.items():
        table_name = _SCALE_ALIASES.get(fn_name, fn_name)
        if table_name not in baselines:
            continue
        agg_df = agg_fn(df)
        print(f"Aggregated predictions for scale {table_name}:")
        print(agg_df.head())
        per_env = (
            agg_df.groupby("env")
            .apply(
                lambda g: rmse(g["y_true"].values, g["y_pred"].values),
                include_groups=False,
            )
            .dropna()
        )
        print(f"Per-domain RMSE for scale {table_name}:")
        print(per_env.head())
        factor = (
            1 / 100 if target == "ET" else 1.0
        )  # ET RMSEs are multiplied by 100 in eval.py
        per_scale[table_name] = (
            1 - summarise(per_env.values) / (factor * baselines[table_name])
            if not per_env.empty
            else np.nan
        )
        print(f"Skill score for scale {table_name}: {per_scale[table_name]:.4f}")

    valid = [v for v in per_scale.values() if not np.isnan(v)]
    overall = float(np.mean(valid)) if valid else float("nan")

    return (overall, per_scale) if return_per_scale else overall
