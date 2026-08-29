"""H1 — non-uniform degradation across information types.

Claim: retrieval accuracy declines at different rates for different fact types.
Test: per-type degradation slopes, and whether their intervals separate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ahnexp import config, metrics, schema, stats


def curves(df: pd.DataFrame, by: str = "fact_type") -> pd.DataFrame:
    """Accuracy against compression pressure, one curve per category."""
    schema.validate(df, needs=("core", "h1"))

    rows = []
    for (group_value, pressure), group in df.groupby([by, "tokens_after_target"]):
        low, high = stats.cluster_bootstrap_ci(group)
        window = int(group["sliding_window"].iloc[0])
        rows.append(
            {
                by: group_value,
                "tokens_after_target": int(pressure),
                "pressure_windows": pressure / window if window else np.nan,
                "memory_condition": group["memory_condition"].iloc[0],
                "accuracy": metrics.accuracy(group),
                "chance_corrected": metrics.chance_corrected_accuracy(group)
                if by == "fact_type" else np.nan,
                "ci_low": low,
                "ci_high": high,
                "n": int(len(group)),
            }
        )

    return pd.DataFrame(rows).sort_values([by, "tokens_after_target"]).reset_index(drop=True)


def slopes(df: pd.DataFrame, by: str = "fact_type", recurrent_only: bool = True) -> pd.DataFrame:
    """Degradation rate per category, with a clustered CI.

    Slope of logit(accuracy) against log2(pressure / window). Normalising the x-axis
    by the window makes the rate comparable across checkpoints; the logit keeps a
    drop from 0.9 to 0.8 from looking like a drop from 0.5 to 0.4.
    """
    subset = df[df["memory_condition"] == "recurrent_memory"] if recurrent_only else df
    subset = subset[subset["tokens_after_target"] > 0]

    rows = []
    for group_value, group in subset.groupby(by):
        point = _slope(group)
        low, high = stats.bootstrap_statistic([group], lambda g: _slope(g))
        rows.append(
            {
                by: group_value,
                "slope": point,
                "ci_low": low,
                "ci_high": high,
                "n_levels": int(group["tokens_after_target"].nunique()),
                "n": int(len(group)),
            }
        )

    return pd.DataFrame(rows).sort_values("slope").reset_index(drop=True)


def _slope(group: pd.DataFrame) -> float:
    cell = group.groupby("tokens_after_target").agg(
        accuracy=("correct", "mean"), window=("sliding_window", "first")
    ).reset_index()
    if len(cell) < 2:
        return float("nan")
    x = np.log2(cell["tokens_after_target"].to_numpy(float) / cell["window"].to_numpy(float))
    y = stats.logit(cell["accuracy"].to_numpy(float))
    return float(np.polyfit(x, y, 1)[0])


def separation(slopes_table: pd.DataFrame, by: str = "fact_type") -> pd.DataFrame:
    """Which category pairs have non-overlapping slope intervals.

    A coarse screen, not a test: non-overlapping intervals imply a difference, but
    overlapping ones do not imply equality. Enough to see whether H1 has any signal
    before spending A40 time on it.
    """
    table = slopes_table.dropna(subset=["ci_low", "ci_high"])
    rows = []
    for i, a in table.iterrows():
        for _, b in table.loc[i + 1:].iterrows():
            disjoint = a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"]
            rows.append(
                {
                    "a": a[by],
                    "b": b[by],
                    "slope_a": a["slope"],
                    "slope_b": b["slope"],
                    "intervals_disjoint": disjoint,
                }
            )
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per fact type: the H1 table."""
    caveats = {name: entry.get("caveat") for name, entry in config.facts()["types"].items()}

    rows = []
    for fact_type, group in df.groupby("fact_type"):
        exact = group[group["memory_condition"] == "exact_memory"]
        recurrent = group[group["memory_condition"] == "recurrent_memory"]
        acc_exact = metrics.accuracy(exact) if len(exact) else np.nan
        acc_recurrent = metrics.accuracy(recurrent) if len(recurrent) else np.nan
        rows.append(
            {
                "fact_type": fact_type,
                "chance": config.facts()["chance"][fact_type],
                "acc_exact": acc_exact,
                "acc_recurrent": acc_recurrent,
                "retention": acc_recurrent / acc_exact if acc_exact else np.nan,
                "chance_corrected_recurrent": metrics.chance_corrected_accuracy(recurrent)
                if len(recurrent) else np.nan,
                "retrieval_failure_rate": metrics.retrieval_failure_rate(recurrent)
                if len(recurrent) else np.nan,
                "n": int(len(group)),
                "caveat": caveats.get(fact_type),
            }
        )

    return pd.DataFrame(rows).sort_values("retention").reset_index(drop=True)
