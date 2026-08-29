"""H3 — miscalibration under compression.

Claim: confidence does not fully track declining accuracy. It falls more slowly,
widening the confidence–accuracy gap, and in the strongest case produces confidently
incorrect answers on facts the compressed memory has lost.

The headline evidence is the gap growing with pressure, not the overall ECE. A model
can have a small aggregate ECE and still be badly overconfident exactly where memory
has failed, because the well-calibrated in-window trials dominate the average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ahnexp import config, metrics, schema, stats


def by_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy, confidence and the gap at each compression level.

    This table is H3. If the `gap` column grows from left to right, confidence is
    failing to track memory.
    """
    schema.validate(df, needs=("core", "h3"))

    rows = []
    for pressure, group in df.groupby("tokens_after_target"):
        window = int(group["sliding_window"].iloc[0])
        ece, _ = metrics.expected_calibration_error(group["confidence"], group["correct"])
        low, high = stats.bootstrap_statistic(
            [group], lambda g: g["confidence"].mean() - g["correct"].mean()
        )
        rows.append(
            {
                "tokens_after_target": int(pressure),
                "pressure_windows": pressure / window if window else np.nan,
                "memory_condition": group["memory_condition"].iloc[0],
                "accuracy": metrics.accuracy(group),
                "confidence": float(group["confidence"].mean()),
                "gap": metrics.confidence_accuracy_gap(group),
                "gap_ci_low": low,
                "gap_ci_high": high,
                "ece": ece,
                "brier": metrics.brier_score(group["confidence"], group["correct"]),
                "cwr": metrics.confidently_wrong_rate(group),
                "n": int(len(group)),
            }
        )

    return pd.DataFrame(rows).sort_values("tokens_after_target").reset_index(drop=True)


def reliability(df: pd.DataFrame, condition: str | None = None) -> pd.DataFrame:
    """Per-bin table behind the reliability diagram (Guo et al. 2017)."""
    subset = df[df["memory_condition"] == condition] if condition else df
    _, table = metrics.expected_calibration_error(subset["confidence"], subset["correct"])
    return table


def by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Exact versus recurrent memory — the cleanest statement of H3.

    Calibration under exact memory is the control: whatever miscalibration exists
    there is a property of the model, not of compression. The difference is the effect.
    """
    rows = []
    for condition, group in df.groupby("memory_condition"):
        ece, _ = metrics.expected_calibration_error(group["confidence"], group["correct"])
        rows.append(
            {
                "memory_condition": condition,
                "accuracy": metrics.accuracy(group),
                "confidence": float(group["confidence"].mean()),
                "gap": metrics.confidence_accuracy_gap(group),
                "ece": ece,
                "brier": metrics.brier_score(group["confidence"], group["correct"]),
                "cwr": metrics.confidently_wrong_rate(group),
                "abstention_rate": metrics.abstention_rate(group),
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("memory_condition").reset_index(drop=True)


def by_fact_type(df: pd.DataFrame, recurrent_only: bool = True) -> pd.DataFrame:
    """Where H1 and H3 meet: is the most-degraded fact type also the least calibrated?"""
    subset = df[df["memory_condition"] == "recurrent_memory"] if recurrent_only else df

    rows = []
    for fact_type, group in subset.groupby("fact_type"):
        ece, _ = metrics.expected_calibration_error(group["confidence"], group["correct"])
        rows.append(
            {
                "fact_type": fact_type,
                "accuracy": metrics.accuracy(group),
                "confidence": float(group["confidence"].mean()),
                "gap": metrics.confidence_accuracy_gap(group),
                "ece": ece,
                "brier": metrics.brier_score(group["confidence"], group["correct"]),
                "cwr": metrics.confidently_wrong_rate(group),
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("gap", ascending=False).reset_index(drop=True)


def confidence_health(df: pd.DataFrame) -> pd.DataFrame:
    """Sanity checks on the confidence signal itself, before interpreting calibration.

    Sequence probability shrinks with answer length, so it can span orders of
    magnitude and pile every trial into the lowest bin. That produces a large ECE
    that says more about tokenisation than about the model's self-knowledge.
    See `protocol/open_decisions.md` #6.
    """
    confidence = df["confidence"]
    n_bins = int(config.experiment()["calibration"]["ece_bins"])
    occupied = pd.cut(confidence, np.linspace(0, 1, n_bins + 1), include_lowest=True).nunique()
    lowest = float((confidence <= 1.0 / n_bins).mean())

    return pd.DataFrame(
        [
            {"check": "mode", "value": config.experiment()["calibration"]["confidence"]},
            {"check": "min", "value": float(confidence.min())},
            {"check": "max", "value": float(confidence.max())},
            {"check": "median", "value": float(confidence.median())},
            {"check": "bins occupied", "value": f"{occupied}/{n_bins}"},
            {
                "check": "fraction in lowest bin",
                "value": f"{lowest:.1%}"
                + ("  <-- signal is degenerate, ECE is not interpretable" if lowest > 0.5 else ""),
            },
        ]
    )
