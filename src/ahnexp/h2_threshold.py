"""H2 — threshold-like collapse, and the architecture comparison.

Claim: degradation is nonlinear in compression pressure — stable while the recurrent
state has spare capacity, then a sharp drop once it saturates.

Two tests, both anchored on the published threshold T:

1. **Drop at T.** Accuracy below T versus at or above it. Pre-registered at T, so
   the split point is not chosen from our data.
2. **Shape.** A piecewise fit that is allowed to break at T, against a single smooth
   fit. If the smooth fit does as well, degradation is gradual and H2 is refuted.

Estimating the changepoint from our own curves would be self-defining the threshold,
which the mentor ruled out. T is read from `config`, which raises until it is cited.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ahnexp import config, metrics, models, schema, stats


def curves(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy against pressure, one curve per architecture."""
    schema.validate(df)

    rows = []
    for (arm, pressure), group in df.groupby(["architecture", "tokens_after_target"]):
        low, high = stats.cluster_bootstrap_ci(group)
        window = int(group["sliding_window"].iloc[0])
        rows.append(
            {
                "architecture": arm,
                "tokens_after_target": int(pressure),
                "sliding_window": window,
                "pressure_windows": pressure / window if window else np.nan,
                "memory_condition": group["memory_condition"].iloc[0],
                "accuracy": metrics.accuracy(group),
                "ci_low": low,
                "ci_high": high,
                "n": int(len(group)),
            }
        )

    return pd.DataFrame(rows).sort_values(["architecture", "tokens_after_target"]).reset_index(drop=True)


def drop_at_threshold(df: pd.DataFrame, threshold_tokens: int | None = None) -> pd.DataFrame:
    """Test 1: accuracy either side of T, per architecture."""
    threshold_tokens = (
        config.compression_threshold() if threshold_tokens is None else int(threshold_tokens)
    )

    rows = []
    for arm, group in df.groupby("architecture"):
        below = group[group["tokens_after_target"] < threshold_tokens]
        above = group[group["tokens_after_target"] >= threshold_tokens]
        if below.empty or above.empty:
            raise ValueError(
                f"The pressure grid does not bracket T = {threshold_tokens} tokens for {arm}. "
                "Add grid points on both sides before testing H2."
            )

        low, high = stats.bootstrap_statistic(
            [below, above], lambda b, a: b["correct"].mean() - a["correct"].mean()
        )
        rows.append(
            {
                "architecture": arm,
                "threshold_tokens": threshold_tokens,
                "acc_below_T": metrics.accuracy(below),
                "acc_above_T": metrics.accuracy(above),
                "drop_at_T": metrics.accuracy(below) - metrics.accuracy(above),
                "ci_low": low,
                "ci_high": high,
                "n_below": int(len(below)),
                "n_above": int(len(above)),
            }
        )

    return pd.DataFrame(rows)


def shape_test(df: pd.DataFrame, threshold_tokens: int | None = None) -> pd.DataFrame:
    """Test 2: is the collapse threshold-like or smooth?

    Compares a single log-linear fit against one allowed to change slope at T, by
    residual sum of squares with an AIC penalty for the two extra parameters. The
    break location is fixed at the published T, so nothing is fitted to the data.
    """
    threshold_tokens = (
        config.compression_threshold() if threshold_tokens is None else int(threshold_tokens)
    )

    rows = []
    for arm, group in df.groupby("architecture"):
        cell = group[group["tokens_after_target"] > 0].groupby("tokens_after_target").agg(
            accuracy=("correct", "mean"), window=("sliding_window", "first"), n=("correct", "size")
        ).reset_index()
        if len(cell) < 4:
            rows.append({"architecture": arm, "verdict": "too few pressure levels"})
            continue

        x = np.log2(cell["tokens_after_target"].to_numpy(float) / cell["window"].to_numpy(float))
        y = stats.logit(cell["accuracy"].to_numpy(float))
        break_at = np.log2(threshold_tokens / cell["window"].iloc[0])

        smooth = _rss(x, y, np.polyfit(x, y, 1))
        piecewise = _piecewise_rss(x, y, break_at)

        rows.append(
            {
                "architecture": arm,
                "rss_smooth": smooth,
                "rss_piecewise": piecewise,
                "aic_smooth": _aic(smooth, len(x), 2),
                "aic_piecewise": _aic(piecewise, len(x), 4),
                "verdict": "threshold-like"
                if _aic(piecewise, len(x), 4) < _aic(smooth, len(x), 2)
                else "smooth",
            }
        )

    return pd.DataFrame(rows)


def _rss(x: np.ndarray, y: np.ndarray, coefficients) -> float:
    return float(np.sum((y - np.polyval(coefficients, x)) ** 2))


def _piecewise_rss(x: np.ndarray, y: np.ndarray, break_at: float) -> float:
    left, right = x < break_at, x >= break_at
    if left.sum() < 2 or right.sum() < 2:
        return float("inf")
    return _rss(x[left], y[left], np.polyfit(x[left], y[left], 1)) + _rss(
        x[right], y[right], np.polyfit(x[right], y[right], 1)
    )


def _aic(rss: float, n: int, k: int) -> float:
    if not np.isfinite(rss) or rss <= 0:
        return float("inf")
    return float(n * np.log(rss / n) + 2 * k)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per architecture: the H2 / architecture-comparison table."""
    schema.validate(df)
    stats.assert_matched_design(df)

    slope_by_arm = _slopes(df).set_index("architecture")
    try:
        drops = drop_at_threshold(df).set_index("architecture")
        shapes = shape_test(df).set_index("architecture")
        locked = True
    except ValueError:
        drops = shapes = pd.DataFrame()
        locked = False

    rows = []
    for arm, group in df.groupby("architecture"):
        exact = group[group["memory_condition"] == "exact_memory"]["correct"]
        recurrent = group[group["memory_condition"] == "recurrent_memory"]["correct"]
        acc_exact = float(exact.mean()) if len(exact) else np.nan
        acc_recurrent = float(recurrent.mean()) if len(recurrent) else np.nan

        row = {
            "architecture": arm,
            "label": models.arm(arm).label,
            "ahn_params_m": models.arm(arm).ahn_params / 1e6,
            "acc_exact": acc_exact,
            "acc_recurrent": acc_recurrent,
            "retention": acc_recurrent / acc_exact if acc_exact else np.nan,
            "slope": slope_by_arm.loc[arm, "slope"] if arm in slope_by_arm.index else np.nan,
            "n": int(len(group)),
        }
        if locked and arm in drops.index:
            row |= {
                "acc_below_T": drops.loc[arm, "acc_below_T"],
                "acc_above_T": drops.loc[arm, "acc_above_T"],
                "drop_at_T": drops.loc[arm, "drop_at_T"],
                "shape": shapes.loc[arm, "verdict"] if arm in shapes.index else None,
                "h2_status": "measured",
            }
        else:
            row |= {
                "acc_below_T": np.nan, "acc_above_T": np.nan, "drop_at_T": np.nan,
                "shape": None, "h2_status": "THRESHOLD_NOT_LOCKED",
            }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("retention", ascending=False).reset_index(drop=True)


def _slopes(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[(df["memory_condition"] == "recurrent_memory") & (df["tokens_after_target"] > 0)]
    rows = []
    for arm, group in subset.groupby("architecture"):
        cell = group.groupby("tokens_after_target").agg(
            accuracy=("correct", "mean"), window=("sliding_window", "first")
        ).reset_index()
        if len(cell) < 2:
            rows.append({"architecture": arm, "slope": np.nan})
            continue
        x = np.log2(cell["tokens_after_target"].to_numpy(float) / cell["window"].to_numpy(float))
        rows.append({"architecture": arm, "slope": float(np.polyfit(x, stats.logit(cell["accuracy"]), 1)[0])})
    return pd.DataFrame(rows)


def architecture_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Paired arm-vs-arm gaps under recurrent memory.

    `deltanet` vs `gated_deltanet` is the cleanest pair: one mechanism apart, ~1%
    apart in parameters.
    """
    arms = sorted(set(df["architecture"]) - {"transformer"})
    pairs = [(a, b) for i, a in enumerate(arms) for b in arms[i + 1:]]
    pairs += [(a, "transformer") for a in arms if "transformer" in set(df["architecture"])]
    return pd.DataFrame([stats.paired_difference(df, a, b) for a, b in pairs])
