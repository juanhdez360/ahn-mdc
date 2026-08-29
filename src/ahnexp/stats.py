"""Resampling utilities shared by H1, H2 and H3.

The unit of resampling is the seed, not the trial. Items are replayed across arms and
pressure levels, so trials are clustered and a naive bootstrap would report intervals
that are far too tight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ahnexp import config

_EPS = 1e-6


def settings() -> dict:
    return config.experiment()["statistics"]


def cluster_bootstrap_ci(
    group: pd.DataFrame,
    value_col: str = "correct",
    cluster_col: str | None = None,
    n_resamples: int | None = None,
    ci: float | None = None,
    random_state: int = 0,
) -> tuple[float, float]:
    """Percentile CI resampling whole clusters with replacement.

    With a single cluster there is nothing to resample across, so this returns NaNs
    rather than a falsely narrow interval.
    """
    cfg = settings()
    cluster_col = cluster_col or cfg["cluster_on"]
    n_resamples = n_resamples or int(cfg["n_resamples"])
    ci = ci or float(cfg["ci"])

    clusters = group[cluster_col].unique()
    if len(clusters) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(random_state)
    by_cluster = [group.loc[group[cluster_col] == c, value_col].to_numpy() for c in clusters]

    means = np.empty(n_resamples)
    for i in range(n_resamples):
        picks = rng.integers(0, len(by_cluster), size=len(by_cluster))
        means[i] = np.concatenate([by_cluster[p] for p in picks]).mean()

    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def bootstrap_statistic(
    frames: list[pd.DataFrame],
    statistic,
    cluster_col: str | None = None,
    n_resamples: int | None = None,
    ci: float | None = None,
    random_state: int = 0,
) -> tuple[float, float]:
    """CI for an arbitrary statistic computed over one or more aligned frames.

    Clusters are resampled once and applied to every frame, which keeps paired
    comparisons paired.
    """
    cfg = settings()
    cluster_col = cluster_col or cfg["cluster_on"]
    n_resamples = n_resamples or int(cfg["n_resamples"])
    ci = ci or float(cfg["ci"])

    clusters = sorted(set.intersection(*(set(f[cluster_col].unique()) for f in frames)))
    if len(clusters) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(random_state)
    indexed = [{c: f[f[cluster_col] == c] for c in clusters} for f in frames]

    values = np.empty(n_resamples)
    for i in range(n_resamples):
        picks = [clusters[p] for p in rng.integers(0, len(clusters), size=len(clusters))]
        resampled = [pd.concat([slot[c] for c in picks]) for slot in indexed]
        values[i] = statistic(*resampled)

    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def paired_difference(
    df: pd.DataFrame,
    arm_a: str,
    arm_b: str,
    value_col: str = "correct",
    condition: str | None = "recurrent_memory",
    random_state: int = 0,
) -> dict:
    """Bootstrap the gap between two arms on shared trials.

    Arms answer the same items, so differencing removes item difficulty and tightens
    the interval considerably.
    """
    cfg = settings()
    subset = df[df["memory_condition"] == condition] if condition else df
    key = ["item_id", "tokens_after_target", "seed"]
    wide = (
        subset[subset["architecture"].isin([arm_a, arm_b])]
        .pivot_table(index=key, columns="architecture", values=value_col)
        .dropna()
    )
    if wide.empty:
        raise ValueError(f"No shared trials between {arm_a} and {arm_b} under {condition!r}.")

    diff = (wide[arm_a] - wide[arm_b]).to_numpy(dtype=float)
    seeds = wide.index.get_level_values("seed").to_numpy()
    clusters = np.unique(seeds)

    if len(clusters) < 2:
        low = high = float("nan")
    else:
        rng = np.random.default_rng(random_state)
        by_cluster = [diff[seeds == c] for c in clusters]
        means = np.empty(int(cfg["n_resamples"]))
        for i in range(len(means)):
            picks = rng.integers(0, len(by_cluster), size=len(by_cluster))
            means[i] = np.concatenate([by_cluster[p] for p in picks]).mean()
        alpha = (1.0 - float(cfg["ci"])) / 2.0
        low, high = float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))

    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "condition": condition,
        "mean_difference": float(diff.mean()),
        "ci_low": low,
        "ci_high": high,
        "n_pairs": int(len(diff)),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni over a family of comparisons."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m, running = len(ordered), 0.0
    adjusted: dict[str, float] = {}
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def logit(p) -> np.ndarray:
    return np.log(np.clip(p, _EPS, 1 - _EPS) / (1 - np.clip(p, _EPS, 1 - _EPS)))


def check_cell_sizes(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Cells too thin to plot, so underpowered points do not reach a figure."""
    by = by or ["architecture", "tokens_after_target"]
    minimum = int(settings()["min_cell_size"])
    counts = df.groupby(by).size().rename("n").reset_index()
    return counts[counts["n"] < minimum]


def assert_matched_design(df: pd.DataFrame) -> None:
    """Every arm must have seen exactly the same trials.

    Silently unbalanced arms are the likeliest way a comparison ends up wrong while
    still looking plausible.
    """
    arms = sorted(df["architecture"].unique())
    if len(arms) < 2:
        return

    cells = df.groupby("architecture").apply(
        lambda g: set(zip(g["item_id"], g["tokens_after_target"], g["seed"])),
        include_groups=False,
    )
    reference = cells[arms[0]]
    problems = []
    for name in arms[1:]:
        if missing := reference - cells[name]:
            problems.append(f"{name} is missing {len(missing)} trials present in {arms[0]}")
        if extra := cells[name] - reference:
            problems.append(f"{name} has {len(extra)} trials absent from {arms[0]}")

    if problems:
        raise AssertionError("Unmatched design:\n  " + "\n  ".join(problems))
