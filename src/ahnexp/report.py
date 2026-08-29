"""Acceptance gates, figures and Markdown tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ahnexp import config, models, stats


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def gate_report(df: pd.DataFrame) -> pd.DataFrame:
    """Everything that must pass before a number leaves the repo."""
    gates = config.experiment()["acceptance"]
    band = gates["exact_memory_accuracy"]
    checks: list[dict[str, Any]] = []

    for arm, group in df.groupby("architecture"):
        exact = group[group["memory_condition"] == "exact_memory"]["correct"]
        checks.append({"gate": "exact_memory_accuracy", "scope": arm, **_band_verdict(exact, band)})

    window = int(df["sliding_window"].iloc[0])
    checks.append({"gate": "window_is_exceeded", "scope": "design", **_window_verdict(df, window)})

    try:
        tokens = config.compression_threshold()
        checks.append({"gate": "threshold_locked", "scope": "H2", "verdict": "PASS",
                       "detail": f"T={tokens} tokens, from a published source"})
    except ValueError as error:
        checks.append({"gate": "threshold_locked", "scope": "H2", "verdict": "BLOCKED",
                       "detail": str(error).split(".")[0]})

    try:
        config.assert_threshold_clears_window(window)
        tokens = config.compression_threshold(strict=False)
        checks.append({"gate": "threshold_clears_window", "scope": "H2", "verdict": "PASS",
                       "detail": f"T={tokens} tokens = {tokens / window:.1f} windows of {window}"})
    except ValueError as error:
        checks.append({"gate": "threshold_clears_window", "scope": "H2", "verdict": "FAIL",
                       "detail": str(error).split(".")[0]})

    thin = stats.check_cell_sizes(df)
    checks.append({"gate": "min_cell_size", "scope": "all",
                   "verdict": "PASS" if thin.empty else "FAIL",
                   "detail": "all cells large enough" if thin.empty
                   else f"{len(thin)} underpowered cells"})

    try:
        stats.assert_matched_design(df)
        checks.append({"gate": "matched_design", "scope": "all", "verdict": "PASS",
                       "detail": f"{df['architecture'].nunique()} arm(s), balanced"})
    except AssertionError as error:
        checks.append({"gate": "matched_design", "scope": "all", "verdict": "FAIL",
                       "detail": str(error)})

    return pd.DataFrame(checks)


def _band_verdict(exact: pd.Series, band: dict) -> dict[str, str]:
    if not len(exact):
        return {"verdict": "FAIL", "detail": "no exact-memory trials"}
    accuracy = float(exact.mean())
    if accuracy >= band["red_flag_at"]:
        return {"verdict": "RED_FLAG",
                "detail": f"{accuracy:.1%} >= {band['red_flag_at']:.0%}: task already solved, "
                          "harden the fact/distractor design"}
    if accuracy > band["acceptable_max"]:
        return {"verdict": "WARN",
                "detail": f"{accuracy:.1%} above the {band['acceptable_max']:.0%} upper bound"}
    if band["defensible"][0] <= accuracy <= band["defensible"][1]:
        return {"verdict": "PASS", "detail": f"{accuracy:.1%} inside the defensible band"}
    if accuracy > band["defensible"][1]:
        return {"verdict": "WARN",
                "detail": f"{accuracy:.1%} above the band, within the "
                          f"{band['acceptable_max']:.0%} bound"}
    return {"verdict": "WARN", "detail": f"{accuracy:.1%} below the {band['defensible'][0]:.0%} floor"}


def _window_verdict(df: pd.DataFrame, window: int) -> dict[str, str]:
    """The P0 check: did anything ever get compressed?

    If no trial exceeded the sliding window, every answer came from exact attention
    and the run measured prompt length, not memory.
    """
    recurrent = int((df["memory_condition"] == "recurrent_memory").sum())
    if recurrent == 0:
        return {"verdict": "FAIL",
                "detail": f"no trial exceeded the {window}-token window — nothing was "
                          "compressed, so these numbers are not about memory"}
    return {"verdict": "PASS",
            "detail": f"{recurrent}/{len(df)} trials past the {window}-token window "
                      f"(max {int(df['tokens_after_target'].max())})"}


def blocking(gates: pd.DataFrame) -> pd.DataFrame:
    return gates[gates["verdict"].isin(["FAIL", "RED_FLAG", "BLOCKED"])]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_curves(curve: pd.DataFrame, group_col: str, path: Path | str, ylabel: str = "Retrieval accuracy",
                value_col: str = "accuracy", show_threshold: bool = True):
    """Accuracy against compression pressure, one line per group."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, group in curve.groupby(group_col):
        group = group.sort_values("pressure_windows")
        label = models.arm(name).label if group_col == "architecture" else name
        ax.plot(group["pressure_windows"], group[value_col], marker="o", label=label)
        if {"ci_low", "ci_high"} <= set(group.columns) and group["ci_low"].notna().all():
            ax.fill_between(group["pressure_windows"], group["ci_low"], group["ci_high"], alpha=0.15)

    ax.axvline(1.0, linestyle=":", linewidth=1, color="black")
    ax.text(1.0, 1.02, " window edge", fontsize=8, color="black", ha="right")

    if show_threshold:
        window = float(curve["sliding_window"].iloc[0]) if "sliding_window" in curve else None
        if window:
            at = config.compression_threshold(strict=False) / window
            ax.axvline(at, linestyle="--", linewidth=1.2, color="crimson")
            ax.text(at, 1.02, "  T (published)", fontsize=8, color="crimson")

    ax.set_xscale("symlog", linthresh=0.25)
    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Compression pressure (tokens after target / sliding window)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_confidence_gap(table: pd.DataFrame, path: Path | str):
    """H3's figure: accuracy and confidence on the same axes, gap shaded."""
    import matplotlib.pyplot as plt

    table = table.sort_values("pressure_windows")
    x = table["pressure_windows"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, table["accuracy"], marker="o", label="Accuracy", color="tab:blue")
    ax.plot(x, table["confidence"], marker="s", label="Confidence", color="tab:orange")
    ax.fill_between(x, table["accuracy"], table["confidence"],
                    where=table["confidence"] >= table["accuracy"],
                    alpha=0.2, color="tab:red", label="Overconfidence")

    ax.axvline(1.0, linestyle=":", linewidth=1, color="black")
    ax.text(1.0, 1.02, " window edge", fontsize=8, color="black", ha="right")
    ax.set_xscale("symlog", linthresh=0.25)
    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Compression pressure (tokens after target / sliding window)")
    ax.set_ylabel("Accuracy / confidence")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_reliability(bins: pd.DataFrame, path: Path | str, title: str = "Reliability"):
    """Reliability diagram (Guo et al. 2017)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    if len(bins):
        ax.plot(bins["confidence"], bins["accuracy"], marker="o", label="observed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    return fig


def _save(fig, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

_PERCENT = {"accuracy", "acc_exact", "acc_recurrent", "acc_below_T", "acc_above_T", "drop_at_T",
            "retention", "retrieval_failure_rate", "abstention_rate", "confidence", "gap", "cwr",
            "chance_corrected", "chance_corrected_recurrent"}


def to_markdown(table: pd.DataFrame, title: str, mode: str = "pilot",
                notes: list[str] | None = None, synthetic: bool = False) -> str:
    """Paper-shaped Markdown with a provenance banner.

    The banner is not decoration: these files are regenerated by notebooks, and a
    stale pilot table that looks like a result is exactly what ends up in a draft.
    """
    lines = [f"# {title}", "", "Generated by the notebooks — do not edit by hand.", ""]
    if synthetic:
        lines += ["> **SYNTHETIC DATA.** Shape check only. These are not measurements.", ""]
    elif mode == "pilot":
        lines += ["> **PILOT RUN.** Pipeline verification. Not citable as a result.", ""]

    columns = [c for c in table.columns if c != "caveat"]
    lines += ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_cell(row[c], c) for c in columns) + " |")

    if notes:
        lines += [""] + [f"{note}  " for note in notes]
    return "\n".join(lines) + "\n"


def _cell(value, column: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.1%}" if column in _PERCENT else f"{value:.3f}"
    return str(value)


def write_table(table: pd.DataFrame, name: str, title: str, mode: str = "pilot",
                notes: list[str] | None = None, synthetic: bool = False) -> Path:
    path = config.table_path(name, mode)
    path.write_text(to_markdown(table, title, mode, notes, synthetic))
    return path
