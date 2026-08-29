"""Per-trial results contract.

The seam between the run pipeline and the three hypothesis analyses. One row = one
(item, architecture, pressure level, seed) trial. H1, H2 and H3 all read this, so
nobody may quietly rename a column.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    description: str


IDENTITY = (
    Column("item_id", "string", "Stable id of the fact/query pair, identical across arms"),
    Column("architecture", "string", "mamba2 | deltanet | gated_deltanet | transformer"),
    Column("seed", "int64", "Seed controlling distractor sampling and ordering"),
)

DESIGN = (
    Column("tokens_after_target", "int64", "Compression pressure, in model tokens"),
    Column("sliding_window", "int64", "Window length in force, for normalisation"),
    Column("memory_condition", "string", "exact_memory | recurrent_memory"),
    Column("fact_type", "string", "Category from config/facts.yaml"),
    Column("distractor_density", "string", "low | high"),
    Column("target_position", "string", "early | mid | late"),
    Column("context_tokens", "int64", "Total prompt length, for length-matching checks"),
)

OUTCOME = (
    Column("correct", "int64", "1 if the normalised prediction matches the gold answer"),
    Column("abstained", "int64", "1 if the model declined to answer — not the same as wrong"),
    Column("confidence", "float64", "Confidence of the generated answer, in [0, 1]"),
    Column("prediction", "string", "Raw generated text, kept for failure analysis"),
    Column("gold", "string", "Gold answer"),
)

COLUMNS = IDENTITY + DESIGN + OUTCOME

# The minimum each hypothesis needs. Being explicit means an analysis keeps working
# when optional columns are added upstream.
REQUIRED = {
    "core": ("item_id", "architecture", "seed", "tokens_after_target", "sliding_window",
             "memory_condition", "correct"),
    "h1": ("fact_type",),
    "h2": (),
    "h3": ("confidence",),
}


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c.name: pd.Series(dtype=c.dtype) for c in COLUMNS})


def validate(df: pd.DataFrame, *, needs: tuple[str, ...] = ("core",)) -> pd.DataFrame:
    """Fail loudly on a malformed results frame, before it reaches a figure."""
    required: list[str] = []
    for group in needs:
        required.extend(REQUIRED[group])

    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Results frame is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Results frame is empty.")

    bad = set(df["correct"].dropna().unique()) - {0, 1}
    if bad:
        raise ValueError(f"`correct` must be 0/1; found {sorted(bad)}")

    if "confidence" in df.columns:
        conf = df["confidence"].dropna()
        if not conf.between(0.0, 1.0).all():
            raise ValueError("`confidence` must lie in [0, 1].")

    duplicated = df.duplicated(subset=["item_id", "architecture", "seed", "tokens_after_target"])
    if duplicated.any():
        raise ValueError(f"{int(duplicated.sum())} duplicated trials in the results frame.")

    return df


def derive_memory_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute `memory_condition` from the mechanical rule.

    The condition is a function of the window in force, so it is derived rather
    than trusted from upstream.
    """
    out = df.copy()
    out["memory_condition"] = (out["tokens_after_target"] >= out["sliding_window"]).map(
        {True: "recurrent_memory", False: "exact_memory"}
    )
    return out


def describe() -> pd.DataFrame:
    """Human-readable schema, for the appendix and for onboarding."""
    return pd.DataFrame(
        [(c.name, c.dtype, c.description) for c in COLUMNS],
        columns=["column", "dtype", "description"],
    )


# ---------------------------------------------------------------------------
# Pilot adapter
# ---------------------------------------------------------------------------

_PILOT_RENAME = {"fact_id": "item_id", "ground_truth": "gold", "token_count": "context_tokens"}


def from_pilot_csv(path, architecture: str, sliding_window: int) -> pd.DataFrame:
    """Adapt `data/pilot_raw_results.csv` to the schema, for one arm.

    The pilot measured pressure in facts. `tokens_after_target` is recovered by
    differencing each item's context length against its own zero-pressure probe,
    then collapsed to the level median — the design variable is the nominal level,
    and per-item length jitter is realisation noise.

    The pilot predates the window fix (open_decisions.md #1), so `sliding_window`
    must be passed in explicitly and the resulting memory conditions are only as
    trustworthy as that number.
    """
    df = pd.read_csv(path).rename(columns=_PILOT_RENAME)

    baseline = df[df["facts_after_target"] == 0].set_index("item_id")["context_tokens"]
    missing = set(df["item_id"]) - set(baseline.index)
    if missing:
        raise ValueError(f"{len(missing)} items have no zero-pressure probe to difference against.")

    realised = df["context_tokens"] - df["item_id"].map(baseline)
    df["tokens_after_target"] = (
        realised.groupby(df["facts_after_target"]).transform("median").round().astype(int)
    )
    df["architecture"] = architecture
    df["sliding_window"] = int(sliding_window)
    df["target_position"] = "early"      # the pilot always placed the target first
    if "abstained" not in df.columns:
        # The pilot did not separate "declined to answer" from "answered wrong".
        df["abstained"] = 0

    return validate(derive_memory_condition(df), needs=("core", "h1", "h3"))
