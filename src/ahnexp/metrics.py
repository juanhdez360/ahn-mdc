"""Metrics, with their published definitions.

Every formula here is the one cited in the research doc. Do not "improve" them —
matching the published definition is the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ahnexp import config


def accuracy(df: pd.DataFrame) -> float:
    return float(df["correct"].mean())


def retrieval_failure_rate(df: pd.DataFrame) -> float:
    """Wrong answers, excluding explicit abstentions.

    An abstention is a different failure from a confidently wrong retrieval, and
    lumping them together hides exactly the behaviour H3 is about.
    """
    answered = df[df["abstained"] == 0] if "abstained" in df.columns else df
    return float((1 - answered["correct"]).mean()) if len(answered) else float("nan")


def abstention_rate(df: pd.DataFrame) -> float:
    return float(df["abstained"].mean()) if "abstained" in df.columns else float("nan")


def expected_calibration_error(
    confidence, correct, n_bins: int | None = None
) -> tuple[float, pd.DataFrame]:
    """ECE with equal-width bins.

    Guo, Pleiss, Sun & Weinberger 2017, "On Calibration of Modern Neural Networks",
    ICML (arXiv:1706.04599):

        ECE = sum_m (|B_m| / N) * | acc(B_m) - conf(B_m) |

    Returns the scalar and the per-bin table behind the reliability diagram.
    """
    cfg = config.experiment()["calibration"]
    n_bins = n_bins or int(cfg["ece_bins"])
    if cfg["ece_binning"] != "equal_width":
        raise NotImplementedError(
            f"Only equal-width binning is implemented; config asks for {cfg['ece_binning']!r}."
        )

    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    total = len(confidence)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece, rows = 0.0, []
    for b in range(n_bins):
        low, high = edges[b], edges[b + 1]
        mask = (confidence > low) & (confidence <= high) if b else (confidence >= low) & (confidence <= high)
        if not mask.any():
            continue
        bin_accuracy = correct[mask].mean()
        bin_confidence = confidence[mask].mean()
        weight = mask.sum() / total
        ece += weight * abs(bin_accuracy - bin_confidence)
        rows.append(
            {
                "bin": f"({low:.1f}, {high:.1f}]",
                "n": int(mask.sum()),
                "accuracy": float(bin_accuracy),
                "confidence": float(bin_confidence),
                "gap": float(bin_confidence - bin_accuracy),
            }
        )

    return float(ece), pd.DataFrame(rows)


def brier_score(confidence, correct) -> float:
    """Mean squared error between confidence and the 0/1 outcome.

    Brier 1950, "Verification of Forecasts Expressed in Terms of Probability",
    Monthly Weather Review 78(1): 1-3.

        BS = (1/N) * sum_i (p_i - o_i)^2
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    return float(np.mean((confidence - correct) ** 2))


def confidently_wrong_rate(df: pd.DataFrame, threshold: float | None = None) -> float:
    """Fraction of high-confidence answers that are wrong.

    The most direct evidence for H3: memory failure the model does not signal.
    The threshold is provisional — see `protocol/open_decisions.md` #5.
    """
    threshold = threshold if threshold is not None else float(
        config.experiment()["calibration"]["cwr_threshold"]
    )
    confident = df[df["confidence"] >= threshold]
    return float(1.0 - confident["correct"].mean()) if len(confident) else float("nan")


def confidence_accuracy_gap(df: pd.DataFrame) -> float:
    """Mean confidence minus accuracy. Positive means overconfident."""
    return float(df["confidence"].mean() - df["correct"].mean())


def chance_corrected_accuracy(df: pd.DataFrame) -> float:
    """Accuracy rescaled against the fact type's chance floor.

    Categories differ in how guessable they are — a two-way temporal question is
    right half the time by chance, a numerical ID essentially never. Comparing raw
    accuracy across them would credit temporal facts with robustness they do not have.
    """
    chance = config.facts()["chance"]
    floors = df["fact_type"].map(chance).astype(float)
    corrected = (df["correct"] - floors) / (1.0 - floors)
    return float(corrected.mean())
