"""AHN memory-degradation and calibration experiment.

Layout mirrors the pipeline: `config` and `schema` are the contracts, `dataset`,
`models` and `evaluate` produce trials, `stats` and `metrics` are shared machinery,
and `h1_degradation` / `h2_threshold` / `h3_calibration` implement one hypothesis
each. `report` turns any of them into gated tables and figures.
"""

from ahnexp import (
    config,
    dataset,
    evaluate,
    h1_degradation,
    h2_threshold,
    h3_calibration,
    metrics,
    models,
    report,
    schema,
    stats,
)

__all__ = [
    "config", "schema", "dataset", "models", "evaluate",
    "stats", "metrics", "report",
    "h1_degradation", "h2_threshold", "h3_calibration",
]
__version__ = "0.2.0"
