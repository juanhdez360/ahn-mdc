"""End-to-end check of the analysis path, with no models and no GPU.

Fabricates trials with a known degradation shape and a known miscalibration, then
runs them through H1, H2, H3 and the gates. Run after touching anything in `src/`:

    uv run python -m ahnexp._smoke
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ahnexp import config, evaluate, h1_degradation, h2_threshold, h3_calibration, report, schema

WINDOW = 256  # AHN trains Qwen2.5-3B with --sliding_window 256
ARM_DECAY = {"gated_deltanet": 0.10, "mamba2": 0.22, "deltanet": 0.30, "transformer": 0.75}
TYPE_DECAY = {"numerical": 1.4, "multi-hop": 1.6, "entity-attribute": 1.0,
              "contradictory": 0.9, "temporal": 0.5}


def fake_results(n_items: int = 20, seeds: tuple[int, ...] = (0, 1, 2, 3)) -> pd.DataFrame:
    """Accuracy decays past the window; confidence decays more slowly (H3)."""
    rng = np.random.default_rng(0)
    levels = evaluate.pressure_levels(WINDOW, mode="full")
    types = list(TYPE_DECAY)

    rows = []
    for arm, arm_decay in ARM_DECAY.items():
        for pressure in levels:
            excess = max(pressure / WINDOW - 1.0, 0.0)
            for seed in seeds:
                for index in range(n_items):
                    fact_type = types[index % len(types)]
                    p_correct = 0.78 * np.exp(-arm_decay * TYPE_DECAY[fact_type] * excess)
                    correct = int(rng.random() < p_correct)
                    # Confidence tracks a shallower decay -> a widening gap.
                    confidence = float(np.clip(
                        0.80 * np.exp(-0.25 * arm_decay * excess) + rng.normal(0, 0.08), 0.01, 0.99
                    ))
                    rows.append({
                        "item_id": f"{fact_type}_{index:03d}",
                        "architecture": arm,
                        "seed": seed,
                        "tokens_after_target": pressure,
                        "sliding_window": WINDOW,
                        "fact_type": fact_type,
                        "distractor_density": "high" if index % 2 else "low",
                        "target_position": "early",
                        "context_tokens": 128 + pressure,
                        "correct": correct,
                        "abstained": 0,
                        "confidence": confidence,
                        "prediction": "x",
                        "gold": "x",
                    })

    return schema.derive_memory_condition(pd.DataFrame(rows))


def main() -> None:
    df = schema.validate(fake_results(), needs=("core", "h1", "h3"))
    t = config.compression_threshold(strict=False)
    print(f"{len(df)} trials · {df['architecture'].nunique()} arms · "
          f"T = {t} tokens = {t / WINDOW:.1f} windows\n")

    print("=== H1: non-uniform degradation ===")
    print(h1_degradation.summary(df).drop(columns=["caveat"]).to_string(index=False))
    slopes = h1_degradation.slopes(df)
    print("\nslopes:")
    print(slopes.to_string(index=False))
    print(f"disjoint pairs: {int(h1_degradation.separation(slopes)['intervals_disjoint'].sum())}")

    print("\n=== H2: threshold-like collapse ===")
    print(h2_threshold.summary(df).to_string(index=False))
    print("\ndrop at T (unverified source, shape check only):")
    print(h2_threshold.drop_at_threshold(df, threshold_tokens=t).to_string(index=False))
    print("\nshape:")
    print(h2_threshold.shape_test(df, threshold_tokens=t).to_string(index=False))

    try:
        config.compression_threshold()
        raise SystemExit("threshold gate did not fire — the citation guard is broken")
    except ValueError:
        print("\nthreshold gate: correctly blocked (no citation locked)")

    print("\n=== H3: miscalibration ===")
    print(h3_calibration.by_condition(df).to_string(index=False))
    print("\nconfidence health:")
    print(h3_calibration.confidence_health(df).to_string(index=False))

    print("\n=== gates ===")
    gates = report.gate_report(df)
    print(gates.to_string(index=False))
    print(f"\nblocking: {len(report.blocking(gates))}")


if __name__ == "__main__":
    main()
