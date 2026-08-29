"""Generation, scoring and confidence — one trial at a time, plus the grid runner."""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ahnexp import config, dataset, models, schema

_ABSTENTIONS = (
    "i don't know", "i do not know", "unknown", "not sure", "cannot determine",
    "no information", "not mentioned", "unclear",
)


def normalise(text: str) -> str:
    """Trim the model's answer down to something comparable with the gold string."""
    text = text.strip().split("\n")[0].replace("`", " ")
    return text.strip().strip(".").strip('"').strip("'").strip().lower()


def is_correct(prediction: str, gold: str) -> int:
    """Word-boundary containment, so "the answer is 4827." matches "4827"."""
    p, g = normalise(prediction), normalise(gold)
    if not g:
        return 0
    if p == g:
        return 1
    return 1 if re.search(rf"(?<!\w){re.escape(g)}(?!\w)", p) else 0


def is_abstention(prediction: str) -> int:
    """Declining to answer is not the same failure as retrieving the wrong thing."""
    p = normalise(prediction)
    return 1 if any(phrase in p for phrase in _ABSTENTIONS) else 0


def run_trial(model, tokenizer, trajectory: dict[str, Any]) -> dict[str, Any]:
    """Generate and score one trajectory from a fresh state.

    Each call is an independent forward pass on its own prompt, which is what
    satisfies the state-reset commitment: no probe can influence a later one.
    """
    import numpy as np
    import torch

    generation = config.experiment()["models"]["matched"]["generation"]
    device = getattr(model, "device", None) or next(model.parameters()).device
    inputs = tokenizer(trajectory["prompt"], return_tensors="pt", truncation=False).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=generation["max_new_tokens"],
            do_sample=generation["do_sample"],
            num_beams=generation["num_beams"],
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output.sequences[0, prompt_len:]
    prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

    scores = model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
    log_probs = scores[0][: len(generated)]
    confidence = _confidence(log_probs, np)

    return {
        "prediction": prediction,
        "gold": trajectory["gold"],
        "correct": is_correct(prediction, trajectory["gold"]),
        "abstained": is_abstention(prediction),
        "confidence": confidence,
    }


def _confidence(log_probs, np) -> float:
    """Sequence probability, or its length-normalised form.

    Sequence probability penalises long answers — the pilot's values span six orders
    of magnitude, which distorts every calibration bin. Which one we use is still an
    open decision (open_decisions.md #6), so both live behind the config switch.
    """
    mode = config.experiment()["calibration"]["confidence"]
    total = float(log_probs.sum())
    if mode == "sequence_probability":
        return float(np.exp(total))
    if mode == "length_normalised":
        return float(np.exp(total / max(len(log_probs), 1)))
    raise ValueError(f"Unknown confidence mode {mode!r}")


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def pressure_levels(sliding_window: int, mode: str = "pilot") -> list[int]:
    """Grid in tokens, from window multiples, with T inserted as its own point."""
    pressure = config.experiment()["pressure"]
    multiples = pressure[config.run_mode(mode)["grid"]]
    cap = int(pressure["max_context_tokens"])

    levels = {min(int(round(m * sliding_window)), cap) for m in multiples}
    if pressure["include_threshold"]:
        levels.add(min(config.compression_threshold(strict=False), cap))
    return sorted(levels)


def run_grid(
    items: Iterable[dataset.Item],
    tokenizer_for_items,
    ahn_repo: str | Path | None = None,
    mode: str = "pilot",
    arms: list[str] | None = None,
    length_matched: bool = False,
) -> pd.DataFrame:
    """Replay the same items across every arm, at every pressure level.

    Arms load one at a time and are released before the next: three merged 3B
    checkpoints do not co-exist on a typical single GPU (Colab or laptop).
    """
    items = list(items)
    arms = arms or models.list_arms()
    seeds = config.run_mode(mode)["seeds"]
    # None → config.ahn_repo() (AHN_REPO / Colab / vendor/AHN)
    resolved_repo = config.ahn_repo(ahn_repo) if ahn_repo is not None else None

    records: list[dict[str, Any]] = []
    descriptions: list[dict[str, Any]] = []

    for name in arms:
        model, tokenizer = models.load(name, ahn_repo=resolved_repo)
        description = models.describe(name, model, tokenizer)
        descriptions.append(description)
        window = description["sliding_window"]
        levels = pressure_levels(window, mode)
        budget = max(levels) if length_matched else 0

        for pressure in levels:
            for seed in seeds:
                for item in items:
                    trajectory = dataset.build_trajectory(
                        item,
                        tokenizer,
                        tokens_after_target=pressure,
                        seed=seed,
                        # Length matching trades filler for pressure so the total
                        # prompt stays constant across the sweep.
                        tokens_before_target=max(budget - pressure, 0),
                    )
                    records.append(
                        {
                            **{k: trajectory[k] for k in
                               ("item_id", "fact_type", "distractor_density", "target_position",
                                "tokens_after_target", "context_tokens", "seed")},
                            "architecture": name,
                            "sliding_window": window,
                            **run_trial(model, tokenizer, trajectory),
                        }
                    )

        del model, tokenizer
        gc.collect()
        _empty_cuda_cache()

    models.assert_matched(descriptions)
    df = schema.derive_memory_condition(pd.DataFrame(records))
    return schema.validate(df, needs=("core", "h1", "h3"))


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
