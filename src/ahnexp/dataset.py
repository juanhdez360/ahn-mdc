"""Synthetic fact generation and matched trajectory construction.

Cleaned up from the pilot (`notebooks/0_pilot_reference.ipynb`) with three fixes the
mentor's feedback requires:

1. Pressure is built to a **token** budget, not a fact count, so the x-axis is in the
   units the AHN module actually compresses.
2. Filler before the target is a separate knob, so target position and total prompt
   length stop being confounded with compression pressure.
3. Distractors are checked for answer collisions — a distractor that leaks the answer
   lets the model score correct without retrieving anything.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ahnexp import config

COLORS = ["blue", "green", "red", "yellow", "purple"]
COMPANIES = ["Google", "Microsoft", "Amazon", "Meta", "Apple"]
CITIES_FROM = ["Paris", "Berlin", "Tokyo", "Delhi", "Sydney"]
CITIES_TO = ["London", "Madrid", "Beijing", "Toronto", "Dubai"]


@dataclass
class Fact:
    fact_type: str
    text: str
    question: str
    answer: str


@dataclass
class Item:
    """One target fact with everything needed to build its trajectories."""

    item_id: str
    fact: Fact
    distractor_density: str
    distractors: list[Fact] = field(default_factory=list)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "fact_type": self.fact.fact_type,
            "distractor_density": self.distractor_density,
            "gold": self.fact.answer,
        }


# ---------------------------------------------------------------------------
# Fact generators — one per category in config/facts.yaml
# ---------------------------------------------------------------------------

def numerical(i: int) -> Fact:
    person, value = f"Person_{i}", str(100000 + i * 137)
    return Fact("numerical", f"{person}'s employee ID is {value}.",
                f"What is {person}'s employee ID?", value)


def temporal(i: int) -> Fact:
    a, b = f"Person_{i}", f"Person_{i + 1}"
    return Fact("temporal", f"{a} arrived before {b}.",
                f"Who arrived first, {a} or {b}?", a)


def entity_attribute(i: int) -> Fact:
    person, color = f"Person_{i}", COLORS[i % len(COLORS)]
    return Fact("entity-attribute", f"{person}'s favorite color is {color}.",
                f"What is {person}'s favorite color?", color)


def multi_hop(i: int) -> Fact:
    a, b, company = f"Person_{i}", f"Person_{i + 1}", COMPANIES[i % len(COMPANIES)]
    return Fact("multi-hop", f"{a} manages {b}. {b} works for {company}.",
                f"Which company does {a}'s subordinate work for?", company)


def contradictory(i: int) -> Fact:
    person = f"Person_{i}"
    old, new = CITIES_FROM[i % 5], CITIES_TO[i % 5]
    return Fact("contradictory", f"{person} lived in {old}. {person} now lives in {new}.",
                f"Where does {person} live now?", new)


GENERATORS: dict[str, Callable[[int], Fact]] = {
    "numerical": numerical,
    "temporal": temporal,
    "entity-attribute": entity_attribute,
    "multi-hop": multi_hop,
    "contradictory": contradictory,
}


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

def generate_items(n_items: int, seed: int = 0, pool_size: int = 4000) -> list[Item]:
    """Balanced item set: equal N per fact type, each with its own distractor pool.

    Generated once and cached, because every arm must see byte-identical prompts.
    """
    rng = random.Random(seed)
    types = list(GENERATORS)
    densities = ["low", "high"]

    items: list[Item] = []
    for index in range(n_items):
        fact_type = types[index % len(types)]
        density = densities[(index // len(types)) % len(densities)]
        target = GENERATORS[fact_type](index)
        item = Item(
            item_id=f"{fact_type}_{index:04d}",
            fact=target,
            distractor_density=density,
            distractors=_distractor_pool(
                fact_type, density, index, rng, pool_size, forbidden=target.answer
            ),
        )
        assert_no_collision(item)
        items.append(item)

    return items


def save_items(
    items: list[Item],
    path: Path | str,
    *,
    seed: int = 0,
    pool_size: int = 4000,
) -> Path:
    """Persist the item catalog (facts + seed). Distractor pools are not stored —
    ``load_items`` regenerates them byte-identically from ``seed`` / ``pool_size``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": int(seed),
        "pool_size": int(pool_size),
        "n_items": len(items),
        "items": [
            {
                "item_id": item.item_id,
                "distractor_density": item.distractor_density,
                "fact": asdict(item.fact),
                "n_distractors": len(item.distractors),
            }
            for item in items
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def load_items(path: Path | str) -> list[Item]:
    """Reload items by regenerating with the saved seed (same prompts as the run)."""
    data = json.loads(Path(path).read_text())
    items = generate_items(
        n_items=int(data["n_items"]),
        seed=int(data["seed"]),
        pool_size=int(data.get("pool_size", 4000)),
    )
    saved_ids = [row["item_id"] for row in data["items"]]
    got_ids = [item.item_id for item in items]
    if saved_ids != got_ids:
        raise ValueError(
            f"Regenerated item_ids differ from {path}. "
            "Was generate_items / facts.yaml changed since the save?"
        )
    return items


def _leaks_answer(text: str, answer: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(answer)}(?!\w)", text, re.IGNORECASE))


def _distractor_pool(
    fact_type: str,
    density: str,
    offset: int,
    rng: random.Random,
    size: int,
    forbidden: str,
) -> list[Fact]:
    """High density concentrates distractors on the target's own type.

    Same-type distractors interfere more, which is the point of the density factor.
    Candidates that mention the target's answer as a whole token are skipped so the
    model cannot score correct from the distractor block.
    """
    types = list(GENERATORS)
    pool: list[Fact] = []
    cursor = 1
    attempts = 0
    while len(pool) < size:
        attempts += 1
        if attempts > size * 40:
            raise ValueError(
                f"Could not fill a {size}-fact pool without leaking {forbidden!r}. "
                "Widen the answer space or lower pool_size."
            )
        dtype = fact_type if density == "high" and rng.random() < 0.8 else rng.choice(types)
        candidate = GENERATORS[dtype](offset + cursor * 7919)
        cursor += 1
        if _leaks_answer(candidate.text, forbidden):
            continue
        pool.append(candidate)
    return pool


def assert_no_collision(item: Item) -> None:
    """A distractor must never contain the target's answer as a whole token.

    Substring matching is too coarse: `Person_1` is a prefix of `Person_15839`, so
    it would flag every later ID as a leak. The scorer uses the same word-boundary
    rule (`evaluate.is_correct`), so the two stay aligned.
    """
    if not config.facts()["collision_control"]["enabled"]:
        return
    for distractor in item.distractors:
        if _leaks_answer(distractor.text, item.fact.answer):
            raise ValueError(
                f"{item.item_id}: distractor leaks the answer {item.fact.answer!r} "
                f"in {distractor.text!r}"
            )


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

_PROMPT = """You are given a set of factual statements.

{context}

Question:
{question}

Answer with only the short answer.
"""


def build_trajectory(
    item: Item,
    tokenizer,
    tokens_after_target: int,
    seed: int,
    tokens_before_target: int = 0,
) -> dict[str, Any]:
    """One matched trajectory: fixed target and query, varying pressure.

    `tokens_before_target` is filler that moves the target later in the context
    without adding compression pressure. Holding `before + after` constant while
    sweeping `after` gives the length-matched control; leaving it at zero
    reproduces the pilot's target-first layout.
    """
    rng = random.Random(seed)
    order = list(item.distractors)
    rng.shuffle(order)

    before, used = _fill(order, tokenizer, tokens_before_target, start=0)
    after, _ = _fill(order, tokenizer, tokens_after_target, start=used)

    context = "\n".join(f"- {f.text}" for f in [*before, item.fact, *after])
    prompt = _PROMPT.format(context=context, question=item.fact.question)

    realised = len(tokenizer(_block(after))["input_ids"]) if after else 0
    return {
        **item.as_metadata(),
        "seed": seed,
        "prompt": prompt,
        "tokens_after_target": realised,
        "requested_tokens_after_target": tokens_after_target,
        "context_tokens": len(tokenizer(prompt)["input_ids"]),
        "target_position": _position(len(before), len(after)),
    }


def _fill(pool: list[Fact], tokenizer, budget: int, start: int) -> tuple[list[Fact], int]:
    """Take facts from the pool until the token budget is met."""
    if budget <= 0:
        return [], start

    taken, total, cursor = [], 0, start
    while total < budget and cursor < len(pool):
        fact = pool[cursor]
        cursor += 1
        taken.append(fact)
        total = len(tokenizer(_block(taken))["input_ids"])

    if total < budget:
        raise ValueError(
            f"Distractor pool exhausted at {total} tokens, short of the {budget} requested. "
            "Increase pool_size in generate_items."
        )
    return taken, cursor


def _block(facts: list[Fact]) -> str:
    return "\n".join(f"- {f.text}" for f in facts)


def _position(n_before: int, n_after: int) -> str:
    total = n_before + n_after
    if total == 0:
        return "early"
    fraction = n_before / total
    return "early" if fraction < 0.25 else "mid" if fraction < 0.75 else "late"
