"""Repo location and the frozen YAML configuration.

`config/experiment.yaml` is the single source of truth. Nothing in `src/` hardcodes
an experimental number.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_ROOT_MARKERS = ("pyproject.toml", "config")
_MERGE_SCRIPT = Path("examples/scripts/utils/merge_weights.py")


def project_root(start: Path | None = None) -> Path:
    """Walk up until the repo root is found.

    Notebooks live in a subdirectory and also run on Colab, so no call site can rely
    on the working directory.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if all((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    raise FileNotFoundError(
        f"Could not locate the repo root above {here}. "
        "Expected a directory containing pyproject.toml and config/."
    )


def bootstrap(start: Path | None = None) -> Path:
    """Put `src/` on sys.path so notebooks work without `pip install -e .`."""
    root = project_root(start)
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def ahn_repo(explicit: Path | str | None = None) -> Path:
    """Resolve the ByteDance-Seed/AHN checkout used for weight merging.

    Order: explicit path → ``AHN_REPO`` env → Colab ``/content/AHN`` →
    ``vendor/AHN`` under this project.
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    if env := os.environ.get("AHN_REPO"):
        candidates.append(Path(env).expanduser())
    if is_colab():
        candidates.append(Path("/content/AHN"))
    candidates.append(project_root() / "vendor" / "AHN")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / _MERGE_SCRIPT).is_file():
            return resolved

    searched = ", ".join(str(p) for p in seen) or "(none)"
    raise FileNotFoundError(
        "AHN repo not found (need examples/scripts/utils/merge_weights.py). "
        f"Tried: {searched}. "
        "Local: git clone https://github.com/ByteDance-Seed/AHN.git vendor/AHN. "
        "Colab: clone to /content/AHN. Or set AHN_REPO."
    )


@lru_cache(maxsize=None)
def _load(relative_path: str) -> dict[str, Any]:
    path = project_root() / relative_path
    with path.open() as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        raise ValueError(f"{relative_path} is empty.")
    return loaded


def clear_caches() -> None:
    """Drop cached YAML (call after editing config/*.yaml or on notebook reload)."""
    _load.cache_clear()


def experiment() -> dict[str, Any]:
    return _load("config/experiment.yaml")


def facts() -> dict[str, Any]:
    return _load("config/facts.yaml")


def run_mode(name: str = "pilot") -> dict[str, Any]:
    return experiment()["run_modes"][name]


def output_path(key: str, mode: str = "pilot") -> Path:
    """Resolve `outputs.raw`, substituting the run-mode suffix."""
    template = experiment()["outputs"][key]
    path = project_root() / template.format(suffix=run_mode(mode)["suffix"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def figure_path(name: str, mode: str = "pilot") -> Path:
    root = project_root() / experiment()["outputs"]["figures"]
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{name}{run_mode(mode)['suffix']}.png"


def table_path(name: str, mode: str = "pilot") -> Path:
    root = project_root() / experiment()["outputs"]["tables"]
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{name}{run_mode(mode)['suffix']}.md"


# ---------------------------------------------------------------------------
# The H2 threshold gate
# ---------------------------------------------------------------------------

def tokens_per_fact() -> float:
    """Conversion between the fact-denominated threshold and the token axis."""
    conversion = experiment()["threshold"]["tokens_per_fact"]
    return float(conversion["measured"] or conversion["estimate"])


def compression_threshold(strict: bool = True) -> int:
    """The H2 threshold T, in model tokens.

    Raises while the citation is unresolved. This is the mentor's constraint
    enforced in code: no self-defined threshold reaches a table. `strict=False` is
    for resolving the pressure grid, which needs T's location before the source is
    recorded.
    """
    threshold = experiment()["threshold"]
    if strict and threshold.get("status") != "LOCKED":
        raise ValueError(
            "H2 threshold T is not locked. Complete the decision record in "
            f"{threshold['decision_record']}, then set status: LOCKED in "
            "config/experiment.yaml. Do not invent a value."
        )

    tokens = threshold.get("tokens")
    if tokens is None:
        facts_value = threshold.get("facts")
        if facts_value is None:
            raise ValueError("Threshold has neither `tokens` nor `facts`.")
        tokens = round(float(facts_value) * tokens_per_fact())
    return int(tokens)


def assert_threshold_clears_window(sliding_window: int) -> None:
    """T must sit outside the lossless window, or there is no compression to test.

    Below one window the target is still in the exact KV cache and every arm,
    including the no-AHN baseline, sits at ceiling.
    """
    minimum = float(experiment()["threshold"]["min_windows"])
    tokens = compression_threshold(strict=False)
    windows = tokens / sliding_window
    if windows < minimum:
        raise ValueError(
            f"T = {tokens} tokens is only {windows:.2f} sliding windows "
            f"({sliding_window} tokens each), below the required {minimum}. The target "
            "would still be in the lossless KV cache, so the experiment would measure "
            "nothing. Shorten the window or re-derive T."
        )
