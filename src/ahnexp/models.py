"""Loading the AHN arms under matched conditions.

Torch and transformers are imported lazily so the analysis half of the pipeline runs
on a laptop with no GPU stack installed.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ahnexp import config

_MERGE_SCRIPT = "examples/scripts/utils/merge_weights.py"


@dataclass(frozen=True)
class Arm:
    name: str
    label: str
    base_repo: str
    ahn_implementation: str | None
    checkpoint: str | None
    ahn_params: int
    extra_install: str | None

    @property
    def needs_merge(self) -> bool:
        return self.checkpoint is not None

    @property
    def merged_dir(self) -> str:
        return f"merged_ckpt/{self.name}"


def list_arms() -> list[str]:
    return list(config.experiment()["models"]["arms"])


def arm(name: str) -> Arm:
    models = config.experiment()["models"]
    if name not in models["arms"]:
        raise KeyError(f"Unknown arm {name!r}. Known: {list_arms()}")
    entry = models["arms"][name]
    return Arm(
        name=name,
        label=entry["label"],
        base_repo=models["base"],
        ahn_implementation=entry["ahn_implementation"],
        checkpoint=entry["checkpoint"],
        ahn_params=int(entry["ahn_params"]),
        extra_install=entry["extra_install"],
    )


def merge_command(name: str, ahn_repo: str | Path) -> list[str]:
    """The AHN weight-merge invocation. See the AHN README (Model Zoo / Inference)."""
    import sys

    spec = arm(name)
    if not spec.needs_merge:
        raise ValueError(f"{name} has no AHN weights; load the base model directly.")
    return [
        sys.executable,
        str(Path(ahn_repo) / "examples/scripts/utils/merge_weights.py"),
        "--base-model", spec.base_repo,
        "--ahn-path", spec.checkpoint,
        "--output-path", str(config.project_root() / spec.merged_dir),
    ]


def _purge_transformers_modules() -> None:
    """Drop in-memory transformers so a disk upgrade is visible in this process."""
    import sys

    doomed = [name for name in sys.modules if name == "transformers" or name.startswith("transformers.")]
    for name in doomed:
        del sys.modules[name]


def _ensure_transformers_for_ahn() -> None:
    """ByteDance AHN needs transformers>=4.51 (``dynamic_rope_update``).

    LLaMA-Factory often reinstalls 4.49. If this process already imported the old
    build, upgrading on disk is not enough — we purge ``sys.modules`` after.
    """
    import importlib.metadata
    import shutil
    import sys

    from packaging.version import Version

    need = Version("4.51.0")
    current = Version(importlib.metadata.version("transformers"))
    if current < need:
        uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
        print(f"transformers {current} < 4.51.0 — upgrading for AHN…")
        if Path(uv).exists():
            subprocess.run(
                [uv, "pip", "install", "--python", sys.executable, "transformers==4.51.0"],
                check=True,
            )
        else:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "transformers==4.51.0"],
                check=True,
            )
        current = Version(importlib.metadata.version("transformers"))
        if current < need:
            raise RuntimeError(
                f"Need transformers>=4.51.0 (have {current}). "
                "Run: uv pip install --python .venv/bin/python 'transformers==4.51.0'"
            )
        _purge_transformers_modules()
        return

    # Right version on disk, but this kernel may still hold an older import.
    try:
        from transformers.modeling_rope_utils import dynamic_rope_update  # noqa: F401
    except ImportError:
        print("transformers import stale — reloading after purge…")
        _purge_transformers_modules()
        from transformers.modeling_rope_utils import dynamic_rope_update  # noqa: F401


def ensure_merged(name: str, ahn_repo: str | Path | None = None) -> Path:
    """Merge once, reuse afterwards. Merging is slow and deterministic."""
    repo = config.ahn_repo(ahn_repo)
    target = config.project_root() / arm(name).merged_dir
    if (target / "config.json").exists():
        return target
    _ensure_transformers_for_ahn()
    env = os.environ.copy()
    # ByteDance package lives at vendor/AHN/src/ahn (namespace). Prefer it explicitly.
    bytedance_src = str(repo / "src")
    env["PYTHONPATH"] = bytedance_src + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    # flash-attn wheels need libcudart.so.12 from the nvidia-* site-packages.
    nvidia_libs = sorted(
        str(p) for p in (config.project_root() / ".venv/lib").glob("python*/site-packages/nvidia/*/lib")
        if p.is_dir()
    )
    if nvidia_libs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(nvidia_libs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    subprocess.run(merge_command(name, repo), check=True, env=env, cwd=str(repo))
    return target


def _register_ahn_custom_classes() -> None:
    """Register ByteDance Qwen2/Qwen3 AHN classes with transformers Auto*.

    Without this, ``from_pretrained`` on a merged checkpoint loads stock Qwen2 and
    silently drops every ``*.ahn.*`` weight.
    """
    _ensure_transformers_for_ahn()
    from ahn.transformer.qwen2_ahn import register_customized_qwen2
    from ahn.transformer.qwen3_ahn import register_customized_qwen3

    register_customized_qwen2()
    register_customized_qwen3()


def load(name: str, ahn_repo: str | Path | None = None, sliding_window: int | None = None):
    """Return `(model, tokenizer)` with the matched settings and a forced window.

    The window override is the fix for `protocol/open_decisions.md` #1. A merged AHN
    checkpoint loads through a stock `Qwen2ForCausalLM` that carries Qwen2.5's own,
    much larger window; without forcing it, the target never leaves exact attention
    and the whole experiment measures prompt length instead of memory.

    ``ahn_repo`` is optional: when omitted, resolution follows ``config.ahn_repo``
    (env ``AHN_REPO``, Colab ``/content/AHN``, or local ``vendor/AHN``).
    """
    _ensure_transformers_for_ahn()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = arm(name)
    models = config.experiment()["models"]
    matched = models["matched"]

    if spec.needs_merge:
        source = str(ensure_merged(name, ahn_repo))
        _register_ahn_custom_classes()
    else:
        source = spec.base_repo

    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=getattr(torch, matched["torch_dtype"]),
        device_map=matched["device_map"],
        trust_remote_code=matched["trust_remote_code"],
        attn_implementation=matched["attn_implementation"],
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=matched["trust_remote_code"])

    window = sliding_window if sliding_window is not None else models["sliding_window"]["force"]
    _force_window(model, window, verbose=models["sliding_window"]["verify_on_checkpoint"])
    return model, tokenizer


def _force_window(model, window: int, verbose: bool = True) -> None:
    reported = getattr(model.config, "sliding_window", None)
    if verbose:
        print(
            f"checkpoint reports sliding_window={reported}, "
            f"use_sliding_window={getattr(model.config, 'use_sliding_window', None)}"
            f" -> forcing {window}"
        )
    model.config.sliding_window = int(window)
    model.config.use_sliding_window = True
    for module in model.modules():
        if hasattr(module, "sliding_window"):
            module.sliding_window = int(window)


def effective_window(model) -> int:
    window = getattr(model.config, "sliding_window", None)
    if window is None:
        raise ValueError(
            "Model reports no sliding_window, so the exact/recurrent boundary is "
            "undefined. Resolve before running."
        )
    return int(window)


def describe(name: str, model, tokenizer) -> dict[str, Any]:
    """Snapshot of the fields that must agree across arms."""
    return {
        "arm": name,
        "sliding_window": effective_window(model),
        "dtype": str(next(model.parameters()).dtype),
        "vocab_size": int(model.config.vocab_size),
        "tokenizer_hash": _tokenizer_fingerprint(tokenizer),
        "hidden_size": int(model.config.hidden_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "ahn_params": arm(name).ahn_params,
    }


def _tokenizer_fingerprint(tokenizer) -> str:
    """Cheap check that every arm tokenises identically.

    Pressure is counted in model tokens, so a tokenizer difference would silently
    change the x-axis of every figure.
    """
    import hashlib

    probe = "The vault code is 41977 on 12 March, filed under Meridian Holdings."
    payload = f"{len(tokenizer)}::{tokenizer(probe)['input_ids']}".encode()
    return hashlib.sha256(payload).hexdigest()[:12]


_ALLOWED_TO_DIFFER = {"arm", "ahn_params"}


def assert_matched(descriptions: list[dict[str, Any]]) -> None:
    """Fail the run if the arms are not comparable.

    The architecture comparison claims architecture is the only difference. If this
    raises, that claim is false.
    """
    if len(descriptions) < 2:
        return

    reference, *rest = descriptions
    mismatches = [
        f"{key}: {reference['arm']}={expected!r} vs {other['arm']}={other.get(key)!r}"
        for key, expected in reference.items()
        if key not in _ALLOWED_TO_DIFFER
        for other in rest
        if other.get(key) != expected
    ]
    if mismatches:
        raise AssertionError("Arms are not matched:\n  " + "\n  ".join(mismatches))
