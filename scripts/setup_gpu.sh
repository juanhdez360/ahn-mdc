#!/usr/bin/env bash
# Install ByteDance AHN + GPU deps into the uv .venv (not system Python).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
UV="${UV:-$(command -v uv)}"

echo "python: $PY"
test -x "$PY" || { echo "missing .venv — run: uv sync"; exit 1; }

if [[ ! -f vendor/AHN/examples/scripts/utils/merge_weights.py ]]; then
  rm -rf vendor/AHN
  mkdir -p vendor
  git clone --depth 1 https://github.com/ByteDance-Seed/AHN.git vendor/AHN
fi

"$UV" pip install --python "$PY" "git+https://github.com/Seerkfang/flash-linear-attention.git@main"
"$UV" pip install --python "$PY" --constraint scripts/constraints-gpu.txt "git+https://github.com/Seerkfang/LLaMA-Factory.git@main"
"$UV" pip install --python "$PY" "flash-attn==2.8.3" --extra-index-url https://wheels.astral.sh/simple/cu128/
"$UV" pip install --python "$PY" -e vendor/AHN
# ByteDance AHN needs these pins; flash-attn cu12 needs the CUDA 12 runtime libs.
"$UV" pip install --python "$PY" "transformers==4.51.0" wandb nvidia-cuda-runtime-cu12

echo "OK — kernel: Python (ahn-mdc / uv). Experiment package is ahnexp; ByteDance is ahn."
echo "If merge fails on libcudart.so.12, export LD_LIBRARY_PATH with nvidia/*/lib from .venv."
