#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
fi

if ! "$PYTHON_BIN" -m venv .venv; then
  sudo apt-get update -y
  sudo apt-get install -y python3-venv python3.12-venv
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

python train_grpo_gsm8k.py
