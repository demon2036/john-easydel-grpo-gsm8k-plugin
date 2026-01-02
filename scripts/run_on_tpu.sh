#!/usr/bin/env bash
set -euo pipefail

if ! python3 -m venv .venv; then
  sudo apt-get update -y
  sudo apt-get install -y python3-venv python3.12-venv
  python3 -m venv .venv
fi
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

python3 train_grpo_gsm8k.py
