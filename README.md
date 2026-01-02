# John EasyDeL Plugin: GRPO on GSM8K (Qwen3-1.7B)

This is a minimal, non-invasive plugin project for testing EasyDeL GRPO training on GSM8K with Qwen3 1.7B. It lives outside the EasyDeL repo and only depends on it via pip.

## What it does
- Runs a short GRPO training loop (10 steps) on GSM8K
- Uses a lightweight reward function based on output format/number extraction
- Designed for TPU v6e-8 (eu-w-4a)

## Quickstart (local or TPU VM)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_grpo_gsm8k.py
```

## TPU runtime version

Use `v2-alpha-tpuv6e` (from `gcloud compute tpus tpu-vm versions list --zone=europe-west4-a`).

## Environment overrides

```bash
# Optional overrides
export MODEL_ID=Qwen/Qwen3-1.7B
export MAX_TRAINING_STEPS=10
export MAX_PROMPT_LENGTH=256
export MAX_COMPLETION_LENGTH=256
export TOTAL_BATCH_SIZE=8
export NUM_RETURN_SEQUENCES=2
# If the model is gated/private:
export HF_TOKEN=hf_your_token_here
```

## Notes
- Reward is format-based (no ground-truth checking) to keep the test minimal.
- If you want correctness rewards, extend the reward function and include answers in the batch.
