import os
import re

import easydel as ed
import jax.numpy as jnp
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-1.7B")
HF_TOKEN = os.environ.get("HF_TOKEN")
MAX_TRAINING_STEPS = int(os.environ.get("MAX_TRAINING_STEPS", "10"))
MAX_PROMPT_LENGTH = int(os.environ.get("MAX_PROMPT_LENGTH", "512"))
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", "256"))
TOTAL_BATCH_SIZE = int(os.environ.get("TOTAL_BATCH_SIZE", "8"))
NUM_RETURN_SEQUENCES = int(os.environ.get("NUM_RETURN_SEQUENCES", "2"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "200"))


def _extract_text(completion):
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return first.get("content", "")
    return completion or ""


def format_reward(prompts, completions, **kwargs):
    """Lightweight reward: prefer outputs with final numeric answer markers."""
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        if re.search(r"####\s*-?\d+", text):
            rewards.append(1.0)
            continue
        if re.search(r"-?\d+\s*$", text):
            rewards.append(0.2)
            continue
        rewards.append(0.0)
    return rewards


def build_dataset():
    dataset = load_dataset("gsm8k", "main", split="train")
    if MAX_SAMPLES > 0:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

    def _make_prompt(example):
        question = example["question"].strip()
        prompt = (
            "Solve the math problem step by step. "
            "End with '#### <answer>' on a new line.\n"
            f"Question: {question}\nAnswer:"
        )
        return {"prompt": prompt}

    return dataset.map(_make_prompt, remove_columns=dataset.column_names)


def main():
    token_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **token_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        backend=ed.EasyDeLBackends.TPU,
        platform=ed.EasyDeLPlatforms.PALLAS,
        auto_shard_model=True,
        sharding_axis_dims=(1, 1, 1, -1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            attn_mechanism=ed.AttentionMechanisms.AUTO,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
        ),
        partition_axis=ed.PartitionAxis(),
        **token_kwargs,
    )

    config = ed.GRPOConfig(
        model_name="qwen3-1.7b-gsm8k-grpo-test",
        save_directory="runs/grpo_gsm8k_test",
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_training_steps=MAX_TRAINING_STEPS,
        total_batch_size=TOTAL_BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_return_sequences=NUM_RETURN_SEQUENCES,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        beta=0.04,
        learning_rate=1e-6,
        scheduler=ed.EasyDeLSchedulers.LINEAR,
        num_train_epochs=1,
        report_steps=1,
        save_steps=None,
        use_wandb=False,
        use_grain=True,
        skip_apply_chat_template=True,
    )

    dataset = build_dataset()

    trainer = ed.GRPOTrainer(
        model=model,
        arguments=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[format_reward],
    )

    trainer.train()


if __name__ == "__main__":
    main()
