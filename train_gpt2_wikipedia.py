#!/usr/bin/env python3
"""Train GPT-2 small from random weights on tokenized English Wikipedia."""

from __future__ import annotations

import json
from importlib.metadata import version as package_version
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

SEED = int(os.environ.get("TRAIN_SEED", "42"))
CONTEXT_LENGTH = 1024


def project_root() -> Path:
    value = os.environ.get("PROJECT")
    if not value:
        raise RuntimeError(
            "PROJECT is not set. Export PROJECT=/anvil/projects/x-cis261275."
        )
    root = Path(value).expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"PROJECT does not exist: {root}")
    return root


@dataclass
class ExactLengthCausalCollator:
    """Stack exact-length blocks and preserve EOS tokens as prediction labels.

    DataCollatorForLanguageModeling(mlm=False) masks every token equal to the
    tokenizer's pad_token_id. Because GPT-2 commonly uses EOS as PAD, that would
    also mask genuine article-boundary EOS tokens. These examples are already
    exactly 1,024 tokens, so no padding is needed and labels can be an exact copy.
    """

    expected_length: int = CONTEXT_LENGTH

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        rows = [feature["input_ids"] for feature in features]
        if not rows or any(len(row) != self.expected_length for row in rows):
            lengths = [len(row) for row in rows]
            raise RuntimeError(f"Malformed training batch lengths: {lengths}")
        input_ids = torch.tensor(rows, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


def validate_block_lengths(dataset_dict: DatasetDict) -> None:
    for split_name in ("train", "validation"):
        split = dataset_dict[split_name]
        if len(split) == 0:
            raise RuntimeError(f"{split_name} split is empty")
        if "input_ids" not in split.column_names:
            raise RuntimeError(f"{split_name} lacks input_ids")
        for chunk in split.data.column("input_ids").chunks:
            if pa.types.is_fixed_size_list(chunk.type):
                if chunk.type.list_size != CONTEXT_LENGTH:
                    raise RuntimeError(
                        f"{split_name} has block size {chunk.type.list_size}, expected 1024"
                    )
            else:
                lengths = pc.list_value_length(chunk)
                stats = pc.min_max(lengths).as_py()
                if int(stats["min"]) != CONTEXT_LENGTH or int(stats["max"]) != CONTEXT_LENGTH:
                    raise RuntimeError(
                        f"{split_name} block lengths are {stats}, expected exactly 1024"
                    )


def main() -> None:
    project = project_root()
    training = project / "training"
    wiki = training / "wikipedia"
    tokenized_dir = wiki / "tokenized_gpt2_1024"
    tokenizer_dir = wiki / "gpt2_tokenizer"
    output_root = training / "gpt2-wikipedia"
    output_dir = output_root / f"seed-{SEED}"
    hf_cache = wiki / "hf_cache"
    tokenization_info_path = tokenized_dir / "TOKENIZATION_INFO.json"

    if not (tokenized_dir / "_SUCCESS").exists():
        raise RuntimeError(
            f"Tokenized dataset is missing or incomplete: {tokenized_dir}. "
            "Run tokenize_wikipedia.py first."
        )
    if not tokenizer_dir.exists():
        raise RuntimeError(f"Saved GPT-2 tokenizer not found: {tokenizer_dir}")
    if not tokenization_info_path.exists():
        raise RuntimeError(f"Tokenization metadata not found: {tokenization_info_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "transformers"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dataset = load_from_disk(str(tokenized_dir))
    if not isinstance(dataset, DatasetDict):
        raise RuntimeError("Tokenized path did not contain a DatasetDict")
    if set(dataset.keys()) != {"train", "validation"}:
        raise RuntimeError(f"Unexpected splits: {list(dataset.keys())}")
    validate_block_lengths(dataset)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenization_info = json.loads(tokenization_info_path.read_text(encoding="utf-8"))

    if int(tokenization_info["context_length"]) != CONTEXT_LENGTH:
        raise RuntimeError("Tokenized context length does not match model context length")
    if int(tokenization_info["vocab_size"]) != len(tokenizer):
        raise RuntimeError(
            f"Tokenizer metadata vocab={tokenization_info['vocab_size']} but loaded "
            f"tokenizer vocab={len(tokenizer)}"
        )

    micro_batch = int(os.environ.get("MICRO_BATCH_SIZE", "32"))
    eval_batch = int(os.environ.get("EVAL_BATCH_SIZE", str(micro_batch)))
    grad_accum = int(os.environ.get("GRAD_ACCUM_STEPS", "2"))
    epochs = float(os.environ.get("NUM_TRAIN_EPOCHS", "1"))
    max_steps = int(os.environ.get("MAX_STEPS", "-1"))
    eval_steps = int(os.environ.get("EVAL_STEPS", "5000"))
    save_steps = int(os.environ.get("SAVE_STEPS", str(eval_steps)))
    logging_steps = int(os.environ.get("LOGGING_STEPS", "100"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "5e-4"))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", "1000"))
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    dataloader_workers = max(1, min(8, allocated_cpus - 1 if allocated_cpus > 1 else 1))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    effective_sequences = micro_batch * grad_accum * world_size
    effective_tokens = effective_sequences * CONTEXT_LENGTH
    epoch_steps = math.ceil(len(dataset["train"]) / effective_sequences)

    print("Dataset columns:", dataset["train"].column_names, flush=True)
    print(f"Train blocks: {len(dataset['train']):,}", flush=True)
    print(f"Validation blocks: {len(dataset['validation']):,}", flush=True)
    print(f"Total train tokens: {len(dataset['train']) * CONTEXT_LENGTH:,}", flush=True)
    print(f"Tokenizer vocabulary: {len(tokenizer):,}", flush=True)
    print(f"Model vocabulary will be: {len(tokenizer):,}", flush=True)
    print(f"Micro-batch per GPU: {micro_batch}", flush=True)
    print(f"Gradient accumulation: {grad_accum}", flush=True)
    print(f"Effective batch: {effective_sequences:,} sequences / {effective_tokens:,} tokens", flush=True)
    print(f"Approximate optimizer steps per epoch: {epoch_steps:,}", flush=True)
    print(f"DataLoader workers: {dataloader_workers} of {allocated_cpus} allocated CPUs", flush=True)
    print(f"Output directory: {output_dir}", flush=True)

    for split_name in ("train", "validation"):
        row = dataset[split_name][0]["input_ids"]
        if len(row) != CONTEXT_LENGTH:
            raise RuntimeError(f"{split_name} first row has length {len(row)}")
        if min(row) < 0 or max(row) >= len(tokenizer):
            raise RuntimeError(
                f"{split_name} contains token id outside [0, {len(tokenizer) - 1}]"
            )
        print(
            f"{split_name} sample: length={len(row)}, min_id={min(row)}, max_id={max(row)}",
            flush=True,
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This training script must run in an AI GPU job.")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"CUDA device count: {torch.cuda.device_count()}", flush=True)
    print(f"GPU 0: {torch.cuda.get_device_name(0)}", flush=True)

    set_seed(SEED)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=CONTEXT_LENGTH,
        n_ctx=CONTEXT_LENGTH,
        n_embd=768,
        n_layer=12,
        n_head=12,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if config.vocab_size != len(tokenizer):
        raise RuntimeError("Model configuration vocabulary does not match tokenizer")

    model = GPT2LMHeadModel(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Randomly initialized model parameters: {parameter_count:,}", flush=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=micro_batch,
        per_device_eval_batch_size=eval_batch,
        gradient_accumulation_steps=grad_accum,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        logging_strategy="steps",
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        report_to="tensorboard",
        run_name=f"gpt2-wikipedia-seed{SEED}",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        load_best_model_at_end=True,
        auto_find_batch_size=False,
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=dataloader_workers,
        dataloader_persistent_workers=dataloader_workers > 0,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        include_num_input_tokens_seen=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=ExactLengthCausalCollator(),
        processing_class=tokenizer,
    )

    run_info = {
        "seed": SEED,
        "dataset_path": str(tokenized_dir),
        "tokenizer_path": str(tokenizer_dir),
        "output_dir": str(output_dir),
        "vocab_size": len(tokenizer),
        "context_length": CONTEXT_LENGTH,
        "parameter_count": parameter_count,
        "micro_batch": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "effective_sequences_per_update": effective_sequences,
        "effective_tokens_per_update": effective_tokens,
        "epochs": epochs,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "package_versions": {
            "accelerate": package_version("accelerate"),
            "datasets": package_version("datasets"),
            "pyarrow": package_version("pyarrow"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
        },
    }
    (output_dir / "RUN_INFO.json").write_text(
        json.dumps(run_info, indent=2, sort_keys=True), encoding="utf-8"
    )

    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint:
        print(f"Resuming training from {last_checkpoint}", flush=True)
    else:
        print("Starting training from random weights", flush=True)

    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_model_dir = output_dir / "final"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    print(f"Saved final model and tokenizer to {final_model_dir}", flush=True)

    model.eval()
    prompt = "The history of artificial intelligence"
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("Generation sanity check:", flush=True)
    print(tokenizer.decode(generated[0], skip_special_tokens=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFATAL: {exc}", file=sys.stderr, flush=True)
        raise
