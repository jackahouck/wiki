#!/usr/bin/env python3
"""Train a parameter-matched GPT-2-small control on tokenized Wikipedia.

The model is initialized from seeded random weights. It does not load an old
OpenWebText checkpoint. The control uses the exact tokenizer saved by
``tokenize_wikipedia.py``, including the bracket tokens, while its training text
contains no bracket annotations.
"""

from __future__ import annotations

import hashlib
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


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true/false or 1/0, got {value!r}")


def tokenizer_semantic_fingerprint(tokenizer: Any) -> str:
    payload = {
        "vocab": sorted((str(token), int(token_id)) for token, token_id in tokenizer.get_vocab().items()),
        "added_vocab": sorted(
            (str(token), int(token_id)) for token, token_id in tokenizer.get_added_vocab().items()
        ),
        "special_tokens_map": tokenizer.special_tokens_map,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class ExactLengthCausalCollator:
    """Stack 1,024-token blocks and keep real EOS tokens as labels."""

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


def validate_dataset(dataset_dict: DatasetDict, vocab_size: int) -> None:
    """Validate every row length and every token ID using Arrow kernels."""
    for split_name in ("train", "validation"):
        split = dataset_dict[split_name]
        if len(split) == 0:
            raise RuntimeError(f"{split_name} split is empty")
        if split.column_names != ["input_ids"]:
            raise RuntimeError(
                f"{split_name} must contain only input_ids, found {split.column_names}"
            )

        global_min: int | None = None
        global_max: int | None = None
        for chunk in split.data.column("input_ids").chunks:
            if pa.types.is_fixed_size_list(chunk.type):
                if chunk.type.list_size != CONTEXT_LENGTH:
                    raise RuntimeError(
                        f"{split_name} has block size {chunk.type.list_size}, expected 1024"
                    )
                values = chunk.values
            else:
                lengths = pc.list_value_length(chunk)
                length_stats = pc.min_max(lengths).as_py()
                if (
                    int(length_stats["min"]) != CONTEXT_LENGTH
                    or int(length_stats["max"]) != CONTEXT_LENGTH
                ):
                    raise RuntimeError(
                        f"{split_name} block lengths are {length_stats}, expected 1024"
                    )
                values = pc.list_flatten(chunk)

            if len(values):
                token_stats = pc.min_max(values).as_py()
                chunk_min = int(token_stats["min"])
                chunk_max = int(token_stats["max"])
                global_min = chunk_min if global_min is None else min(global_min, chunk_min)
                global_max = chunk_max if global_max is None else max(global_max, chunk_max)

        if global_min is None or global_max is None:
            raise RuntimeError(f"{split_name} contains no token IDs")
        if global_min < 0 or global_max >= vocab_size:
            raise RuntimeError(
                f"{split_name} token IDs range from {global_min} to {global_max}, "
                f"but valid IDs are 0 through {vocab_size - 1}"
            )
        print(
            f"{split_name}: validated all {len(split):,} rows; length=1024; "
            f"token_id_range=[{global_min}, {global_max}]",
            flush=True,
        )


def write_or_validate_run_info(path: Path, run_info: dict[str, Any]) -> None:
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        protected_keys = [
            "seed",
            "dataset_path",
            "tokenizer_fingerprint",
            "vocab_size",
            "context_length",
            "micro_batch",
            "gradient_accumulation_steps",
            "effective_sequences_per_update",
            "epochs",
            "max_steps",
            "learning_rate",
            "warmup_steps",
            "eval_steps",
            "save_steps",
        ]
        mismatches = {
            key: (old.get(key), run_info.get(key))
            for key in protected_keys
            if old.get(key) != run_info.get(key)
        }
        if mismatches:
            raise RuntimeError(
                "Existing output directory has incompatible run settings; refusing "
                f"to resume. Mismatches: {mismatches}"
            )
        return
    path.write_text(json.dumps(run_info, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    project = project_root()
    training = project / "training"
    wiki = training / "wikipedia"
    tokenized_dir = wiki / "tokenized_gpt2_1024"
    tokenizer_dir = wiki / "comparison_tokenizer"
    output_root = training / "gpt2-wikipedia"
    output_dir = output_root / f"control-brackettok-seed-{SEED}"
    hf_cache = wiki / "hf_cache"
    tokenization_info_path = tokenized_dir / "TOKENIZATION_INFO.json"

    if not (tokenized_dir / "_SUCCESS").exists():
        raise RuntimeError(
            f"Tokenized dataset is missing or incomplete: {tokenized_dir}. "
            "Run tokenize_wikipedia.py first."
        )
    if not tokenizer_dir.exists():
        raise RuntimeError(f"Saved comparison tokenizer not found: {tokenizer_dir}")
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

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Saved comparison tokenizer has no EOS token ID")
    tokenizer.pad_token = tokenizer.eos_token
    loaded_tokenizer_fingerprint = tokenizer_semantic_fingerprint(tokenizer)
    tokenization_info = json.loads(tokenization_info_path.read_text(encoding="utf-8"))

    if int(tokenization_info["context_length"]) != CONTEXT_LENGTH:
        raise RuntimeError("Tokenized context length does not match model context length")
    if int(tokenization_info["vocab_size"]) != len(tokenizer):
        raise RuntimeError(
            f"Tokenizer metadata vocab={tokenization_info['vocab_size']} but loaded "
            f"tokenizer vocab={len(tokenizer)}"
        )
    if tokenization_info.get("tokenizer_fingerprint") != loaded_tokenizer_fingerprint:
        raise RuntimeError(
            "Tokenizer mapping does not match the tokenizer that created the dataset"
        )

    validate_dataset(dataset, len(tokenizer))

    micro_batch = int(os.environ.get("MICRO_BATCH_SIZE", "32"))
    eval_batch = int(os.environ.get("EVAL_BATCH_SIZE", "64"))
    grad_accum = int(os.environ.get("GRAD_ACCUM_STEPS", "2"))
    epochs = float(os.environ.get("NUM_TRAIN_EPOCHS", "1"))
    max_steps = int(os.environ.get("MAX_STEPS", "-1"))
    eval_steps = int(os.environ.get("EVAL_STEPS", "1000"))
    save_steps = int(os.environ.get("SAVE_STEPS", str(eval_steps)))
    logging_steps = int(os.environ.get("LOGGING_STEPS", "100"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "5e-4"))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", "1000"))
    use_torch_compile = env_bool("USE_TORCH_COMPILE", False)
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    dataloader_workers = max(1, min(8, allocated_cpus - 1 if allocated_cpus > 1 else 1))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if micro_batch <= 0 or eval_batch <= 0 or grad_accum <= 0:
        raise RuntimeError("Batch sizes and gradient accumulation must be positive")
    if eval_steps <= 0 or save_steps <= 0:
        raise RuntimeError("EVAL_STEPS and SAVE_STEPS must be positive")
    if save_steps % eval_steps != 0:
        raise RuntimeError(
            "SAVE_STEPS must be a multiple of EVAL_STEPS when loading the best model"
        )

    effective_sequences = micro_batch * grad_accum * world_size
    effective_tokens = effective_sequences * CONTEXT_LENGTH
    epoch_steps = math.ceil(len(dataset["train"]) / effective_sequences)

    print("Dataset columns:", dataset["train"].column_names, flush=True)
    print(f"Train blocks: {len(dataset['train']):,}", flush=True)
    print(f"Validation blocks: {len(dataset['validation']):,}", flush=True)
    print(f"Total train tokens: {len(dataset['train']) * CONTEXT_LENGTH:,}", flush=True)
    print(f"Tokenizer vocabulary: {len(tokenizer):,}", flush=True)
    print(f"Tokenizer semantic SHA-256: {loaded_tokenizer_fingerprint}", flush=True)
    print(f"Micro-batch per GPU: {micro_batch}", flush=True)
    print(f"Evaluation batch per GPU: {eval_batch}", flush=True)
    print(f"Gradient accumulation: {grad_accum}", flush=True)
    print(f"Effective batch: {effective_sequences:,} sequences / {effective_tokens:,} tokens", flush=True)
    print(f"Approximate optimizer steps per epoch: {epoch_steps:,}", flush=True)
    print(f"DataLoader workers: {dataloader_workers} of {allocated_cpus} allocated CPUs", flush=True)
    print(f"Output directory: {output_dir}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this script in the AI GPU job.")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"CUDA device count: {torch.cuda.device_count()}", flush=True)
    print(f"GPU 0: {torch.cuda.get_device_name(0)}", flush=True)

    # This occurs before model construction, so the random weights are seeded.
    set_seed(SEED)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=CONTEXT_LENGTH,
        n_ctx=CONTEXT_LENGTH,
        n_embd=768,
        n_layer=12,
        n_head=12,
        use_cache=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if config.vocab_size != len(tokenizer):
        raise RuntimeError("Model configuration vocabulary does not match tokenizer")

    # Deliberately from random weights: do not replace this with from_pretrained().
    model = GPT2LMHeadModel(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Seeded randomly initialized model parameters: {parameter_count:,}", flush=True)

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
        run_name=f"gpt2-wikipedia-control-brackettok-seed{SEED}",
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
        prediction_loss_only=True,
        torch_compile=use_torch_compile,
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
        "initialization": "GPT2LMHeadModel(config) after transformers.set_seed(seed)",
        "dataset_path": str(tokenized_dir),
        "tokenizer_path": str(tokenizer_dir),
        "tokenizer_fingerprint": loaded_tokenizer_fingerprint,
        "output_dir": str(output_dir),
        "vocab_size": len(tokenizer),
        "context_length": CONTEXT_LENGTH,
        "parameter_count": parameter_count,
        "micro_batch": micro_batch,
        "eval_batch": eval_batch,
        "gradient_accumulation_steps": grad_accum,
        "effective_sequences_per_update": effective_sequences,
        "effective_tokens_per_update": effective_tokens,
        "epochs": epochs,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "load_best_model_at_end": True,
        "torch_compile": use_torch_compile,
        "package_versions": {
            "accelerate": package_version("accelerate"),
            "datasets": package_version("datasets"),
            "pyarrow": package_version("pyarrow"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
        },
    }
    write_or_validate_run_info(output_dir / "RUN_INFO.json", run_info)

    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint:
        print(f"Resuming training from {last_checkpoint}", flush=True)
    else:
        print("Starting training from seeded random weights", flush=True)

    trainer.train(resume_from_checkpoint=last_checkpoint)

    best_checkpoint = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric
    if best_checkpoint is None or best_metric is None:
        raise RuntimeError("Training finished without recording a best validation checkpoint")

    best_info = {
        "best_model_checkpoint": best_checkpoint,
        "best_eval_loss": float(best_metric),
        "global_step_at_training_end": int(trainer.state.global_step),
        "note": "Trainer loaded this best-validation checkpoint before final save.",
    }
    (output_dir / "BEST_CHECKPOINT.json").write_text(
        json.dumps(best_info, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Best validation checkpoint: {best_checkpoint}", flush=True)
    print(f"Best validation loss: {best_metric}", flush=True)

    # Trainer has already loaded the best checkpoint because load_best_model_at_end=True.
    trainer.model.config.use_cache = True
    final_model_dir = output_dir / "final-best-validation"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    print(f"Saved best-validation model and tokenizer to {final_model_dir}", flush=True)

    trainer.model.eval()
    prompt = "The history of artificial intelligence"
    encoded = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
    with torch.no_grad():
        generated = trainer.model.generate(
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
