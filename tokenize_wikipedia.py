#!/usr/bin/env python3
"""Tokenize the canonical cleaned Wikipedia DatasetDict with GPT-2.

Input:
    $PROJECT/training/wikipedia/cleaned

Output:
    $PROJECT/training/wikipedia/tokenized_gpt2_1024

Each article receives exactly one EOS token. Tokens are concatenated within
train and validation separately, split into exact 1,024-token blocks, and only
the final incomplete block of each split is dropped. The script checkpoints at
Parquet-shard boundaries and can be resubmitted after interruption.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version as package_version
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer

CONTEXT_LENGTH = 1024
TOKENIZER_NAME = "openai-community/gpt2"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SEED = 42
DEFAULT_ARTICLE_BATCH_SIZE = 32
DEFAULT_BLOCKS_PER_SHARD = 8192


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


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def write_block_shard(path: Path, blocks: list[list[int]]) -> None:
    if not blocks:
        return
    for block in blocks:
        if len(block) != CONTEXT_LENGTH:
            raise RuntimeError(f"Attempted to write block of length {len(block)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    fixed_type = pa.list_(pa.int32(), CONTEXT_LENGTH)
    table = pa.table({"input_ids": pa.array(blocks, type=fixed_type)})
    pq.write_table(table, tmp, compression="zstd", use_dictionary=False)
    os.replace(tmp, path)


def source_fingerprint(split: Dataset) -> str:
    return str(getattr(split, "_fingerprint", "unknown"))


def remove_orphan_shards(split_dir: Path, next_shard_index: int) -> None:
    for path in split_dir.glob("part-*.parquet"):
        try:
            index = int(path.stem.split("-")[1])
        except (IndexError, ValueError):
            continue
        if index >= next_shard_index:
            print(f"Removing orphan shard not committed by state file: {path}", flush=True)
            path.unlink()


def tokenize_split(
    split_name: str,
    dataset: Dataset,
    tokenizer: Any,
    staging_root: Path,
    article_batch_size: int,
    blocks_per_shard: int,
    force: bool,
) -> dict[str, Any]:
    split_dir = staging_root / split_name
    state_path = split_dir / "state.json"
    split_dir.mkdir(parents=True, exist_ok=True)

    expected_fingerprint = source_fingerprint(dataset)
    expected_vocab_size = len(tokenizer)

    if force:
        shutil.rmtree(split_dir, ignore_errors=True)
        split_dir.mkdir(parents=True, exist_ok=True)

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("source_fingerprint") != expected_fingerprint:
            raise RuntimeError(
                f"{split_name}: source dataset fingerprint changed. Use --force to rebuild."
            )
        if state.get("vocab_size") != expected_vocab_size:
            raise RuntimeError(
                f"{split_name}: tokenizer vocabulary changed. Use --force to rebuild."
            )
        if state.get("context_length") != CONTEXT_LENGTH:
            raise RuntimeError(f"{split_name}: context length changed; use --force")
    else:
        state = {
            "split": split_name,
            "source_fingerprint": expected_fingerprint,
            "vocab_size": expected_vocab_size,
            "context_length": CONTEXT_LENGTH,
            "next_article_index": 0,
            "next_shard_index": 0,
            "pending_tokens": [],
            "total_articles": len(dataset),
            "articles_processed": 0,
            "tokens_with_eos_seen": 0,
            "blocks_written": 0,
            "dropped_final_tokens": None,
            "complete": False,
        }
        atomic_write_json(state_path, state)

    if state.get("complete"):
        print(f"{split_name}: tokenization already complete", flush=True)
        return state

    remove_orphan_shards(split_dir, int(state["next_shard_index"]))

    next_article = int(state["next_article_index"])
    next_shard = int(state["next_shard_index"])
    pending_tokens = [int(token) for token in state.get("pending_tokens", [])]
    tokens_seen = int(state.get("tokens_with_eos_seen", 0))
    blocks_written = int(state.get("blocks_written", 0))
    blocks: list[list[int]] = []

    if len(pending_tokens) >= CONTEXT_LENGTH:
        raise RuntimeError(f"{split_name}: saved pending buffer is unexpectedly large")

    print(
        f"{split_name}: resuming at article {next_article:,}/{len(dataset):,}, "
        f"shard {next_shard:,}, pending tokens {len(pending_tokens):,}",
        flush=True,
    )

    def commit_shard(article_index_after_commit: int) -> None:
        nonlocal next_shard, blocks_written, blocks
        if not blocks:
            return
        shard_path = split_dir / f"part-{next_shard:06d}.parquet"
        write_block_shard(shard_path, blocks)
        blocks_written += len(blocks)
        next_shard += 1
        state.update(
            {
                "next_article_index": article_index_after_commit,
                "next_shard_index": next_shard,
                "pending_tokens": pending_tokens,
                "articles_processed": article_index_after_commit,
                "tokens_with_eos_seen": tokens_seen,
                "blocks_written": blocks_written,
            }
        )
        atomic_write_json(state_path, state)
        print(
            f"{split_name}: committed {shard_path.name}; "
            f"articles={article_index_after_commit:,}, blocks={blocks_written:,}",
            flush=True,
        )
        blocks = []

    while next_article < len(dataset):
        batch_end = min(len(dataset), next_article + article_batch_size)
        texts = dataset[next_article:batch_end]["text"]
        if not texts:
            raise RuntimeError(f"{split_name}: empty read batch at article {next_article}")

        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]

        for local_offset, article_ids in enumerate(encoded):
            article_index = next_article + local_offset
            if not article_ids:
                raise RuntimeError(f"{split_name}: article {article_index} tokenized to zero tokens")

            pending_tokens.extend(int(token) for token in article_ids)
            pending_tokens.append(int(tokenizer.eos_token_id))
            tokens_seen += len(article_ids) + 1

            full_length = (len(pending_tokens) // CONTEXT_LENGTH) * CONTEXT_LENGTH
            for start in range(0, full_length, CONTEXT_LENGTH):
                blocks.append(pending_tokens[start : start + CONTEXT_LENGTH])
            if full_length:
                pending_tokens = pending_tokens[full_length:]

            article_after = article_index + 1
            if len(blocks) >= blocks_per_shard:
                commit_shard(article_after)

        next_article = batch_end

    if blocks:
        commit_shard(len(dataset))

    dropped = len(pending_tokens)
    state.update(
        {
            "next_article_index": len(dataset),
            "next_shard_index": next_shard,
            "pending_tokens": [],
            "articles_processed": len(dataset),
            "tokens_with_eos_seen": tokens_seen,
            "blocks_written": blocks_written,
            "dropped_final_tokens": dropped,
            "complete": True,
        }
    )
    atomic_write_json(state_path, state)
    print(
        f"{split_name}: complete; articles={len(dataset):,}, blocks={blocks_written:,}, "
        f"tokens kept={blocks_written * CONTEXT_LENGTH:,}, final tokens dropped={dropped:,}",
        flush=True,
    )
    return state


def validate_fixed_length(dataset_dict: DatasetDict) -> None:
    if set(dataset_dict.keys()) != {"train", "validation"}:
        raise RuntimeError(f"Unexpected splits: {list(dataset_dict.keys())}")

    for split_name, split in dataset_dict.items():
        if split.column_names != ["input_ids"]:
            raise RuntimeError(
                f"{split_name}: expected only input_ids, found {split.column_names}"
            )
        if len(split) == 0:
            raise RuntimeError(f"{split_name}: no token blocks")

        column = split.data.column("input_ids")
        minimum = CONTEXT_LENGTH
        maximum = CONTEXT_LENGTH
        for chunk in column.chunks:
            if pa.types.is_fixed_size_list(chunk.type):
                if chunk.type.list_size != CONTEXT_LENGTH:
                    raise RuntimeError(
                        f"{split_name}: fixed-size list length is {chunk.type.list_size}"
                    )
            else:
                lengths = pc.list_value_length(chunk)
                stats = pc.min_max(lengths).as_py()
                minimum = min(minimum, int(stats["min"]))
                maximum = max(maximum, int(stats["max"]))
        if minimum != CONTEXT_LENGTH or maximum != CONTEXT_LENGTH:
            raise RuntimeError(
                f"{split_name}: block lengths range from {minimum} to {maximum}"
            )

        print(
            f"{split_name}: {len(split):,} rows; every row is exactly "
            f"{CONTEXT_LENGTH:,} tokens",
            flush=True,
        )
        for index in [0, len(split) // 2, len(split) - 1]:
            ids = split[index]["input_ids"]
            print(
                f"  sample row {index:,}: length={len(ids)}, "
                f"first_ids={ids[:12]}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--article-batch-size", type=int, default=DEFAULT_ARTICLE_BATCH_SIZE
    )
    parser.add_argument(
        "--blocks-per-shard", type=int, default=DEFAULT_BLOCKS_PER_SHARD
    )
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()

    project = project_root()
    wiki = project / "training" / "wikipedia"
    cleaned_dir = wiki / "cleaned"
    hf_cache = wiki / "hf_cache"
    output_dir = wiki / "tokenized_gpt2_1024"
    temp_output = wiki / "tokenized_gpt2_1024.incomplete"
    tokenizer_dir = wiki / "gpt2_tokenizer"
    staging_root = hf_cache / "tokenized_gpt2_1024_staging"

    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "transformers"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    tokenizer_threads = max(1, min(8, allocated_cpus))
    os.environ.setdefault("RAYON_NUM_THREADS", str(tokenizer_threads))
    print(
        f"Tokenizer threads: {tokenizer_threads} of {allocated_cpus} allocated CPUs",
        flush=True,
    )

    if not (cleaned_dir / "_SUCCESS").exists():
        raise RuntimeError(
            f"Cleaned dataset is missing or incomplete: {cleaned_dir}. "
            "Run prepare_wikipedia.py first."
        )

    if output_dir.exists() and (output_dir / "_SUCCESS").exists() and not args.force:
        print(f"Found completed tokenized dataset: {output_dir}", flush=True)
        tokenized = load_from_disk(str(output_dir))
        validate_fixed_length(tokenized)
        return

    if args.force:
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(temp_output, ignore_errors=True)
        shutil.rmtree(staging_root, ignore_errors=True)

    cleaned = load_from_disk(str(cleaned_dir))
    if not isinstance(cleaned, DatasetDict):
        raise RuntimeError("Cleaned Wikipedia path did not contain a DatasetDict")
    print("Cleaned columns:", cleaned["train"].column_names, flush=True)
    print(f"Cleaned train articles: {len(cleaned['train']):,}", flush=True)
    print(f"Cleaned validation articles: {len(cleaned['validation']):,}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        revision=TOKENIZER_REVISION,
        cache_dir=str(hf_cache),
        trust_remote_code=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(str(tokenizer_dir))
    if tokenizer.eos_token_id is None:
        raise RuntimeError("GPT-2 tokenizer has no EOS token id")
    print(f"Tokenizer: {TOKENIZER_NAME}", flush=True)
    print(f"Tokenizer vocabulary size: {len(tokenizer):,}", flush=True)
    print(f"EOS/pad token id: {tokenizer.eos_token_id}", flush=True)

    states: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "validation"):
        states[split_name] = tokenize_split(
            split_name=split_name,
            dataset=cleaned[split_name],
            tokenizer=tokenizer,
            staging_root=staging_root,
            article_batch_size=max(1, args.article_batch_size),
            blocks_per_shard=max(1, args.blocks_per_shard),
            force=args.force,
        )

    data_files: dict[str, list[str]] = {}
    for split_name in ("train", "validation"):
        files = sorted((staging_root / split_name).glob("part-*.parquet"))
        if not files:
            raise RuntimeError(f"No tokenized Parquet shards found for {split_name}")
        data_files[split_name] = [str(path) for path in files]

    print("Assembling tokenized Parquet shards into a DatasetDict...", flush=True)
    tokenized = load_dataset(
        "parquet",
        data_files=data_files,
        cache_dir=str(hf_cache / "datasets"),
    )
    validate_fixed_length(tokenized)

    total_train_tokens = len(tokenized["train"]) * CONTEXT_LENGTH
    total_validation_tokens = len(tokenized["validation"]) * CONTEXT_LENGTH
    micro_batch = int(os.environ.get("MICRO_BATCH_SIZE", "32"))
    grad_accum = int(os.environ.get("GRAD_ACCUM_STEPS", "2"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_sequences = micro_batch * grad_accum * world_size
    approximate_steps = math.ceil(len(tokenized["train"]) / effective_sequences)

    print(f"Total train tokens: {total_train_tokens:,}", flush=True)
    print(f"Total validation tokens: {total_validation_tokens:,}", flush=True)
    print(
        f"Approximate optimizer steps for one epoch at micro-batch={micro_batch}, "
        f"gradient accumulation={grad_accum}, GPUs={world_size}: {approximate_steps:,}",
        flush=True,
    )

    shutil.rmtree(temp_output, ignore_errors=True)
    print(f"Saving tokenized DatasetDict to {temp_output}...", flush=True)
    tokenized.save_to_disk(str(temp_output))

    metadata = {
        "source_cleaned_dataset": str(cleaned_dir),
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_path": str(tokenizer_dir),
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "context_length": CONTEXT_LENGTH,
        "train_blocks": len(tokenized["train"]),
        "validation_blocks": len(tokenized["validation"]),
        "train_tokens": total_train_tokens,
        "validation_tokens": total_validation_tokens,
        "split_states": states,
        "package_versions": {
            "datasets": package_version("datasets"),
            "pyarrow": package_version("pyarrow"),
            "tokenizers": package_version("tokenizers"),
            "transformers": package_version("transformers"),
        },
        "note": (
            "Only input_ids are stored. The training collator creates attention_mask and "
            "labels at batch time without masking real EOS article-boundary tokens."
        ),
    }
    atomic_write_json(temp_output / "TOKENIZATION_INFO.json", metadata)
    (temp_output / "_SUCCESS").write_text("ok\n", encoding="utf-8")

    reloaded = load_from_disk(str(temp_output))
    validate_fixed_length(reloaded)

    shutil.rmtree(output_dir, ignore_errors=True)
    os.replace(temp_output, output_dir)
    print(f"\nCompleted tokenized dataset: {output_dir}", flush=True)

    if not args.keep_staging:
        print(f"Removing temporary tokenized Parquet shards: {staging_root}", flush=True)
        shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFATAL: {exc}", file=sys.stderr, flush=True)
        raise
