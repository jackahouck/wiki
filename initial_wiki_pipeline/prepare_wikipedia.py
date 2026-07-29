#!/usr/bin/env python3
"""Download and prepare rendered English Wikipedia for Hugging Face.

Source:
    wikimedia/structured-wikipedia, configuration enwiki_namespace_0

Output:
    $PROJECT/training/wikipedia/cleaned

The output is a Hugging Face DatasetDict with 99% train / 1% validation
splits and these columns:
    id, title, text, url, revision_id, license

The script is restartable at the source-Parquet-shard level. It never loads
article text for the entire corpus into RAM. Only the numeric id and revision
columns are materialized during the optional duplicate-removal pass.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
from importlib.metadata import version as package_version
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import snapshot_download

DATASET_REPO = "wikimedia/structured-wikipedia"
DATASET_CONFIG = "enwiki_namespace_0"
DATASET_REVISION = "d2604837664ae7bbf1574108fc2bbf5e54f3b0ed"
SNAPSHOT_DATE = "2026-05-13"
SEED = 42
VALIDATION_FRACTION = 0.01
SOURCE_PREFIX = "https://en.wikipedia.org/wiki/"
HTML_TAG_RE = re.compile(r"<[A-Za-z][^>]{0,500}>")
WHITESPACE_RE = re.compile(r"[ \t\x0b\f\r]+")
MANY_NEWLINES_RE = re.compile(r"\n{3,}")

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("title", pa.string()),
        pa.field("text", pa.large_string()),
        pa.field("url", pa.string()),
        pa.field("revision_id", pa.int64()),
        pa.field("license", pa.string()),
    ]
)


def project_root() -> Path:
    value = os.environ.get("PROJECT")
    if not value:
        raise RuntimeError(
            "PROJECT is not set. On Anvil, export "
            "PROJECT=/anvil/projects/x-cis261275 before running."
        )
    root = Path(value).expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"PROJECT does not exist: {root}")
    return root


def worker_count(requested: int | None = None) -> int:
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    limit = min(8, max(1, allocated))
    if requested is not None:
        limit = min(limit, max(1, requested))
    return limit


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def parse_json_field(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def clean_visible_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = text.translate({
        ord("\u00a0"): " ",  # no-break space
        ord("\u202f"): " ",  # narrow no-break space
        ord("\u2009"): " ",  # thin space
        ord("\u2007"): " ",  # figure space
    })
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def cell_text(cell: Any) -> str:
    if not isinstance(cell, dict):
        return clean_visible_text(cell)
    value = clean_visible_text(cell.get("value"))
    if value:
        return value
    nested = parse_json_field(cell.get("nested_table"))
    if isinstance(nested, dict):
        return flatten_table(nested)
    return ""


def flatten_table(table: dict[str, Any]) -> str:
    lines: list[str] = []
    for row in table.get("headers") or []:
        values = [cell_text(cell) for cell in row]
        values = [value for value in values if value]
        if values:
            lines.append(" | ".join(values))
    for row in table.get("rows") or []:
        values = [cell_text(cell) for cell in row]
        values = [value for value in values if value]
        if values:
            lines.append(" | ".join(values))
    return "\n".join(lines)


def table_lookup(tables: Any) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not isinstance(tables, list):
        return lookup
    for table in tables:
        if isinstance(table, dict) and table.get("identifier"):
            lookup[str(table["identifier"])] = table
    return lookup


def flatten_part(part: Any, tables: dict[str, dict[str, Any]]) -> list[str]:
    if not isinstance(part, dict):
        text = clean_visible_text(part)
        return [text] if text else []

    kind = str(part.get("type") or "").strip().lower().replace("_", "-")
    children = part.get("has_parts") or []
    child_lines: list[str] = []
    if isinstance(children, list):
        for child in children:
            child_lines.extend(flatten_part(child, tables))

    own_lines: list[str] = []
    value = clean_visible_text(part.get("value"))
    name = clean_visible_text(part.get("name"))

    if kind == "section":
        if value:
            own_lines.append(value)
        # "Abstract" is a synthetic label for the lead section, not visible text.
        if name and name.casefold() != "abstract" and (own_lines or child_lines):
            return [name, *own_lines, *child_lines]
        return [*own_lines, *child_lines]

    if kind in {"paragraph", "list-item", "list_item", "field", "definition", "term"}:
        if name and value and name.casefold() not in value.casefold():
            own_lines.append(f"{name}: {value}")
        elif value:
            own_lines.append(value)
        elif name and not child_lines:
            own_lines.append(name)

    elif kind in {"list", "definition-list", "definitionlist"}:
        values = part.get("values") or []
        if isinstance(values, list):
            own_lines.extend(clean_visible_text(item) for item in values if clean_visible_text(item))
        elif value:
            own_lines.append(value)

    elif kind == "table":
        references = part.get("table_references") or []
        if isinstance(references, list):
            for reference in references:
                identifier = reference.get("identifier") if isinstance(reference, dict) else None
                table = tables.get(str(identifier)) if identifier is not None else None
                if table:
                    rendered = flatten_table(table)
                    if rendered:
                        own_lines.append(rendered)
        # Some records contain inline table-like content instead of references.
        if not own_lines and (part.get("rows") or part.get("headers")):
            rendered = flatten_table(part)
            if rendered:
                own_lines.append(rendered)

    elif kind in {"image", "images"}:
        # Keep human-visible captions/alt text but never URLs or markup.
        if value:
            own_lines.append(value)
        for image in part.get("images") or []:
            if isinstance(image, dict):
                caption = clean_visible_text(image.get("caption"))
                alt = clean_visible_text(image.get("alternative_text"))
                if caption:
                    own_lines.append(caption)
                elif alt:
                    own_lines.append(alt)

    else:
        # Future schema additions: retain visible scalar text and recurse.
        if value:
            own_lines.append(value)
        elif name and not child_lines:
            own_lines.append(name)

    return [line for line in [*own_lines, *child_lines] if line]


def flatten_sections(sections_value: Any, tables_value: Any) -> str:
    sections = parse_json_field(sections_value)
    tables = table_lookup(parse_json_field(tables_value))
    if not isinstance(sections, list):
        return ""

    lines: list[str] = []
    for section in sections:
        lines.extend(flatten_part(section, tables))

    cleaned: list[str] = []
    previous = None
    for line in lines:
        line = clean_visible_text(line)
        if line and line != previous:
            cleaned.append(line)
            previous = line
    return "\n\n".join(cleaned).strip()


def extract_license(row: dict[str, Any]) -> str:
    licenses = row.get("license") or []
    if isinstance(licenses, list) and licenses:
        first = licenses[0]
        if isinstance(first, dict):
            return str(first.get("identifier") or first.get("name") or "")
    return ""


def process_source_shard(args: tuple[str, str, str]) -> dict[str, Any]:
    source_str, output_str, stats_str = args
    source = Path(source_str)
    output = Path(output_str)
    stats_path = Path(stats_str)

    if output.exists() and stats_path.exists():
        return json.loads(stats_path.read_text(encoding="utf-8"))

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_suffix(output.suffix + ".tmp")
    temp_stats = stats_path.with_suffix(stats_path.suffix + ".tmp")
    temp_output.unlink(missing_ok=True)
    temp_stats.unlink(missing_ok=True)

    parquet = pq.ParquetFile(source)
    columns = [
        "identifier",
        "name",
        "url",
        "version",
        "in_language",
        "is_part_of",
        "license",
        "sections",
        "tables",
    ]

    writer: pq.ParquetWriter | None = None
    stats = {
        "source": str(source),
        "output": str(output),
        "seen": 0,
        "written": 0,
        "empty_or_broken": 0,
        "invalid_source": 0,
        "json_or_schema_failures": 0,
    }

    try:
        writer = pq.ParquetWriter(
            temp_output,
            OUTPUT_SCHEMA,
            compression="zstd",
            use_dictionary=["title", "url", "license"],
        )

        pending: dict[str, list[Any]] = {field.name: [] for field in OUTPUT_SCHEMA}

        def flush() -> None:
            if not pending["id"]:
                return
            arrays = [pa.array(pending[field.name], type=field.type) for field in OUTPUT_SCHEMA]
            writer.write_table(pa.Table.from_arrays(arrays, schema=OUTPUT_SCHEMA))
            for values in pending.values():
                values.clear()

        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                stats["seen"] += 1
                language = row.get("in_language") or {}
                project = row.get("is_part_of") or {}
                url = str(row.get("url") or "")
                if (
                    language.get("identifier") != "en"
                    or project.get("identifier") != "enwiki"
                    or not url.startswith(SOURCE_PREFIX)
                ):
                    stats["invalid_source"] += 1
                    continue

                title = clean_visible_text(row.get("name"))
                try:
                    body = flatten_sections(row.get("sections"), row.get("tables"))
                except Exception:
                    stats["json_or_schema_failures"] += 1
                    continue

                if not title or len(body) < 20:
                    stats["empty_or_broken"] += 1
                    continue

                article_text = f"{title}\n\n{body}".strip()
                if len(article_text) < 30:
                    stats["empty_or_broken"] += 1
                    continue

                version = row.get("version") or {}
                revision_id = int(version.get("identifier") or 0)
                pending["id"].append(int(row["identifier"]))
                pending["title"].append(title)
                pending["text"].append(article_text)
                pending["url"].append(url)
                pending["revision_id"].append(revision_id)
                pending["license"].append(extract_license(row))
                stats["written"] += 1

                if len(pending["id"]) >= 512:
                    flush()

        flush()
        writer.close()
        writer = None

        if stats["invalid_source"]:
            raise RuntimeError(
                f"{source.name}: found {stats['invalid_source']} records that were not "
                "English enwiki records. Refusing to continue."
            )
        if stats["written"] == 0:
            raise RuntimeError(f"{source.name}: produced no usable articles")

        os.replace(temp_output, output)
        temp_stats.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp_stats, stats_path)
        return stats
    except Exception:
        if writer is not None:
            writer.close()
        temp_output.unlink(missing_ok=True)
        temp_stats.unlink(missing_ok=True)
        raise


def deduplicate_latest_revision(dataset: Dataset) -> Dataset:
    if len(dataset) == 0:
        raise RuntimeError("Cannot deduplicate an empty dataset")

    ids = dataset.data.column("id").combine_chunks().to_numpy(zero_copy_only=False)
    revisions = dataset.data.column("revision_id").combine_chunks().to_numpy(zero_copy_only=False)
    order = np.lexsort((revisions, ids))
    sorted_ids = ids[order]
    keep_sorted = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        keep_sorted[:-1] = sorted_ids[:-1] != sorted_ids[1:]
    keep_indices = np.sort(order[keep_sorted])
    duplicate_count = len(dataset) - len(keep_indices)
    print(f"Duplicate article snapshots removed: {duplicate_count:,}", flush=True)
    if duplicate_count:
        dataset = dataset.select(keep_indices.tolist())
    return dataset


def preserved_milwaukee_values(text: str) -> list[str]:
    normalized = clean_visible_text(text)
    found: list[str] = []
    if re.search(r"8,376\s*m\b", normalized):
        found.append("8,376 m")
    if re.search(r"27,480\s*ft\b", normalized):
        found.append("27,480 ft")
    return found


def find_milwaukee_deep(dataset: Dataset, workers: int) -> dict[str, Any]:
    matches = dataset.filter(
        lambda batch: ["milwaukee deep" in text.casefold() for text in batch["text"]],
        batched=True,
        batch_size=512,
        num_proc=workers,
        desc="Checking Milwaukee Deep rendering",
    )
    if len(matches) == 0:
        raise RuntimeError(
            "Validation failed: no cleaned article contains 'Milwaukee Deep'. "
            "The source schema or flattening logic may have changed."
        )
    for row in matches:
        preserved = preserved_milwaukee_values(row["text"])
        if preserved:
            print("Milwaukee Deep validation article:", row["title"], flush=True)
            for needle in preserved:
                print(f"  preserved value: {needle}", flush=True)
            return row
    titles = [row["title"] for row in matches.select(range(min(10, len(matches))))]
    raise RuntimeError(
        "Validation failed: 'Milwaukee Deep' was found, but neither '8,376 m' nor "
        f"'27,480 ft' was preserved. Candidate titles: {titles}"
    )


def validate_cleaned(dataset_dict: DatasetDict, workers: int) -> None:
    required_splits = {"train", "validation"}
    if set(dataset_dict.keys()) != required_splits:
        raise RuntimeError(f"Expected splits {required_splits}, found {set(dataset_dict.keys())}")

    required_columns = {"id", "title", "text", "url", "revision_id", "license"}
    for split_name, split in dataset_dict.items():
        missing = required_columns.difference(split.column_names)
        if missing:
            raise RuntimeError(f"{split_name} is missing columns: {sorted(missing)}")
        if len(split) == 0:
            raise RuntimeError(f"{split_name} is empty")

    print("\nCleaned DatasetDict validation", flush=True)
    print("Splits:", dataset_dict, flush=True)
    print("Columns:", dataset_dict["train"].column_names, flush=True)
    print(f"Train articles: {len(dataset_dict['train']):,}", flush=True)
    print(f"Validation articles: {len(dataset_dict['validation']):,}", flush=True)

    sample_count = min(10_000, len(dataset_dict["train"]))
    if sample_count:
        step = max(1, len(dataset_dict["train"]) // sample_count)
        indices = list(range(0, len(dataset_dict["train"]), step))[:sample_count]
        sample = dataset_dict["train"].select(indices)
        html_count = sum(bool(HTML_TAG_RE.search(text)) for text in sample["text"])
        bad_url_count = sum(not url.startswith(SOURCE_PREFIX) for url in sample["url"])
        print(
            f"HTML-like tags in deterministic {sample_count:,}-article sample: "
            f"{html_count:,}",
            flush=True,
        )
        print(f"Non-English-Wikipedia URLs in sample: {bad_url_count:,}", flush=True)
        if html_count > max(10, sample_count // 100):
            raise RuntimeError("Too many raw HTML-like tags remain in cleaned text")
        if bad_url_count:
            raise RuntimeError("Found non-English-Wikipedia URL(s) in cleaned output")

    # Search both splits because the deterministic split can place the article in either one.
    found = False
    for split_name in ("train", "validation"):
        matches = dataset_dict[split_name].filter(
            lambda batch: ["milwaukee deep" in text.casefold() for text in batch["text"]],
            batched=True,
            batch_size=512,
            num_proc=workers,
            desc=f"Milwaukee Deep check ({split_name})",
        )
        for row in matches:
            if preserved_milwaukee_values(row["text"]):
                print(f"Milwaukee Deep check passed in {split_name}: {row['title']}", flush=True)
                found = True
                break
        if found:
            break
    if not found:
        raise RuntimeError("Milwaukee Deep rendered-value check failed after save/load")

    for split_name in ("train", "validation"):
        split = dataset_dict[split_name]
        print(f"\n{split_name} samples:", flush=True)
        for index in [0, len(split) // 2, len(split) - 1]:
            row = split[index]
            preview = row["text"][:400].replace("\n", " ")
            print(f"  id={row['id']} title={row['title']!r} url={row['url']}", flush=True)
            print(f"    {preview}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Rebuild the cleaned DatasetDict")
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep cleaned Parquet staging shards after successful save_to_disk",
    )
    args = parser.parse_args()

    project = project_root()
    training = project / "training"
    wiki = training / "wikipedia"
    raw_dir = wiki / "raw"
    hf_cache = wiki / "hf_cache"
    cleaned_dir = wiki / "cleaned"
    staging_dir = hf_cache / "cleaned_parquet_staging"
    temp_final = wiki / "cleaned.incomplete"
    workers = worker_count(args.workers)

    for directory in (raw_dir, hf_cache, staging_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache / "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache / "hub"))

    if cleaned_dir.exists() and (cleaned_dir / "_SUCCESS").exists() and not args.force:
        print(f"Found completed cleaned dataset: {cleaned_dir}", flush=True)
        validate_cleaned(load_from_disk(str(cleaned_dir)), workers)
        return

    if args.force:
        shutil.rmtree(cleaned_dir, ignore_errors=True)
        shutil.rmtree(temp_final, ignore_errors=True)

    print(f"PROJECT: {project}", flush=True)
    print(f"Workers: {workers}", flush=True)
    print(f"Downloading only {DATASET_CONFIG} from {DATASET_REPO}", flush=True)

    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=["enwiki/data/*.parquet", "enwiki/schema.json", "README.md"],
        local_dir=str(raw_dir),
    )

    source_shards = sorted((raw_dir / "enwiki" / "data").glob("*.parquet"))
    if not source_shards:
        raise RuntimeError(
            f"No source Parquet shards found under {raw_dir / 'enwiki' / 'data'}"
        )
    print(f"Official English namespace-0 shards found: {len(source_shards):,}", flush=True)

    jobs: list[tuple[str, str, str]] = []
    for source in source_shards:
        output = staging_dir / f"{source.stem}.cleaned.parquet"
        stats = staging_dir / f"{source.stem}.stats.json"
        jobs.append((str(source), str(output), str(stats)))

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_source_shard, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            stats = future.result()
            print(
                f"Prepared {Path(stats['source']).name}: "
                f"{stats['written']:,}/{stats['seen']:,} articles kept",
                flush=True,
            )

    cleaned_shards = sorted(staging_dir.glob("*.cleaned.parquet"))
    if len(cleaned_shards) != len(source_shards):
        raise RuntimeError(
            f"Expected {len(source_shards)} cleaned shards, found {len(cleaned_shards)}"
        )

    print("Loading cleaned Parquet shards as a memory-mapped Hugging Face Dataset...", flush=True)
    full = load_dataset(
        "parquet",
        data_files=[str(path) for path in cleaned_shards],
        split="train",
        cache_dir=str(hf_cache / "datasets"),
    )
    if len(full) == 0:
        raise RuntimeError("Cleaned dataset is empty")
    print(f"Articles before duplicate removal: {len(full):,}", flush=True)
    print("Columns:", full.column_names, flush=True)

    full = deduplicate_latest_revision(full)
    find_milwaukee_deep(full, workers)

    print(f"Creating reproducible 99/1 split with seed {SEED}...", flush=True)
    split = full.train_test_split(
        test_size=VALIDATION_FRACTION,
        seed=SEED,
        shuffle=True,
    )
    dataset_dict = DatasetDict(train=split["train"], validation=split["test"])

    shutil.rmtree(temp_final, ignore_errors=True)
    print(f"Saving canonical untokenized DatasetDict to {temp_final}...", flush=True)
    dataset_dict.save_to_disk(str(temp_final))

    source_info = {
        "source_dataset": DATASET_REPO,
        "source_config": DATASET_CONFIG,
        "source_repository_revision": DATASET_REVISION,
        "source_snapshot_extracted": SNAPSHOT_DATE,
        "source_namespace": 0,
        "source_language": "en",
        "source_project": "enwiki",
        "dataset_card_license": "CC-BY-SA-4.0",
        "license_note": "Per-article license values are retained in the license column.",
        "split_seed": SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "text_construction": "title + flattened rendered sections in original order",
        "raw_directory": str(raw_dir),
        "package_versions": {
            "datasets": package_version("datasets"),
            "huggingface-hub": package_version("huggingface-hub"),
            "numpy": package_version("numpy"),
            "pyarrow": package_version("pyarrow"),
        },
    }
    atomic_write_json(temp_final / "SOURCE_INFO.json", source_info)
    (temp_final / "_SUCCESS").write_text("ok\n", encoding="utf-8")

    loaded = load_from_disk(str(temp_final))
    validate_cleaned(loaded, workers)

    shutil.rmtree(cleaned_dir, ignore_errors=True)
    os.replace(temp_final, cleaned_dir)
    print(f"\nCompleted cleaned Wikipedia dataset: {cleaned_dir}", flush=True)

    if not args.keep_staging:
        print(f"Removing temporary cleaned Parquet shards: {staging_dir}", flush=True)
        shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFATAL: {exc}", file=sys.stderr, flush=True)
        raise
