#!/usr/bin/env python3
"""Write cleaned training text as local JSONL shards."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jepa_slm.text_cleaning import clean_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cleaned JEPA-SLM corpus shards.")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--input", action="append", default=[], help="Local .txt/.jsonl path or glob.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        # Keep in sync with configs/train_clean_local.yaml data.dataset glob.
        default=Path("data/clean/fineweb-edu-sample10bt-100k"),
    )
    parser.add_argument("--max-docs", type=int, default=100_000)
    parser.add_argument("--shard-docs", type=int, default=50_000)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=20_000)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=10_000)
    return parser.parse_args()


def iter_local_inputs(patterns: list[str], text_field: str) -> Iterator[object]:
    for pattern in patterns:
        paths = [Path(path) for path in sorted(glob.glob(pattern))] or [Path(pattern)]
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    if path.suffix == ".jsonl":
                        yield json.loads(line).get(text_field)
                    else:
                        yield line


def iter_hf_dataset(args: argparse.Namespace) -> Iterator[object]:
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, name=args.subset, split=args.split, streaming=True)
    for row in dataset:
        yield row.get(args.text_field)


def clean_output_dir(output_dir: Path, overwrite: bool) -> None:
    existing = list(output_dir.glob("clean-*.jsonl")) + list(output_dir.glob("clean-*.jsonl.tmp"))
    existing += [output_dir / "metadata.json"]
    existing = [path for path in existing if path.exists()]
    if existing and not overwrite:
        raise SystemExit(f"{output_dir} already has cleaned files; pass --overwrite or use a new output dir.")
    for path in existing:
        path.unlink()


def open_shard(output_dir: Path, index: int):
    path = output_dir / f"clean-{index:05d}.jsonl"
    tmp_path = path.with_suffix(".jsonl.tmp")
    return path, tmp_path, tmp_path.open("w", encoding="utf-8")


def publish_shard(path: Path, tmp_path: Path) -> None:
    if tmp_path.exists() and tmp_path.stat().st_size:
        tmp_path.replace(path)
    elif tmp_path.exists():
        tmp_path.unlink()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir(args.output_dir, args.overwrite)
    source = iter_local_inputs(args.input, args.text_field) if args.input else iter_hf_dataset(args)
    max_docs = None if args.max_docs <= 0 else args.max_docs

    shard_index = 0
    shard_docs = 0
    seen = 0
    kept = 0
    chars = 0
    shard_path, tmp_path, handle = open_shard(args.output_dir, shard_index)
    completed = False

    try:
        for value in source:
            seen += 1
            text = clean_text(
                value,
                normalize=not args.no_normalize,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
            )
            if text is None:
                if max_docs is not None and seen >= max_docs:
                    break
                continue

            handle.write(json.dumps({"text": text}, ensure_ascii=False))
            handle.write("\n")
            kept += 1
            shard_docs += 1
            chars += len(text)

            if shard_docs >= args.shard_docs:
                handle.close()
                publish_shard(shard_path, tmp_path)
                shard_index += 1
                shard_docs = 0
                shard_path, tmp_path, handle = open_shard(args.output_dir, shard_index)

            if max_docs is not None and seen >= max_docs:
                break
            if args.log_every > 0 and seen % args.log_every == 0:
                print(f"seen={seen} kept={kept} chars={chars}", flush=True)
        completed = True
    finally:
        handle.close()
        if completed:
            publish_shard(shard_path, tmp_path)
        elif tmp_path.exists():
            tmp_path.unlink()

    metadata = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "inputs": args.input,
        "text_field": args.text_field,
        "seen_documents": seen,
        "kept_documents": kept,
        "kept_characters": chars,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "normalize_text": not args.no_normalize,
        "shard_docs": args.shard_docs,
        "complete": True,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Clean shards: {args.output_dir / 'clean-*.jsonl'}")
    print(f"Seen documents: {seen}")
    print(f"Kept documents: {kept}")
    print(f"Kept characters: {chars}")


if __name__ == "__main__":
    main()
    # pyarrow's IO thread pool can deadlock in its C++ destructor at interpreter
    # shutdown (seen on macOS) after streaming HF datasets. All output — shards
    # and metadata.json — is fully written by now, so skip interpreter teardown.
    sys.stdout.flush()
    os._exit(0)
