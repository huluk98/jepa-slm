#!/usr/bin/env python3
"""Exact token-length accounting of a cleaned JSONL corpus under a tokenizer.

Reports total tokens, per-document length percentiles, packed-block counts, and
the padded-mode truncation loss — the numbers needed to size max_steps.

    .venv/bin/python scripts/corpus_token_stats.py \
        --corpus "data/clean/fineweb-edu-sample10bt-full/clean-*.jsonl"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jepa_slm.config import DataSettings
from jepa_slm.data import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corpus token accounting.")
    parser.add_argument(
        "--corpus", default="data/clean/fineweb-edu-sample10bt-100k/clean-*.jsonl"
    )
    parser.add_argument("--tokenizer-name", default="google-t5/t5-small")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--source-length", type=int, default=512)
    parser.add_argument("--target-length", type=int, default=256)
    parser.add_argument("--batch-docs", type=int, default=512)
    parser.add_argument("--sequences-per-step", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(args.corpus))
    if not paths:
        raise SystemExit(f"no shards match {args.corpus!r}")

    tokenizer = load_tokenizer(
        DataSettings(tokenizer_name=args.tokenizer_name, tokenizer_path=args.tokenizer_path)
    )
    print(f"[tokenizer] {type(tokenizer).__name__} len={len(tokenizer)}")

    doc_tokens: list[int] = []
    start = time.perf_counter()

    def flush(texts: list[str]) -> None:
        for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
            doc_tokens.append(len(ids))

    batch: list[str] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                batch.append(json.loads(line)["text"])
                if len(batch) >= args.batch_docs:
                    flush(batch)
                    batch = []
                    if len(doc_tokens) % 512_000 == 0:
                        rate = len(doc_tokens) / (time.perf_counter() - start)
                        print(f"  {len(doc_tokens)} docs ({rate:.0f}/s)", flush=True)
    if batch:
        flush(batch)

    docs = len(doc_tokens)
    total = sum(doc_tokens)
    doc_tokens.sort()

    def pct(p: float) -> int:
        return doc_tokens[min(docs - 1, int(p / 100 * docs))]

    source, target, seq_step = args.source_length, args.target_length, args.sequences_per_step
    packed_stream = total + docs  # one EOS separator per document
    blocks = packed_stream // source
    padded_seen = sum(min(t + 1, source) for t in doc_tokens)

    print(f"\n[corpus] {docs:,} docs, {time.perf_counter() - start:.0f}s")
    print(f"total content tokens     : {total:,}")
    print(
        f"tokens per doc           : mean={total / docs:.0f} median={pct(50)} "
        f"p5={pct(5)} p95={pct(95)} max={doc_tokens[-1]}"
    )
    print(
        f"docs > {source} tokens        : "
        f"{sum(1 for t in doc_tokens if t > source) / docs:.1%}"
    )
    print(f"\n[packed, source_length={source}]")
    print(f"full blocks              : {blocks:,}")
    print(f"steps/epoch @ {seq_step} blocks : {blocks // seq_step:,}")
    print(f"\n[padded, source {source} / target {target}]")
    print(
        f"source tokens seen       : {padded_seen:,} "
        f"({padded_seen / packed_stream:.1%} of corpus; rest lost to truncation)"
    )
    print(f"steps/epoch @ {seq_step} docs   : {docs // seq_step:,}")


if __name__ == "__main__":
    main()
    # pyarrow/tokenizers can deadlock in exit teardown after large runs (seen on
    # macOS); all output is flushed, so skip interpreter teardown.
    sys.stdout.flush()
    os._exit(0)
