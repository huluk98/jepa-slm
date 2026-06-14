#!/usr/bin/env python3
"""Train a T5-style SentencePiece tokenizer from a Hugging Face dataset stream."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train JEPA-SLM SentencePiece tokenizer.")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--model-prefix", default="jepa_slm_spm")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--extra-ids", type=int, default=100)
    parser.add_argument("--max-docs", type=int, default=5_000_000)
    parser.add_argument("--max-chars", type=int, default=8_000_000_000)
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument("--no-byte-fallback", action="store_true")
    return parser.parse_args()


def round_up(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def stream_corpus(args: argparse.Namespace, corpus_path: Path) -> dict[str, int]:
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, name=args.subset, split=args.split, streaming=True)
    docs = 0
    chars = 0

    with corpus_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            text = row.get(args.text_field)
            if not text:
                continue
            text = " ".join(str(text).split())
            if not text:
                continue
            handle.write(text)
            handle.write("\n")
            docs += 1
            chars += len(text)
            if docs >= args.max_docs or chars >= args.max_chars:
                break

    return {"documents": docs, "characters": chars}


def train_sentencepiece(args: argparse.Namespace, corpus_path: Path) -> Path:
    import sentencepiece as spm

    model_prefix = args.output_dir / args.model_prefix
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        model_type="unigram",
        character_coverage=args.character_coverage,
        byte_fallback=not args.no_byte_fallback,
        normalization_rule_name="nmt_nfkc",
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
        input_sentence_size=min(args.max_docs, 10_000_000),
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        hard_vocab_limit=False,
    )
    return model_prefix.with_suffix(".model")


def write_metadata(args: argparse.Namespace, stats: dict[str, int], model_path: Path) -> None:
    active_tokenizer_size = args.vocab_size + args.extra_ids
    embedding_vocab_size = round_up(active_tokenizer_size, 128)
    metadata = {
        "family": "t5_sentencepiece_unigram",
        "sentencepiece_model": str(model_path),
        "base_sentencepiece_vocab_size": args.vocab_size,
        "extra_sentinel_tokens": args.extra_ids,
        "active_tokenizer_size": active_tokenizer_size,
        "embedding_vocab_size": embedding_vocab_size,
        "sentinel_format": "<extra_id_{i}>",
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "byte_fallback": not args.no_byte_fallback,
        "training_dataset": args.dataset,
        "training_subset": args.subset,
        "training_stats": stats,
    }
    (args.output_dir / "tokenizer_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.output_dir / "tokenizer_corpus.txt"
    stats = stream_corpus(args, corpus_path)
    model_path = train_sentencepiece(args, corpus_path)
    write_metadata(args, stats, model_path)

    print(f"SentencePiece model: {model_path}")
    print(f"Documents sampled: {stats['documents']}")
    print(f"Characters sampled: {stats['characters']}")
    print(f"Metadata: {args.output_dir / 'tokenizer_metadata.json'}")


if __name__ == "__main__":
    main()
