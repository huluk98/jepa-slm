#!/usr/bin/env python3
"""Recommend a tokenizer for the JEPA-SLM training setup."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenizerCandidate:
    name: str
    vocab: str
    score: float
    strengths: tuple[str, ...]
    risks: tuple[str, ...]


CANDIDATES = (
    TokenizerCandidate(
        name="T5-style SentencePiece Unigram",
        vocab="32000 base SPM slots + 100 sentinels; 32128 model embedding rows",
        score=95.0,
        strengths=(
            "Native fit for T5/UL2 span corruption.",
            "Supports sentinel tokens for denoising targets.",
            "Works directly from raw text and is stable for encoder-decoder training.",
            "The 32128 embedding size matches common T5 configs.",
        ),
        risks=(
            "Less code-specialized than a large byte-level BPE.",
            "Needs care around sentinel token ordering.",
        ),
    ),
    TokenizerCandidate(
        name="Byte-level BPE",
        vocab="32000-50000 merges",
        score=82.0,
        strengths=(
            "Robust for arbitrary web/code bytes.",
            "Popular for decoder-only LMs.",
        ),
        risks=(
            "Less natural for T5 sentinel-span corruption.",
            "Can produce longer sequences for some educational prose.",
        ),
    ),
    TokenizerCandidate(
        name="Reuse an existing T5 tokenizer",
        vocab="32128 model rows in common HF configs",
        score=78.0,
        strengths=(
            "Fastest way to start ablations.",
            "Compatible with many seq2seq libraries.",
        ),
        risks=(
            "Not trained on SmolLM-Corpus/FineWeb-Edu distribution.",
            "Less ideal if code/tutorial data is a serious part of the mixture.",
        ),
    ),
    TokenizerCandidate(
        name="Byte/character model",
        vocab="256 bytes plus specials",
        score=62.0,
        strengths=(
            "No unknown tokens.",
            "Simple and multilingual-friendly.",
        ),
        risks=(
            "Much longer sequences at 0.2B scale.",
            "Higher attention cost hurts H20 throughput for this model.",
        ),
    ),
)


def table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def make_report() -> str:
    rows = [(c.name, f"{c.score:.1f}", c.vocab) for c in sorted(CANDIDATES, key=lambda x: -x.score)]
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    best = max(CANDIDATES, key=lambda x: x.score)

    report = f"""# Tokenizer Recommendation For JEPA-SLM

Generated: {now}

## Recommendation

Use **{best.name}**.

Current repo status: the model config previously only assumed a numeric vocabulary size. The tokenizer should be made explicit as:

```text
32000 base SentencePiece Unigram vocabulary slots
+ 100 T5-style sentinel tokens: <extra_id_0> ... <extra_id_99>
= 32100 active tokenizer IDs
rounded/padded to 32128 model embedding rows
```

The recommended model config value is therefore:

```yaml
vocab_size: 32128
```

## Ranking

{table(("candidate", "score", "vocab"), rows)}

## Why This Tokenizer

T5/UL2-style training depends on span corruption. The decoder target is not the whole original text; it is a sequence of sentinel tokens and missing spans. That means sentinel tokens are not optional implementation details. They are part of the objective.

SentencePiece Unigram is the closest fit because it trains from raw text, handles whitespace internally, and is the tokenizer family used by T5-style models. For this project, byte fallback is enabled so web/code/tutorial text does not collapse into `<unk>` when unusual characters appear.

The 32000 SentencePiece slots include SentencePiece meta pieces and, when enabled, byte-fallback pieces. The sentinel tokens are added by the T5 tokenizer wrapper rather than learned inside the SentencePiece model.

## How To Train

Pilot tokenizer training:

```bash
conda activate jepa-h20
python scripts/train_sentencepiece_tokenizer.py \\
  --dataset HuggingFaceFW/fineweb-edu \\
  --subset sample-10BT \\
  --text-field text \\
  --output-dir artifacts/tokenizer \\
  --vocab-size 32000 \\
  --extra-ids 100 \\
  --max-docs 5000000
```

Main tokenizer training should sample the same proportions as the main corpus:

```text
55% fineweb-edu-dedup
20% cosmopedia-v2
10% math
10% code
5% instruction distillation
```

## Validation Gates

- Unknown-token rate should be near zero when byte fallback is enabled.
- Median tokens per document should not inflate badly versus an existing T5 tokenizer.
- Sentinel IDs must round-trip exactly.
- `len(tokenizer)` should expose the active tokenizer size, while model embeddings should use `32128`.
- Re-run the parameter verifier after changing vocab size.

## Candidate Notes

"""

    chunks = [report]
    for candidate in sorted(CANDIDATES, key=lambda x: -x.score):
        chunks.append(f"### {candidate.name}\n")
        chunks.append("Strengths:\n")
        for item in candidate.strengths:
            chunks.append(f"- {item}\n")
        chunks.append("Risks:\n")
        for item in candidate.risks:
            chunks.append(f"- {item}\n")
        chunks.append("\n")
    return "".join(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tokenizer recommendation report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tokenizer_recommendation.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(make_report(), encoding="utf-8")
    print("Best tokenizer: T5-style SentencePiece Unigram")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
