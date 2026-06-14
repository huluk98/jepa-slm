#!/usr/bin/env python3
"""Search tokenizer and model-shape choices for the JEPA-SLM recipe."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import jepa_verifier_agent as verifier


@dataclass(frozen=True)
class TokenizerOption:
    name: str
    family: str
    base_vocab_size: int
    extra_ids: int
    embedding_vocab_size: int
    score: float
    rationale: str
    risk: str


@dataclass(frozen=True)
class DesignReport:
    candidate: verifier.ModelCandidate
    params: verifier.ParamEstimate
    memory: verifier.MemoryEstimate
    score: float
    notes: tuple[str, ...]


TOKENIZERS = (
    TokenizerOption(
        name="T5-style SentencePiece Unigram",
        family="sentencepiece_unigram",
        base_vocab_size=32_000,
        extra_ids=100,
        embedding_vocab_size=32_128,
        score=96.0,
        rationale="Best fit for span corruption because sentinels are first-class tokens and the model remains T5-compatible.",
        risk="Slightly less code-specialized than a larger byte-level BPE.",
    ),
    TokenizerOption(
        name="Byte-level BPE with sentinels",
        family="byte_bpe",
        base_vocab_size=32_768,
        extra_ids=100,
        embedding_vocab_size=32_896,
        score=84.0,
        rationale="Very robust for arbitrary bytes and code-heavy text.",
        risk="Less natural for T5/UL2 span-corruption targets than SentencePiece Unigram.",
    ),
    TokenizerOption(
        name="SmolLM-style larger BPE",
        family="byte_bpe",
        base_vocab_size=49_152,
        extra_ids=100,
        embedding_vocab_size=49_280,
        score=80.0,
        rationale="Strong precedent for decoder-only small LMs trained on SmolLM-Corpus.",
        risk="Consumes about 13M more embedding parameters than 32128 at d_model=768, which is expensive in a 0.2B encoder-decoder.",
    ),
    TokenizerOption(
        name="Byte/character tokenizer",
        family="byte_or_char",
        base_vocab_size=256,
        extra_ids=100,
        embedding_vocab_size=384,
        score=58.0,
        rationale="No unknown tokens and simple multilingual behavior.",
        risk="Longer sequences raise attention cost enough to hurt a small H20 training run.",
    ),
)


def round_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def architecture_space(vocab_size: int) -> list[verifier.ModelCandidate]:
    candidates: list[verifier.ModelCandidate] = []
    for d_model in (640, 704, 768, 832):
        head_options = tuple(
            heads
            for heads in range(8, 17)
            if d_model % heads == 0 and d_model // heads in (48, 64, 80, 88, 96)
        )
        ff_options = sorted(
            {
                round_to_multiple(int(d_model * 3.0), 256),
                round_to_multiple(int(d_model * 3.5), 256),
                round_to_multiple(int(d_model * 4.0), 256),
            }
        )
        for d_ff in ff_options:
            for heads in head_options:
                for encoder_layers in (8, 10, 12, 14):
                    for decoder_layers in (6, 8, 10, 12):
                        if encoder_layers + 2 < decoder_layers:
                            continue
                        predictor_width = min(512, verifier.round_to_multiple(int(d_model * 0.67), 64))
                        predictor_layers = 2 if d_model <= 704 else 3
                        candidates.append(
                            verifier.ModelCandidate(
                                name=(
                                    f"d{d_model}-e{encoder_layers}-d{decoder_layers}-"
                                    f"ff{d_ff}-h{heads}-p{predictor_width}x{predictor_layers}"
                                ),
                                d_model=d_model,
                                encoder_layers=encoder_layers,
                                decoder_layers=decoder_layers,
                                d_ff=d_ff,
                                num_heads=heads,
                                predictor_width=predictor_width,
                                predictor_layers=predictor_layers,
                                vocab_size=vocab_size,
                            )
                        )
    return candidates


def score_architecture(
    candidate: verifier.ModelCandidate,
    params: verifier.ParamEstimate,
    memory: verifier.MemoryEstimate,
    target_params: int,
    vram_gb: float,
) -> tuple[float, tuple[str, ...]]:
    score = 82.0
    notes: list[str] = []

    distance = abs(params.trainable_params - target_params) / target_params
    score -= distance * 70.0

    head_dim = candidate.d_model // candidate.num_heads
    if head_dim == 64:
        score += 4.0
        notes.append("64-dim heads match T5/FlashAttention-friendly practice")
    elif head_dim in (80, 96):
        score -= 1.5
    else:
        score -= 4.0

    if candidate.num_heads % 4 == 0:
        score += 2.0
    else:
        score -= 2.0

    if candidate.d_model == 768:
        score += 5.0
        notes.append("width/head shape matches T5-base proportions")
    elif candidate.d_model in (704, 832):
        score -= 1.0

    if candidate.d_ff == 4 * candidate.d_model:
        score += 4.0
        notes.append("4x FFN is the clean T5-style non-gated setting")
    elif candidate.d_ff < 3.5 * candidate.d_model:
        score -= 4.0

    if candidate.encoder_layers < candidate.decoder_layers:
        score -= 8.0
    elif candidate.encoder_layers - candidate.decoder_layers in (0, 2, 4):
        score += 3.0
        notes.append("encoder capacity is sufficient for JEPA targets")

    if candidate.decoder_layers < 8:
        score -= 6.0
    if candidate.encoder_layers + candidate.decoder_layers < 18:
        score -= 4.0

    predictor_overhead = params.predictor_params / max(1, params.trainable_params - params.predictor_params)
    score -= max(0.0, predictor_overhead - 0.08) * 120.0
    score -= max(0.0, memory.peak_gb / vram_gb - 0.85) * 80.0

    return min(100.0, max(0.0, score)), tuple(notes)


def evaluate_architectures(args: argparse.Namespace, tokenizer: TokenizerOption) -> list[DesignReport]:
    reports: list[DesignReport] = []
    for candidate in architecture_space(tokenizer.embedding_vocab_size):
        params = verifier.estimate_params(candidate)
        memory = verifier.estimate_memory(
            candidate,
            params,
            source_len=args.source_len,
            target_len=args.target_len,
            micro_batch=args.micro_batch,
            checkpointing=args.gradient_checkpointing,
        )
        checks = verifier.check_candidate(
            candidate,
            params,
            memory,
            target_params=args.target_params,
            tolerance=args.tolerance,
            vram_gb=args.vram_gb,
        )
        if any(check.status == verifier.FAIL for check in checks):
            continue
        score, notes = score_architecture(candidate, params, memory, args.target_params, args.vram_gb)
        reports.append(DesignReport(candidate, params, memory, score, notes))
    return sorted(reports, key=lambda item: item.score, reverse=True)


def markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def make_report(args: argparse.Namespace) -> str:
    tokenizer_rows = [
        (
            option.name,
            option.embedding_vocab_size,
            f"{option.score:.1f}",
            option.rationale,
        )
        for option in sorted(TOKENIZERS, key=lambda item: item.score, reverse=True)
    ]

    best_tokenizer = max(TOKENIZERS, key=lambda item: item.score)
    architecture_reports = evaluate_architectures(args, best_tokenizer)
    if not architecture_reports:
        raise SystemExit("No feasible architecture candidates found.")

    best = architecture_reports[0]
    c = best.candidate
    p = best.params
    top_rows = []
    for report in architecture_reports[: args.top_k]:
        candidate = report.candidate
        params = report.params
        top_rows.append(
            (
                candidate.d_model,
                candidate.encoder_layers,
                candidate.decoder_layers,
                candidate.d_ff,
                candidate.num_heads,
                candidate.d_model // candidate.num_heads,
                verifier.format_params(params.trainable_params),
                f"{report.score:.1f}",
            )
        )

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""# JEPA-SLM Model Design Search

Generated: {now}

## Final Recommendation

Use this tokenizer:

```text
{best_tokenizer.name}
base SentencePiece vocabulary: 32000
T5 sentinel tokens: 100
active tokenizer IDs: 32100
model embedding rows: {best_tokenizer.embedding_vocab_size}
```

Use this model shape:

```yaml
d_model: {c.d_model}
encoder_layers: {c.encoder_layers}
decoder_layers: {c.decoder_layers}
d_ff: {c.d_ff}
attention_heads: {c.num_heads}
attention_head_dim: {c.d_model // c.num_heads}
vocab_size: {best_tokenizer.embedding_vocab_size}
trainable_params: {verifier.format_params(p.trainable_params)}
```

This is the best practical point for a 0.2B JEPA-augmented encoder-decoder because it keeps the clean T5-base width/head/FFN ratios while reducing depth from 12+12 to fit the JEPA predictor and EMA encoder budget.

## Tokenizer Search

{markdown_table(("tokenizer", "embedding vocab", "score", "rationale"), tokenizer_rows)}

Decision: choose the T5-style SentencePiece Unigram tokenizer. The larger SmolLM-style 49152-token BPE is attractive for decoder-only small LMs, but at `d_model=768` it adds about 13M tied embedding parameters compared with `32128`. For this encoder-decoder, that steals too much capacity from layers.

## Architecture Search

{markdown_table(("d_model", "enc", "dec", "d_ff", "heads", "head_dim", "params", "score"), top_rows)}

Decision: choose `d_model=768`, `encoder_layers=10`, `decoder_layers=10`, `d_ff=3072`, `attention_heads=12`.

Close alternative: `d_model=768`, `encoder_layers=12`, `decoder_layers=8`, `d_ff=3072`, `attention_heads=12`. Use it if encoder representations matter more than generation quality. The balanced 10/10 split is safer for a first full run because the decoder still has enough capacity for span-corruption and seq2seq generation.

## Evidence Anchors

- Hugging Face T5 config defaults and public T5-base use `vocab_size=32128`.
- T5-base uses `d_model=768`, `d_ff=3072`, and `num_heads=12`.
- T5 tokenizer APIs expose `extra_ids=100` sentinel tokens and are based on Unigram.
- SentencePiece trains subword models directly from raw sentences, which fits FineWeb-Edu and SmolLM-Corpus streaming.
- SmolLM uses a 49152-token tokenizer for decoder-only SLMs and reports strong results, but its larger vocab is a worse parameter trade for this 0.2B encoder-decoder.
- MobileLLM shows that architecture choices matter at sub-billion scale and favors efficient depth, embedding sharing, and attention efficiency.

## Practical Training Note

Keep this as the first serious training run:

```text
T5/UL2 span corruption CE + encoder-side JEPA auxiliary loss
lambda_jepa warmup to 0.25
tied embeddings
encoder-only EMA target
```

If the CE-only baseline beats CE+JEPA after equal tokens, test the 12-encoder/8-decoder alternative before changing tokenizer size.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search JEPA-SLM tokenizer and model shape.")
    parser.add_argument("--target-params", type=int, default=200_000_000)
    parser.add_argument("--tolerance", type=float, default=0.12)
    parser.add_argument("--source-len", type=int, default=512)
    parser.add_argument("--target-len", type=int, default=256)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--vram-gb", type=float, default=24.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/model_design_search_report.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = make_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print("Best tokenizer: T5-style SentencePiece Unigram")
    print("Best architecture: d_model=768, encoder_layers=10, decoder_layers=10, d_ff=3072, heads=12")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
