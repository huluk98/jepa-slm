#!/usr/bin/env python3
"""Rank FFN activations and position encodings for JEPA encoder-decoders."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ActivationCandidate:
    name: str
    score: float
    parameter_rule: str
    pros: tuple[str, ...]
    cons: tuple[str, ...]
    recommendation: str


@dataclass(frozen=True)
class PositionCandidate:
    name: str
    score: float
    pros: tuple[str, ...]
    cons: tuple[str, ...]
    recommendation: str


ACTIVATIONS = (
    ActivationCandidate(
        name="SwiGLU",
        score=94.0,
        parameter_rule="Use about 2/3 of the non-gated d_ff to keep parameter count stable.",
        pros=(
            "Best modern default for custom LLM-style Transformer blocks.",
            "Gating improves FFN expressivity and token-wise feature selection.",
            "SiLU's smoothness is useful inside the gate without using plain SiLU everywhere.",
            "Works well with pre-norm/RMSNorm-style small models.",
        ),
        cons=(
            "Adds a second up projection, so keeping the same d_ff increases parameters and FLOPs.",
            "Slightly less plug-and-play with vanilla Hugging Face T5 code than GELU/GEGLU.",
            "Can produce larger activations than GELU, so BF16 training should keep clipping/scale monitoring.",
        ),
        recommendation="Use for the custom JEPA-SLM implementation, with adjusted d_ff.",
    ),
    ActivationCandidate(
        name="GEGLU",
        score=90.0,
        parameter_rule="Same gated parameter rule as SwiGLU.",
        pros=(
            "Strong T5-family choice; easy to justify for encoder-decoder models.",
            "Gated like SwiGLU while staying closer to T5.1.1 conventions.",
            "Good compatibility story if starting from T5-style code.",
        ),
        cons=(
            "Slightly less common than SwiGLU in newer decoder-only LLM stacks.",
            "Still needs d_ff reduction for fair parameter matching.",
        ),
        recommendation="Use if T5 compatibility matters more than modern LLM convention.",
    ),
    ActivationCandidate(
        name="GELU",
        score=82.0,
        parameter_rule="Non-gated FFN; current d_ff values are already sized for it.",
        pros=(
            "Simple, stable, and widely implemented.",
            "Good baseline for ablations because parameter accounting is straightforward.",
            "Lower activation outlier risk than gated SiLU variants.",
        ),
        cons=(
            "No gate, so it is usually less expressive per layer than GEGLU/SwiGLU.",
            "Less aligned with current small-LLM architecture practice.",
        ),
        recommendation="Keep as the baseline if you want minimal implementation risk.",
    ),
    ActivationCandidate(
        name="Plain SiLU",
        score=68.0,
        parameter_rule="Non-gated FFN; same parameter shape as GELU.",
        pros=(
            "Smooth and cheap.",
            "Useful as the nonlinear component inside SwiGLU.",
        ),
        cons=(
            "Plain SiLU is not the part that usually gives modern LLM gains; the gate is.",
            "Less benchmark precedent than GELU or gated GLU variants for encoder-decoder pretraining.",
        ),
        recommendation="Do not use as the main FFN activation unless testing a control ablation.",
    ),
)


POSITIONS = (
    PositionCandidate(
        name="T5 relative position bias",
        score=92.0,
        pros=(
            "Native fit for T5-style encoder-decoder models.",
            "Simple in encoder self-attention, decoder self-attention, and cross-attention.",
            "Works well at the 512-token source and 256-token target lengths in this repo.",
            "Least likely to break JEPA target/student alignment.",
        ),
        cons=(
            "Less attractive if the model must extrapolate far beyond trained context length.",
            "Not the dominant convention in newer decoder-only LLMs.",
        ),
        recommendation="Use for the first serious JEPA+encoder-decoder run.",
    ),
    PositionCandidate(
        name="RoPE in encoder and decoder self-attention only",
        score=86.0,
        pros=(
            "Good modern relative-position behavior without learned position tables.",
            "Improves long-context extrapolation story compared with fixed learned positions.",
            "Can be tested without making cross-attention geometry ambiguous.",
        ),
        cons=(
            "It is not a tokenizer change; it changes attention Q/K rotation.",
            "Less T5-compatible and needs careful implementation in bidirectional encoder attention.",
            "Cross-attention still needs a deliberate policy.",
        ),
        recommendation="Run as the RoPE ablation after the relative-bias baseline.",
    ),
    PositionCandidate(
        name="Full RoPE including cross-attention",
        score=72.0,
        pros=(
            "Most consistent if every attention module is rotary.",
            "Potentially useful for very long source/target alignments.",
        ),
        cons=(
            "Cross-attention has two coordinate systems: source positions and target positions.",
            "Easy to introduce hard-to-debug alignment bugs in seq2seq generation.",
            "Not necessary for HomeBench-style short command inputs.",
        ),
        recommendation="Research ablation only, not first-run architecture.",
    ),
    PositionCandidate(
        name="Learned absolute positions",
        score=58.0,
        pros=(
            "Very simple to implement.",
            "Fine for fixed short command lengths.",
        ),
        cons=(
            "Weak extrapolation behavior.",
            "Less aligned with both T5 and modern RoPE-based LLM practice.",
        ),
        recommendation="Avoid unless you need the simplest possible debugging baseline.",
    ),
)


def gated_d_ff(non_gated_d_ff: int, multiple: int = 64) -> int:
    """Approximate equal-parameter gated FFN width.

    Non-gated FFN uses two large projections. Gated GLU variants use three.
    Keeping parameters roughly fixed means d_ff_gated ~= 2/3 d_ff_non_gated.
    """

    raw = int(round(non_gated_d_ff * 2.0 / 3.0))
    return max(multiple, int(round(raw / multiple) * multiple))


def markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def bullet(items: tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in items)


def make_report() -> str:
    activation_rows = [
        (item.name, f"{item.score:.1f}", item.parameter_rule, item.recommendation)
        for item in sorted(ACTIVATIONS, key=lambda item: item.score, reverse=True)
    ]
    position_rows = [
        (item.name, f"{item.score:.1f}", item.recommendation)
        for item in sorted(POSITIONS, key=lambda item: item.score, reverse=True)
    ]

    best_activation = max(ACTIVATIONS, key=lambda item: item.score)
    best_position = max(POSITIONS, key=lambda item: item.score)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""# JEPA Encoder-Decoder Activation And Position Report

Generated: {now}

## Recommendation

Use **{best_activation.name}** for a custom implementation, but keep the first positional baseline as **{best_position.name}**.

RoPE is **not** a tokenizer choice. Keep the tokenizer as T5-style SentencePiece Unigram with `vocab_size: 32128`; evaluate RoPE only as a positional-attention ablation.

Recommended activation settings:

```yaml
# 0.2B shape
ffn_activation: swiglu
d_ff: {gated_d_ff(3072)}

# 0.1B HomeBench-style shape
ffn_activation: swiglu
d_ff: {gated_d_ff(2048)}
```

If you stay with the current non-gated FFN, use `GELU` and keep `d_ff: 3072` for the 0.2B model and `d_ff: 2048` for the 0.1B model.

## Activation Ranking

{markdown_table(("activation", "score", "parameter rule", "recommendation"), activation_rows)}

### SwiGLU

Pros:

{bullet(ACTIVATIONS[0].pros)}

Cons:

{bullet(ACTIVATIONS[0].cons)}

### GEGLU

Pros:

{bullet(ACTIVATIONS[1].pros)}

Cons:

{bullet(ACTIVATIONS[1].cons)}

### GELU

Pros:

{bullet(ACTIVATIONS[2].pros)}

Cons:

{bullet(ACTIVATIONS[2].cons)}

### Plain SiLU

Pros:

{bullet(ACTIVATIONS[3].pros)}

Cons:

{bullet(ACTIVATIONS[3].cons)}

## Position Encoding Ranking

{markdown_table(("position method", "score", "recommendation"), position_rows)}

## RoPE Decision

Use relative position bias first. It is simpler, T5-native, and safer for encoder-decoder cross-attention. Then run a RoPE ablation in encoder self-attention and decoder self-attention only.

Do not call RoPE a tokenizer improvement. The tokenizer decides token IDs; RoPE changes how attention represents token positions after tokenization.

## JEPA-Specific Notes

- The JEPA target encoder and student encoder should use the same positional scheme.
- Relative bias is safer for masked-span target alignment because it is already standard in T5-style bidirectional encoders.
- SwiGLU/GEGLU may improve the semantic quality of encoder states, but the latent loss can amplify activation outliers. Track representation norm, variance, and BF16 overflow/NaN counters.
- If SwiGLU destabilizes early training, fall back to GEGLU or GELU before changing the tokenizer.

## First Ablation Matrix

| run | activation | position | d_ff 0.2B | d_ff 0.1B | purpose |
| --- | --- | --- | --- | --- | --- |
| A | GELU | relative bias | 3072 | 2048 | compatibility baseline |
| B | SwiGLU | relative bias | {gated_d_ff(3072)} | {gated_d_ff(2048)} | recommended custom run |
| C | GEGLU | relative bias | {gated_d_ff(3072)} | {gated_d_ff(2048)} | T5-family gated run |
| D | SwiGLU | RoPE self-attn only | {gated_d_ff(3072)} | {gated_d_ff(2048)} | position ablation |

## Evidence Anchors

- GLU variant work found that gated variants such as GEGLU/SwiGLU improve Transformer FFN quality over ReLU/GELU-style baselines.
- PaLM-style modern LLM stacks use SwiGLU, which is a strong precedent for custom dense LLM blocks.
- T5-small and T5-base configs use encoder-decoder relative attention buckets, which supports relative bias as the compatibility baseline.
- RoFormer introduced RoPE as a rotary positional method that encodes relative position behavior in attention, but that is separate from tokenization.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank activation and position choices for JEPA encoder-decoders.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/activation_position_report.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(make_report(), encoding="utf-8")
    print("Best activation: SwiGLU with reduced d_ff")
    print("Best first position method: T5 relative position bias")
    print("RoPE: run as self-attention ablation, not tokenizer")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
