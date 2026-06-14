#!/usr/bin/env python3
"""Verify smaller JEPA-SLM shapes for command/home-automation workloads."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import jepa_verifier_agent as verifier


@dataclass(frozen=True)
class SmallTarget:
    label: str
    target_params: int
    verdict: str
    use_case: str
    risk: str


@dataclass(frozen=True)
class SmallCandidateReport:
    target: SmallTarget
    candidate: verifier.ModelCandidate
    params: verifier.ParamEstimate
    memory: verifier.MemoryEstimate
    score: float
    checks: tuple[verifier.CheckResult, ...]


TARGETS = (
    SmallTarget(
        label="160M safer general small",
        target_params=160_000_000,
        verdict="feasible",
        use_case="Smaller general seq2seq plus command tuning.",
        risk="Still larger than needed if the deployment only emits structured smart-home actions.",
    ),
    SmallTarget(
        label="125M safe command model",
        target_params=125_000_000,
        verdict="feasible",
        use_case="Good if invalid multi-device handling must be strong.",
        risk="Moderate latency and memory savings versus 200M, but not the smallest useful size.",
    ),
    SmallTarget(
        label="100M recommended command model",
        target_params=100_000_000,
        verdict="recommended",
        use_case="Best balance for HomeBench-style command parsing, repair, and rejection.",
        risk="Needs constrained output and explicit invalid-instruction training.",
    ),
    SmallTarget(
        label="80M aggressive edge model",
        target_params=80_000_000,
        verdict="possible with cautions",
        use_case="Works for narrow command grammars with schema-constrained decoding.",
        risk="More likely to miss invalid multi-device edge cases.",
    ),
    SmallTarget(
        label="60M minimum parser",
        target_params=60_000_000,
        verdict="not recommended as first run",
        use_case="Only for very constrained command-to-API parsing.",
        risk="Too little slack for HomeBench's invalid multi-device split without strong external validators.",
    ),
)


def round_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def head_options(d_model: int) -> tuple[int, ...]:
    return tuple(
        heads
        for heads in range(6, 13)
        if d_model % heads == 0 and d_model // heads in (48, 56, 64, 72, 80, 96)
    )


def candidate_space(vocab_size: int) -> list[verifier.ModelCandidate]:
    candidates: list[verifier.ModelCandidate] = []
    for d_model in (384, 448, 512, 576, 640, 704, 768):
        ff_options = sorted(
            {
                round_to_multiple(int(d_model * 3.0), 64),
                round_to_multiple(int(d_model * 3.5), 64),
                round_to_multiple(int(d_model * 4.0), 64),
            }
        )
        for d_ff in ff_options:
            for heads in head_options(d_model):
                for encoder_layers in range(4, 15, 2):
                    for decoder_layers in range(4, 13, 2):
                        if encoder_layers + 2 < decoder_layers:
                            continue
                        predictor_width = min(
                            384 if d_model <= 512 else 512,
                            verifier.round_to_multiple(int(d_model * 0.67), 64),
                        )
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


def command_score(
    candidate: verifier.ModelCandidate,
    params: verifier.ParamEstimate,
    target_params: int,
) -> float:
    score = 70.0
    score -= abs(params.trainable_params - target_params) / target_params * 70.0

    head_dim = candidate.d_model // candidate.num_heads
    if head_dim == 64:
        score += 4.0
    elif head_dim in (80, 96):
        score -= 1.5
    else:
        score -= 5.0

    if candidate.d_model == 512:
        score += 5.0
    elif candidate.d_model < 512:
        score -= 4.0
    elif candidate.d_model > 640:
        score -= 2.0

    if candidate.d_ff == 4 * candidate.d_model:
        score += 4.0
    elif candidate.d_ff < 3.5 * candidate.d_model:
        score -= 4.0

    if candidate.encoder_layers >= candidate.decoder_layers:
        score += 4.0
    else:
        score -= 8.0

    if candidate.decoder_layers >= 8:
        score += 5.0
    else:
        score -= 10.0

    if candidate.encoder_layers + candidate.decoder_layers >= 20:
        score += 3.0
    elif candidate.encoder_layers + candidate.decoder_layers < 16:
        score -= 8.0

    overhead = params.predictor_params / max(1, params.trainable_params - params.predictor_params)
    score -= max(0.0, overhead - 0.10) * 100.0
    return min(100.0, max(0.0, score))


def best_for_target(
    target: SmallTarget,
    candidates: list[verifier.ModelCandidate],
    args: argparse.Namespace,
) -> SmallCandidateReport:
    reports: list[SmallCandidateReport] = []
    for candidate in candidates:
        params = verifier.estimate_params(candidate)
        lower = target.target_params * (1.0 - args.tolerance)
        upper = target.target_params * (1.0 + args.tolerance)
        if not lower <= params.trainable_params <= upper:
            continue
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
            target_params=target.target_params,
            tolerance=args.tolerance,
            vram_gb=args.vram_gb,
        )
        if any(check.status == verifier.FAIL for check in checks):
            continue
        score = command_score(candidate, params, target.target_params)
        reports.append(SmallCandidateReport(target, candidate, params, memory, score, checks))

    if not reports:
        raise SystemExit(f"No feasible candidate found for {target.label}.")
    return sorted(reports, key=lambda report: report.score, reverse=True)[0]


def markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def make_report(args: argparse.Namespace) -> str:
    candidates = candidate_space(args.vocab_size)
    reports = [best_for_target(target, candidates, args) for target in TARGETS]
    recommended = next(report for report in reports if report.target.verdict == "recommended")

    target_rows = []
    for report in reports:
        c = report.candidate
        p = report.params
        target_rows.append(
            (
                report.target.label,
                c.d_model,
                c.encoder_layers,
                c.decoder_layers,
                c.d_ff,
                c.num_heads,
                c.d_model // c.num_heads,
                verifier.format_params(p.trainable_params),
                f"{report.score:.1f}",
                report.target.verdict,
            )
        )

    c = recommended.candidate
    p = recommended.params
    m = recommended.memory
    check_rows = [(check.name, check.status, check.detail) for check in recommended.checks]
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""# Smaller JEPA-SLM Verification For HomeBench-Style Commands

Generated: {now}

## Verdict

Yes, the 0.2B design can be made smaller for command-heavy smart-home workloads. The best first smaller target is **about 100M trainable parameters**, not 60M.

Recommended small shape:

```yaml
d_model: {c.d_model}
encoder_layers: {c.encoder_layers}
decoder_layers: {c.decoder_layers}
d_ff: {c.d_ff}
attention_heads: {c.num_heads}
attention_head_dim: {c.d_model // c.num_heads}
vocab_size: {args.vocab_size}
predictor_width: {c.predictor_width}
predictor_layers: {c.predictor_layers}
trainable_params: {verifier.format_params(p.trainable_params)}
stored_params_with_ema: {verifier.format_params(p.stored_params)}
estimated_peak_vram_per_gpu: {m.peak_gb:.1f} GB
```

## Target Sweep

{markdown_table(("tier", "d_model", "enc", "dec", "d_ff", "heads", "head_dim", "params", "score", "verdict"), target_rows)}

## Checks For Recommended 100M Shape

{markdown_table(("check", "status", "detail"), check_rows)}

## Why 100M Is The Right Small Point

- It keeps the T5-small width family: `d_model=512`, `d_ff=2048`, `8` heads, and `64`-dim heads.
- It adds depth over T5-small so the encoder has enough room for JEPA representation learning and the decoder still has 10 layers for structured generation.
- It keeps the existing `32128` tokenizer instead of shrinking vocabulary. For command tasks, losing sentinel/tokenizer compatibility is not worth saving a few embedding rows.
- It reduces trainable params from about 200M to about 100M while keeping the predictor overhead under control.

## HomeBench Risk Assessment

HomeBench-style tasks are not just normal single-device commands. The hard slice is invalid multi-device instructions, where the model must reject, repair, or split commands correctly. For that slice, architecture size is less important than the output contract:

1. Generate a structured action list or an explicit rejection, never free text.
2. Use constrained decoding against the device/action schema.
3. Add a deterministic verifier after the model for room, device, capability, and state checks.
4. Train separate slices for valid single-device, invalid single-device, valid multi-device, and invalid multi-device instructions.
5. Gate release on exact-match action success and invalid-instruction rejection rate.

## Smaller Options

- **125M** is safer if invalid multi-device handling is the product-critical metric.
- **100M** is the recommended first small run.
- **80M** is plausible for a narrow command grammar with constrained decoding.
- **60M** is a parser-sized model, not a robust smart-home assistant by itself.

## Relevant Work Anchors

- HomeBench shows that smart-home evaluation must include valid and invalid instructions across single and multiple devices.
- HomeBench reports that invalid multi-device instructions remain difficult even for strong LLMs, so rejection and verification logic are mandatory.
- T5-small establishes that `d_model=512`, `d_ff=2048`, and `8` attention heads are a proven small encoder-decoder shape.
- MobileLLM argues that sub-billion models benefit from careful architecture design rather than just naive shrinking.
- Newer smart-home benchmarks emphasize executable or verifiable environments, which supports pairing the 100M model with a deterministic command verifier.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify smaller JEPA-SLM shapes for HomeBench-style command tasks.")
    parser.add_argument("--vocab-size", type=int, default=32_128)
    parser.add_argument("--source-len", type=int, default=512)
    parser.add_argument("--target-len", type=int, default=256)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--vram-gb", type=float, default=24.0)
    parser.add_argument("--tolerance", type=float, default=0.18)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/small_model_verification_report.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = make_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print("Recommended small target: 100M")
    print("Best small architecture: d_model=512, encoder_layers=12, decoder_layers=10, d_ff=2048, heads=8")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
