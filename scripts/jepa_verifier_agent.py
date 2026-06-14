#!/usr/bin/env python3
"""Verify whether a small encoder-decoder model is a good fit for JEPA-style training.

This is a lightweight architecture verification agent. It does not train a model.
Instead it loops over plausible encoder-decoder configurations near a target
parameter budget, checks whether a JEPA/data2vec-style auxiliary objective makes
sense for the config, estimates rough training memory, and writes a markdown
report with the recommended recipe and reasoning.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


OK = "OK"
WARN = "WARN"
FAIL = "FAIL"


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    d_model: int
    encoder_layers: int
    decoder_layers: int
    d_ff: int
    num_heads: int
    predictor_width: int
    predictor_layers: int
    vocab_size: int
    top_k_target_layers: int = 4
    jepa_loss_weight: float = 0.25
    token_ce_weight: float = 1.0
    tied_embeddings: bool = True


@dataclass(frozen=True)
class ParamEstimate:
    embedding_params: int
    encoder_params: int
    decoder_params: int
    predictor_params: int
    trainable_params: int
    target_ema_params: int
    stored_params: int


@dataclass(frozen=True)
class MemoryEstimate:
    optimizer_state_gb: float
    target_ema_gb: float
    activation_gb: float
    misc_gb: float
    peak_gb: float


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class CandidateReport:
    candidate: ModelCandidate
    params: ParamEstimate
    memory: MemoryEstimate
    checks: tuple[CheckResult, ...]
    score: float


def attention_params(d_model: int) -> int:
    """Q, K, V, O projections plus small bias terms."""
    return 4 * d_model * d_model + 4 * d_model


def ffn_params(d_model: int, d_ff: int) -> int:
    """Non-gated FFN: input projection and output projection plus biases."""
    return 2 * d_model * d_ff + d_ff + d_model


def layer_norm_params(d_model: int, count: int) -> int:
    return 2 * d_model * count


def encoder_layer_params(d_model: int, d_ff: int) -> int:
    return attention_params(d_model) + ffn_params(d_model, d_ff) + layer_norm_params(d_model, 2)


def decoder_layer_params(d_model: int, d_ff: int) -> int:
    self_attn = attention_params(d_model)
    cross_attn = attention_params(d_model)
    return self_attn + cross_attn + ffn_params(d_model, d_ff) + layer_norm_params(d_model, 3)


def predictor_params(d_model: int, width: int, layers: int, ff_mult: int = 4) -> int:
    d_ff = width * ff_mult
    projection = 0
    if width != d_model:
        projection = d_model * width + width * d_model + width + d_model
    transformer = layers * (attention_params(width) + ffn_params(width, d_ff) + layer_norm_params(width, 2))
    mask_and_position = 2 * width
    return projection + transformer + mask_and_position


def estimate_params(candidate: ModelCandidate) -> ParamEstimate:
    if candidate.tied_embeddings:
        embedding_params = candidate.vocab_size * candidate.d_model
    else:
        embedding_params = 3 * candidate.vocab_size * candidate.d_model

    encoder_core = candidate.encoder_layers * encoder_layer_params(candidate.d_model, candidate.d_ff)
    decoder_core = candidate.decoder_layers * decoder_layer_params(candidate.d_model, candidate.d_ff)
    predictor = predictor_params(
        candidate.d_model,
        candidate.predictor_width,
        candidate.predictor_layers,
    )
    trainable = embedding_params + encoder_core + decoder_core + predictor

    # The EMA target should be an encoder-only copy for this design. It is stored,
    # updated after optimizer steps, and excluded from gradients/optimizer state.
    target_ema = encoder_core
    stored = trainable + target_ema

    return ParamEstimate(
        embedding_params=embedding_params,
        encoder_params=encoder_core,
        decoder_params=decoder_core,
        predictor_params=predictor,
        trainable_params=trainable,
        target_ema_params=target_ema,
        stored_params=stored,
    )


def estimate_memory(
    candidate: ModelCandidate,
    params: ParamEstimate,
    source_len: int,
    target_len: int,
    micro_batch: int,
    checkpointing: bool,
) -> MemoryEstimate:
    # Mixed precision AdamW rough accounting: bf16 params + bf16 grads +
    # fp32 master weights + fp32 first and second moments.
    optimizer_bytes_per_param = 16
    target_ema_bytes_per_param = 4
    activation_bytes = 2

    optimizer_state_gb = params.trainable_params * optimizer_bytes_per_param / 1e9
    target_ema_gb = params.target_ema_params * target_ema_bytes_per_param / 1e9

    activation_multiplier = 7 if checkpointing else 18
    encoder_tokens = micro_batch * source_len
    decoder_tokens = micro_batch * target_len
    predictor_tokens = max(1, int(encoder_tokens * 0.35))

    activation_elements = (
        encoder_tokens * candidate.d_model * candidate.encoder_layers
        + decoder_tokens * candidate.d_model * candidate.decoder_layers
        + predictor_tokens * candidate.predictor_width * candidate.predictor_layers
    )
    activation_gb = activation_elements * activation_bytes * activation_multiplier / 1e9

    # Attention workspaces, temporary logits, dataloader padding, allocator slack.
    misc_gb = 2.0
    peak_gb = optimizer_state_gb + target_ema_gb + activation_gb + misc_gb

    return MemoryEstimate(
        optimizer_state_gb=optimizer_state_gb,
        target_ema_gb=target_ema_gb,
        activation_gb=activation_gb,
        misc_gb=misc_gb,
        peak_gb=peak_gb,
    )


def round_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def candidate_space(vocab_size: int) -> Iterable[ModelCandidate]:
    for d_model in (576, 640, 704, 768, 832):
        if d_model % 64 != 0:
            continue
        d_ff = 4 * d_model
        heads = max(1, d_model // 64)
        for encoder_layers in (8, 10, 12, 14):
            for decoder_layers in (6, 8, 10, 12):
                if encoder_layers + 2 < decoder_layers:
                    continue
                predictor_width = min(512, round_to_multiple(int(d_model * 0.67), 64))
                predictor_layers = 2 if d_model <= 704 else 3
                name = (
                    f"d{d_model}-e{encoder_layers}-d{decoder_layers}-"
                    f"p{predictor_width}x{predictor_layers}"
                )
                yield ModelCandidate(
                    name=name,
                    d_model=d_model,
                    encoder_layers=encoder_layers,
                    decoder_layers=decoder_layers,
                    d_ff=d_ff,
                    num_heads=heads,
                    predictor_width=predictor_width,
                    predictor_layers=predictor_layers,
                    vocab_size=vocab_size,
                )


def check_candidate(
    candidate: ModelCandidate,
    params: ParamEstimate,
    memory: MemoryEstimate,
    target_params: int,
    tolerance: float,
    vram_gb: float,
) -> tuple[CheckResult, ...]:
    checks: list[CheckResult] = []
    lower = target_params * (1.0 - tolerance)
    upper = target_params * (1.0 + tolerance)
    wider_lower = target_params * (1.0 - tolerance * 1.75)
    wider_upper = target_params * (1.0 + tolerance * 1.75)

    if lower <= params.trainable_params <= upper:
        status = OK
    elif wider_lower <= params.trainable_params <= wider_upper:
        status = WARN
    else:
        status = FAIL
    checks.append(
        CheckResult(
            "parameter budget",
            status,
            f"{format_params(params.trainable_params)} trainable vs target {format_params(target_params)}",
        )
    )

    overhead = params.predictor_params / max(1, params.trainable_params - params.predictor_params)
    if overhead <= 0.08:
        status = OK
    elif overhead <= 0.12:
        status = WARN
    else:
        status = FAIL
    checks.append(
        CheckResult(
            "predictor overhead",
            status,
            f"predictor is {overhead:.1%} of the base encoder-decoder",
        )
    )

    if candidate.token_ce_weight > 0:
        status = OK
        detail = "token CE is retained, so the model remains generative"
    else:
        status = FAIL
        detail = "pure JEPA would not teach the decoder to emit tokens"
    checks.append(CheckResult("generation objective", status, detail))

    if candidate.top_k_target_layers >= 2 and candidate.jepa_loss_weight > 0:
        status = OK
        detail = "EMA targets plus normalized top-layer latent prediction are configured"
    else:
        status = WARN
        detail = "collapse risk is higher without multi-layer normalized targets"
    checks.append(CheckResult("collapse controls", status, detail))

    if candidate.encoder_layers >= candidate.decoder_layers:
        status = OK
        detail = "encoder capacity is at least decoder capacity, which suits representation pretraining"
    else:
        status = WARN
        detail = "decoder-heavy split is less aligned with JEPA's main benefit"
    checks.append(CheckResult("encoder emphasis", status, detail))

    if memory.peak_gb <= vram_gb * 0.85:
        status = OK
    elif memory.peak_gb <= vram_gb:
        status = WARN
    else:
        status = FAIL
    checks.append(
        CheckResult(
            "rough VRAM",
            status,
            f"estimated peak {memory.peak_gb:.1f} GB on a {vram_gb:.1f} GB budget",
        )
    )

    if params.target_ema_params < params.trainable_params * 0.6:
        status = OK
        detail = "EMA target is encoder-only and has no optimizer state"
    else:
        status = WARN
        detail = "target copy is large; avoid a full encoder-decoder teacher unless needed"
    checks.append(CheckResult("EMA target scope", status, detail))

    return tuple(checks)


def score_report(
    params: ParamEstimate,
    memory: MemoryEstimate,
    checks: tuple[CheckResult, ...],
    target_params: int,
    vram_gb: float,
) -> float:
    score = 100.0
    distance = abs(params.trainable_params - target_params) / target_params
    score -= distance * 80.0
    predictor_overhead = params.predictor_params / max(1, params.trainable_params - params.predictor_params)
    score -= max(0.0, predictor_overhead - 0.08) * 120.0
    score -= max(0.0, memory.peak_gb / vram_gb - 0.85) * 80.0
    for check in checks:
        if check.status == WARN:
            score -= 6.0
        elif check.status == FAIL:
            score -= 22.0
    return max(0.0, score)


def evaluate_candidates(args: argparse.Namespace) -> list[CandidateReport]:
    reports: list[CandidateReport] = []
    for candidate in candidate_space(args.vocab_size):
        params = estimate_params(candidate)
        memory = estimate_memory(
            candidate,
            params,
            source_len=args.source_len,
            target_len=args.target_len,
            micro_batch=args.micro_batch,
            checkpointing=args.gradient_checkpointing,
        )
        checks = check_candidate(
            candidate,
            params,
            memory,
            target_params=args.target_params,
            tolerance=args.tolerance,
            vram_gb=args.vram_gb,
        )
        score = score_report(params, memory, checks, args.target_params, args.vram_gb)
        reports.append(CandidateReport(candidate, params, memory, checks, score))
    return sorted(reports, key=lambda item: item.score, reverse=True)


def format_params(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    return f"{value / 1_000_000:.1f}M"


def status_counts(checks: Iterable[CheckResult]) -> str:
    checks = tuple(checks)
    return f"{sum(c.status == OK for c in checks)} OK / {sum(c.status == WARN for c in checks)} WARN / {sum(c.status == FAIL for c in checks)} FAIL"


def markdown_table(headers: tuple[str, ...], rows: Iterable[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def make_report(args: argparse.Namespace, reports: list[CandidateReport]) -> str:
    best = reports[0]
    c = best.candidate
    p = best.params
    m = best.memory
    top = reports[: args.top_k]

    top_rows = []
    for report in top:
        candidate = report.candidate
        params = report.params
        memory = report.memory
        top_rows.append(
            (
                candidate.name,
                format_params(params.trainable_params),
                format_params(params.encoder_params),
                format_params(params.decoder_params),
                format_params(params.predictor_params),
                f"{memory.peak_gb:.1f} GB",
                f"{report.score:.1f}",
                status_counts(report.checks),
            )
        )

    check_rows = [
        (check.name, check.status, check.detail)
        for check in best.checks
    ]

    today = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    target_b = format_params(args.target_params)
    verdict = "feasible"
    if any(check.status == FAIL for check in best.checks):
        verdict = "not yet feasible without changing constraints"
    elif any(check.status == WARN for check in best.checks):
        verdict = "feasible with cautions"

    return f"""# JEPA Encoder-Decoder Verification Report

Generated: {today}

## Verdict

The verifier rates a JEPA-style auxiliary objective for a {target_b} encoder-decoder model as **{verdict}** under the current assumptions.

Best candidate: **{c.name}**

{markdown_table(
        (
            "field",
            "value",
        ),
        (
            ("d_model", c.d_model),
            ("encoder layers", c.encoder_layers),
            ("decoder layers", c.decoder_layers),
            ("d_ff", c.d_ff),
            ("attention heads", c.num_heads),
            ("predictor", f"{c.predictor_width} width x {c.predictor_layers} layers"),
            ("trainable params", format_params(p.trainable_params)),
            ("stored params incl. EMA", format_params(p.stored_params)),
            ("estimated peak VRAM", f"{m.peak_gb:.1f} GB"),
        ),
    )}

## Candidate Loop Results

{markdown_table(
        (
            "candidate",
            "trainable",
            "encoder",
            "decoder",
            "predictor",
            "VRAM",
            "score",
            "checks",
        ),
        top_rows,
    )}

## Verification Checks For Recommended Candidate

{markdown_table(("check", "status", "detail"), check_rows)}

## Recommended Training Shape

Train it as an encoder-decoder model with JEPA as an auxiliary representation objective, not as a pure JEPA model.

1. Build a normal encoder-decoder Transformer with tied token embeddings.
2. Keep a standard token-level denoising or seq2seq cross-entropy loss.
3. Add an EMA target encoder. The target encoder sees the unmasked source sequence.
4. The student encoder sees a masked/corrupted source sequence.
5. A compact predictor receives student encoder states plus mask-position queries and predicts the EMA target encoder states for masked source spans.
6. Normalize target states and compute latent loss on masked positions only.
7. Optimize `total_loss = CE + lambda_jepa * latent_loss`, with `lambda_jepa` ramped from 0 to about {c.jepa_loss_weight} over early training.
8. Update the target encoder with EMA after each optimizer step.

Pseudo-code:

```python
for batch in train_loader:
    source, labels = batch["source_ids"], batch["labels"]
    masked_source, masked_positions = span_mask(source)

    with torch.no_grad():
        target_states = ema_encoder(source, output_hidden_states=True)
        target = normalize(mean_top_k(target_states, k={c.top_k_target_layers}))
        target = gather_positions(target, masked_positions)

    student_states = model.encoder(masked_source)
    pred = predictor(student_states, masked_positions)
    latent_loss = smooth_l1_or_mse(normalize(pred), target)

    decoder_logits = model.decoder(labels[:, :-1], encoder_states=student_states)
    ce_loss = cross_entropy(decoder_logits, labels[:, 1:])

    loss = ce_loss + lambda_jepa * latent_loss
    loss.backward()
    optimizer.step()
    update_ema(ema_encoder, model.encoder)
```

## Why This Architecture Is Worth Training

- **Small models need representation help.** A 0.2B encoder-decoder has limited capacity. Latent prediction pressures the encoder to learn contextual, semantic states instead of spending all learning signal on next-token surface form.
- **The decoder still needs token supervision.** JEPA predicts hidden representations, not text. Keeping CE is what makes the model useful for generation, translation, summarization, or instruction-style seq2seq work.
- **EMA targets stabilize the objective.** The predictor learns against a slowly moving teacher, which reduces representation collapse and avoids the student chasing its own current noise.
- **A small predictor is the right bottleneck.** The predictor absorbs the latent prediction task so the encoder does not have to contort itself into directly reconstructing target states.
- **Encoder-only EMA keeps this plausible at 0.2B.** A full encoder-decoder teacher would be wasteful. Copying only the encoder keeps the extra storage meaningful but not absurd.
- **The latent objective is cheaper than reconstruction-heavy alternatives.** You predict normalized hidden states for masked spans instead of reconstructing every token/pixel/detail.

## What Would Make It Unwise

- Training with only JEPA loss while expecting text generation.
- Using a large predictor that becomes a second model.
- Copying the full encoder-decoder as the EMA target.
- Training on too little text; the CE-only baseline may simply be easier and more reliable.
- No collapse monitoring. A falling latent loss is not enough if representation variance also collapses.

## Empirical Verification Plan

Run a short ablation before full training:

1. CE-only baseline for the same tokens, batch size, and optimizer.
2. CE + JEPA with `lambda_jepa` ramped to 0.1.
3. CE + JEPA with `lambda_jepa` ramped to 0.25.

Track these gates:

- Validation CE/perplexity should not regress by more than 1-2 percent after warmup.
- JEPA loss should fall while target representation variance stays non-zero.
- Mean pairwise cosine similarity between unrelated examples should not approach 1.
- Linear probe, retrieval, or classification performance on encoder states should improve over CE-only.
- Downstream seq2seq metric should match or beat CE-only after equal compute.

## Assumptions

- Vocabulary size: {args.vocab_size}
- Source length: {args.source_len}
- Target length: {args.target_len}
- Micro-batch size: {args.micro_batch}
- VRAM budget: {args.vram_gb:.1f} GB
- Gradient checkpointing: {args.gradient_checkpointing}
- Parameter target: {format_params(args.target_params)}
- Target tolerance: {args.tolerance:.0%}

The VRAM estimate is intentionally rough. Treat it as an early warning system, not a replacement for a real profiler.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a 0.2B encoder-decoder JEPA training recipe and generate a reasoning report."
    )
    parser.add_argument("--target-params", type=int, default=200_000_000)
    parser.add_argument("--tolerance", type=float, default=0.12)
    parser.add_argument("--vocab-size", type=int, default=32_128)
    parser.add_argument("--source-len", type=int, default=512)
    parser.add_argument("--target-len", type=int, default=256)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--vram-gb", type=float, default=24.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/jepa_verification_report.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = evaluate_candidates(args)
    if not reports:
        raise SystemExit("No candidates were generated.")

    report = make_report(args, reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    best = reports[0]
    print(f"Best candidate: {best.candidate.name}")
    print(f"Trainable params: {format_params(best.params.trainable_params)}")
    print(f"Estimated peak VRAM: {best.memory.peak_gb:.1f} GB")
    print(f"Checks: {status_counts(best.checks)}")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
