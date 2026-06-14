"""Configuration dataclasses for JEPA-augmented encoder-decoder experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelShape:
    """Transformer dimensions for a small encoder-decoder model."""

    d_model: int = 768
    encoder_layers: int = 10
    decoder_layers: int = 10
    d_ff: int = 3072
    attention_heads: int = 12
    vocab_size: int = 32_000
    predictor_width: int = 512
    predictor_layers: int = 3


@dataclass(frozen=True)
class TrainingObjective:
    """High-level objective weights and schedules."""

    ce_weight: float = 1.0
    jepa_weight_peak: float = 0.25
    jepa_warmup_fraction: float = 0.1
    ema_tau_start: float = 0.99
    ema_tau_end: float = 0.9995
    top_k_target_layers: int = 4
