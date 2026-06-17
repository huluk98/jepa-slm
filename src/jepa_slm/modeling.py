"""PyTorch modules for JEPA-augmented T5-style encoder-decoder training."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import T5Config, T5ForConditionalGeneration

from .config import JepaSettings, ModelShape
from .masking import gather_positions


@dataclass(frozen=True)
class JepaBatch:
    """Batch contract consumed by `JepaEncoderDecoder`."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    decoder_input_ids: torch.Tensor
    masked_input_ids: torch.Tensor
    masked_attention_mask: torch.Tensor
    masked_positions: torch.Tensor
    masked_position_mask: torch.Tensor


@dataclass(frozen=True)
class JepaForwardOutput:
    """Loss values and detached diagnostics from one training step."""

    loss: torch.Tensor
    ce_loss: torch.Tensor
    jepa_loss: torch.Tensor
    jepa_weight: float
    logits: torch.Tensor
    encoder_state_variance: torch.Tensor
    predictor_state_variance: torch.Tensor


class JepaPredictor(nn.Module):
    """Small Transformer-style predictor over masked encoder states."""

    def __init__(self, d_model: int, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, width) if width != d_model else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=max(1, min(heads, width // 64 if width >= 64 else 1)),
            dim_feedforward=width * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.out_proj = nn.Linear(width, d_model) if width != d_model else nn.Identity()

    def forward(self, masked_states: torch.Tensor, position_mask: torch.Tensor) -> torch.Tensor:
        padding_mask = ~position_mask.bool()
        hidden = self.in_proj(masked_states)
        hidden = self.blocks(hidden, src_key_padding_mask=padding_mask)
        return self.out_proj(hidden)


def build_t5_config(shape: ModelShape, pad_token_id: int, eos_token_id: int) -> T5Config:
    """Build a T5 config from repository model dimensions."""

    gated = shape.ffn_activation.lower() in {"swiglu", "geglu"}
    feed_forward_proj = "gated-gelu" if gated else "gelu"
    return T5Config(
        vocab_size=shape.vocab_size,
        d_model=shape.d_model,
        d_ff=shape.d_ff,
        num_layers=shape.encoder_layers,
        num_decoder_layers=shape.decoder_layers,
        num_heads=shape.attention_heads,
        d_kv=max(1, shape.d_model // shape.attention_heads),
        dropout_rate=0.0,
        feed_forward_proj=feed_forward_proj,
        tie_word_embeddings=True,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        decoder_start_token_id=pad_token_id,
        relative_attention_num_buckets=32,
    )


class JepaEncoderDecoder(nn.Module):
    """T5 encoder-decoder with an encoder-only JEPA auxiliary objective."""

    def __init__(
        self,
        shape: ModelShape,
        jepa: JepaSettings,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
    ) -> None:
        super().__init__()
        if jepa.target_encoder != "ema_encoder_only":
            raise ValueError("Only encoder-only EMA targets are supported.")

        self.shape = shape
        self.jepa = jepa
        self.config = build_t5_config(shape, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
        self.model = T5ForConditionalGeneration(self.config)
        self.ema_encoder = copy.deepcopy(self.model.encoder)
        self.ema_encoder.requires_grad_(False)
        self.ema_encoder.eval()
        self.predictor = JepaPredictor(
            d_model=shape.d_model,
            width=jepa.predictor_width,
            layers=jepa.predictor_layers,
            heads=shape.attention_heads,
        )

    @property
    def encoder(self) -> nn.Module:
        return self.model.encoder

    @property
    def decoder(self) -> nn.Module:
        return self.model.decoder

    def gradient_checkpointing_enable(self) -> None:
        self.model.gradient_checkpointing_enable()

    @torch.no_grad()
    def update_ema_encoder(self, tau: float) -> None:
        """Update target encoder from the student encoder only."""

        source = dict(self.model.encoder.named_parameters())
        for name, target_param in self.ema_encoder.named_parameters():
            target_param.data.mul_(tau).add_(source[name].data, alpha=1.0 - tau)

        source_buffers = dict(self.model.encoder.named_buffers())
        for name, target_buffer in self.ema_encoder.named_buffers():
            if name in source_buffers and target_buffer.dtype.is_floating_point:
                target_buffer.data.copy_(source_buffers[name].data)

    def _target_states(self, batch: JepaBatch) -> torch.Tensor:
        with torch.no_grad():
            output = self.ema_encoder(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = output.hidden_states[-self.jepa.top_k_target_layers :]
            target = torch.stack(hidden_states, dim=0).mean(dim=0)
            if self.jepa.normalize_targets:
                target = F.layer_norm(target, (target.size(-1),))
            target = gather_positions(target, batch.masked_positions)
        return target.detach()

    def forward(self, batch: JepaBatch, jepa_weight: float) -> JepaForwardOutput:
        """Compute CE plus encoder-only JEPA loss.

        Decoder logits consume student encoder states through the normal T5
        forward path. No decoded tokens are ever used to construct the JEPA
        target, which is the architectural contract this wrapper enforces.
        """

        target = self._target_states(batch)

        encoder_output = self.model.encoder(
            input_ids=batch.masked_input_ids,
            attention_mask=batch.masked_attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        student_states = encoder_output.last_hidden_state
        masked_student = gather_positions(student_states, batch.masked_positions)
        prediction = self.predictor(masked_student, batch.masked_position_mask)

        if self.jepa.normalize_targets:
            prediction_for_loss = F.layer_norm(prediction, (prediction.size(-1),))
        else:
            prediction_for_loss = prediction

        per_token_loss = F.smooth_l1_loss(prediction_for_loss, target, reduction="none").mean(dim=-1)
        valid = batch.masked_position_mask.float()
        jepa_loss = (per_token_loss * valid).sum() / valid.sum().clamp_min(1.0)

        decoder_outputs = self.model(
            input_ids=batch.masked_input_ids,
            attention_mask=batch.masked_attention_mask,
            decoder_input_ids=batch.decoder_input_ids,
            labels=batch.labels,
            encoder_outputs=encoder_output,
            return_dict=True,
        )
        ce_loss = decoder_outputs.loss
        loss = ce_loss + float(jepa_weight) * jepa_loss

        return JepaForwardOutput(
            loss=loss,
            ce_loss=ce_loss,
            jepa_loss=jepa_loss,
            jepa_weight=float(jepa_weight),
            logits=decoder_outputs.logits,
            encoder_state_variance=student_states.detach().float().var(),
            predictor_state_variance=prediction.detach().float().var(),
        )


def assert_encoder_only_jepa_contract(model: JepaEncoderDecoder) -> None:
    """Raise if the architecture drifts away from encoder-only JEPA."""

    if model.jepa.target_encoder != "ema_encoder_only":
        raise AssertionError("JEPA target must be encoder-only.")
    if any(param.requires_grad for param in model.ema_encoder.parameters()):
        raise AssertionError("EMA target encoder must not receive gradients.")
    decoder_param_ids = {id(param) for param in model.model.decoder.parameters()}
    ema_param_ids = {id(param) for param in model.ema_encoder.parameters()}
    if decoder_param_ids & ema_param_ids:
        raise AssertionError("EMA encoder must not share decoder parameters.")
