"""PyTorch modules for JEPA-augmented T5-style encoder-decoder training.

Design (best-of-both T5 + JEPA):

* **T5/UL2 backbone** keeps token-level cross entropy as the generative
  contract: the decoder reconstructs the target text from the (span-corrupted)
  student encoder states. Relative-position bias and tied embeddings are kept.
* **JEPA auxiliary loss** is encoder-only. An EMA *target* encoder reads the
  clean source and produces normalized top-k-layer latents; a small predictor
  reads the *full* student-encoder context (plus a learned mask marker at masked
  positions) and predicts those latents at the masked positions. Targets are
  stop-gradiented. This matches I-JEPA / data2vec where the predictor attends
  from visible context to masked targets, rather than only refining the masked
  slots in isolation.
* **Collapse safety**: EMA + affine-free target normalization (data2vec), plus
  an optional VICReg variance/covariance penalty and a per-dimension std
  monitor so a falling latent loss cannot hide representational collapse.
"""

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
    vicreg_loss: torch.Tensor
    jepa_weight: float
    logits: torch.Tensor
    encoder_repr_std: torch.Tensor
    predictor_repr_std: torch.Tensor


class JepaPredictor(nn.Module):
    """Small Transformer predictor over student encoder states.

    In full-context mode it consumes the entire student sequence with a learned
    marker added at masked positions, so its self-attention can read visible
    context to reconstruct masked-target latents. The parent then gathers/weights
    predictions at masked positions for the loss.
    """

    def __init__(self, d_model: int, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, width) if width != d_model else nn.Identity()
        self.mask_marker = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.mask_marker, std=0.02)
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

    def forward(
        self,
        states: torch.Tensor,
        key_padding_mask: torch.Tensor,
        marker_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict latents.

        Args:
            states: ``[batch, seq, d_model]`` student encoder states.
            key_padding_mask: ``[batch, seq]`` bool, ``True`` where ignored (pad).
            marker_mask: ``[batch, seq]`` bool/float, where to add the mask marker.
        """
        hidden = self.in_proj(states)
        hidden = hidden + marker_mask.unsqueeze(-1).to(hidden.dtype) * self.mask_marker
        hidden = self.blocks(hidden, src_key_padding_mask=key_padding_mask.bool())
        return self.out_proj(hidden)


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """Return the off-diagonal elements of a square matrix as a 1-D tensor."""
    n, m = matrix.shape
    assert n == m
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def build_t5_config(shape: ModelShape, pad_token_id: int, eos_token_id: int) -> T5Config:
    """Build a T5 config from repository model dimensions."""

    activation = shape.ffn_activation.lower()
    if activation in {"swiglu", "geglu", "gated-gelu", "gated-silu"}:
        # T5 exposes gated GELU (GeGLU). True SwiGLU (gated SiLU) is not a native
        # T5 feed_forward_proj; fall back to gated-gelu and surface the choice.
        if activation in {"swiglu", "gated-silu"}:
            print(
                "[jepa-slm] note: T5 has no native SwiGLU; using gated-gelu (GeGLU) instead.",
                flush=True,
            )
        feed_forward_proj = "gated-gelu"
    else:
        feed_forward_proj = "gelu" if activation == "gelu" else activation
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
        attn_implementation: str | None = None,
    ) -> None:
        super().__init__()
        if jepa.target_encoder != "ema_encoder_only":
            raise ValueError("Only encoder-only EMA targets are supported.")

        self.shape = shape
        self.jepa = jepa
        self.config = build_t5_config(shape, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
        self.model = _build_t5(self.config, attn_implementation)
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
        """EMA-teacher latents over the *clean* source, normalized, full length."""
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
        return target.detach()

    def _masked_token_map(self, batch: JepaBatch) -> torch.Tensor:
        """Build a ``[batch, seq]`` boolean map of masked source positions."""
        batch_size, seq_len = batch.masked_input_ids.shape
        is_masked = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=batch.masked_input_ids.device
        )
        valid = batch.masked_position_mask.bool()
        if valid.any():
            row_index = (
                torch.arange(batch_size, device=is_masked.device)
                .unsqueeze(1)
                .expand_as(batch.masked_positions)
            )
            is_masked[row_index[valid], batch.masked_positions[valid]] = True
        return is_masked

    def _vicreg(
        self, latents: torch.Tensor, weight_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shape-stable weighted VICReg variance + covariance over masked tokens.

        Returns ``(loss, per_dim_std_mean)``; the std is also used as the monitor.
        """
        d = latents.size(-1)
        w = weight_mask.to(latents.dtype).unsqueeze(-1)  # [B, L, 1]
        n = w.sum().clamp_min(1.0)
        flat = latents.reshape(-1, d)
        wflat = w.reshape(-1, 1)
        mean = (flat * wflat).sum(dim=0) / n
        centered = (flat - mean) * wflat  # zeros out non-masked rows
        denom = (n - 1.0).clamp_min(1.0)  # unbiased estimator (VICReg convention)
        var = (centered.pow(2)).sum(dim=0) / denom
        std = torch.sqrt(var + 1e-4)
        std_monitor = std.mean().detach()

        variance_loss = F.relu(self.jepa.vicreg_variance_gamma - std).mean()
        covariance_loss = latents.new_zeros(())
        if self.jepa.vicreg_covariance_weight > 0:
            cov = (centered.transpose(0, 1) @ centered) / denom
            covariance_loss = _off_diagonal(cov).pow(2).sum() / d
        vicreg_loss = (
            self.jepa.vicreg_variance_weight * variance_loss
            + self.jepa.vicreg_covariance_weight * covariance_loss
        )
        return vicreg_loss, std_monitor

    def forward(self, batch: JepaBatch, jepa_weight: float) -> JepaForwardOutput:
        """Compute CE plus encoder-only JEPA loss.

        Decoder logits consume student encoder states through the normal T5
        forward path. No decoded tokens are ever used to construct the JEPA
        target, which is the architectural contract this wrapper enforces.
        """

        target = self._target_states(batch)
        is_masked = self._masked_token_map(batch)

        encoder_output = self.model.encoder(
            input_ids=batch.masked_input_ids,
            attention_mask=batch.masked_attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        student_states = encoder_output.last_hidden_state

        if self.jepa.predictor_full_context:
            # Predict over the full contextualized sequence; score masked slots.
            prediction = self.predictor(
                student_states,
                key_padding_mask=batch.masked_attention_mask == 0,
                marker_mask=is_masked,
            )
            target_for_loss = target
            loss_weight = is_masked
            predictor_states = prediction
        else:
            # Legacy ablation: predictor sees only the gathered masked vectors.
            gathered_student = gather_positions(student_states, batch.masked_positions)
            prediction = self.predictor(
                gathered_student,
                key_padding_mask=batch.masked_position_mask == 0,
                marker_mask=batch.masked_position_mask,
            )
            target_for_loss = gather_positions(target, batch.masked_positions)
            loss_weight = batch.masked_position_mask
            predictor_states = prediction

        if self.jepa.normalize_targets:
            prediction_for_loss = F.layer_norm(prediction, (prediction.size(-1),))
        else:
            prediction_for_loss = prediction

        weight = loss_weight.to(student_states.dtype)
        denom = weight.sum().clamp_min(1.0)
        per_token_loss = F.smooth_l1_loss(
            prediction_for_loss, target_for_loss, reduction="none"
        ).mean(dim=-1)
        jepa_loss = (per_token_loss * weight).sum() / denom

        vicreg_loss, encoder_repr_std = self._vicreg(student_states, is_masked)
        predictor_repr_std = self._masked_std(predictor_states, loss_weight)

        decoder_outputs = self.model(
            input_ids=batch.masked_input_ids,
            attention_mask=batch.masked_attention_mask,
            decoder_input_ids=batch.decoder_input_ids,
            labels=batch.labels,
            encoder_outputs=encoder_output,
            return_dict=True,
        )
        ce_loss = decoder_outputs.loss
        loss = ce_loss + float(jepa_weight) * jepa_loss + vicreg_loss

        return JepaForwardOutput(
            loss=loss,
            ce_loss=ce_loss,
            jepa_loss=jepa_loss,
            vicreg_loss=vicreg_loss.detach(),
            jepa_weight=float(jepa_weight),
            logits=decoder_outputs.logits,
            encoder_repr_std=encoder_repr_std,
            predictor_repr_std=predictor_repr_std,
        )

    @staticmethod
    def _masked_std(latents: torch.Tensor, weight_mask: torch.Tensor) -> torch.Tensor:
        """Mean over dims of the per-dimension std across masked tokens (detached)."""
        with torch.no_grad():
            d = latents.size(-1)
            w = weight_mask.to(latents.dtype).reshape(-1, 1)
            n = w.sum().clamp_min(1.0)
            flat = latents.reshape(-1, d)
            mean = (flat * w).sum(dim=0) / n
            var = (((flat - mean) * w).pow(2)).sum(dim=0) / n.clamp_min(2.0)
            return torch.sqrt(var + 1e-4).mean()


def _build_t5(config: T5Config, attn_implementation: str | None) -> T5ForConditionalGeneration:
    """Instantiate T5, trying the requested attention backend then falling back.

    T5's relative-position bias is incompatible with FlashAttention-2, so values
    like ``flash_attention_2_if_available`` resolve to ``sdpa`` when supported
    and ``eager`` otherwise.
    """
    requested = (attn_implementation or "auto").lower()
    if requested in {"flash_attention_2_if_available", "flash_attention_2", "auto", "sdpa"}:
        candidates = ("sdpa", "eager")
    elif requested == "eager":
        candidates = ("eager",)
    else:
        candidates = (requested, "eager")

    for impl in candidates:
        try:
            return T5ForConditionalGeneration._from_config(config, attn_implementation=impl)
        except Exception:  # noqa: BLE001 - fall back across transformers versions
            try:
                config._attn_implementation = impl  # type: ignore[attr-defined]
                return T5ForConditionalGeneration(config)
            except Exception:  # noqa: BLE001
                continue
    return T5ForConditionalGeneration(config)


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
