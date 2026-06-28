"""Span masking helpers for encoder-side JEPA and T5-style denoising."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskedBatch:
    """Masked source tokens and the positions used for JEPA latent prediction."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    masked_positions: torch.Tensor
    masked_position_mask: torch.Tensor


def span_mask_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int,
    pad_token_id: int,
    mask_fraction: float = 0.15,
    mean_span_length: int = 3,
    max_predictions: int | None = None,
) -> MaskedBatch:
    """Mask contiguous spans and return padded masked-position indices.

    The returned `masked_positions` are source positions only. They are the sole
    positions used for JEPA target gathering, which keeps latent prediction
    encoder-only and prevents decoder outputs from feeding back into JEPA.
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, seq]")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")

    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    masked = input_ids.clone()
    valid_mask = attention_mask.bool() & input_ids.ne(pad_token_id)
    all_positions: list[torch.Tensor] = []

    for row in range(batch_size):
        valid_positions = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            all_positions.append(torch.empty(0, dtype=torch.long, device=device))
            continue

        target_count = max(1, int(round(valid_positions.numel() * mask_fraction)))
        if max_predictions is not None:
            target_count = min(target_count, max_predictions)

        chosen: list[int] = []
        chosen_set: set[int] = set()
        attempts = 0
        while len(chosen) < target_count and attempts < target_count * 8:
            attempts += 1
            start_idx = int(torch.randint(0, valid_positions.numel(), (1,), device=device).item())
            start = int(valid_positions[start_idx].item())
            span_len = max(1, int(torch.poisson(torch.tensor(float(mean_span_length))).item()))
            for pos in range(start, min(seq_len, start + span_len)):
                if valid_mask[row, pos] and pos not in chosen_set:
                    chosen.append(pos)
                    chosen_set.add(pos)
                if len(chosen) >= target_count:
                    break

        if not chosen:
            chosen = [int(valid_positions[0].item())]

        row_positions = torch.tensor(sorted(chosen[:target_count]), dtype=torch.long, device=device)
        masked[row, row_positions] = mask_token_id
        all_positions.append(row_positions)

    width = max(1, max(pos.numel() for pos in all_positions))
    masked_positions = torch.zeros(batch_size, width, dtype=torch.long, device=device)
    masked_position_mask = torch.zeros(batch_size, width, dtype=torch.bool, device=device)
    for row, positions in enumerate(all_positions):
        if positions.numel() == 0:
            continue
        count = min(width, positions.numel())
        masked_positions[row, :count] = positions[:count]
        masked_position_mask[row, :count] = True

    return MaskedBatch(
        input_ids=masked,
        attention_mask=attention_mask,
        masked_positions=masked_positions,
        masked_position_mask=masked_position_mask,
    )


def gather_positions(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather hidden states at `[batch, num_positions]` source indices."""

    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, seq, dim]")
    if positions.ndim != 2:
        raise ValueError("positions must have shape [batch, num_positions]")
    gather_index = positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
    return hidden.gather(dim=1, index=gather_index)
