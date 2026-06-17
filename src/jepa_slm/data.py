"""Dataset and collator utilities for JEPA-SLM training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, T5Tokenizer

from .config import DataSettings
from .masking import span_mask_batch
from .modeling import JepaBatch


class TokenizerLike(Protocol):
    pad_token_id: int
    eos_token_id: int | None
    unk_token_id: int | None

    def __call__(
        self,
        texts: list[str],
        max_length: int,
        padding: str,
        truncation: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        ...


class ByteSmokeTokenizer:
    """Small deterministic tokenizer for offline smoke tests.

    This is not recommended for real training. It lets the training loop execute
    in CI or on a fresh machine before a SentencePiece tokenizer exists.
    """

    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2

    def __init__(self, vocab_size: int) -> None:
        if vocab_size < 259:
            raise ValueError("ByteSmokeTokenizer requires vocab_size >= 259.")
        self.vocab_size = vocab_size

    def __call__(
        self,
        texts: list[str],
        max_length: int,
        padding: str,
        truncation: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        if padding != "max_length" or return_tensors != "pt":
            raise ValueError("ByteSmokeTokenizer only supports max_length padding and pt tensors.")
        rows = []
        masks = []
        for text in texts:
            token_ids = [3 + byte for byte in text.encode("utf-8", errors="replace")]
            if truncation:
                token_ids = token_ids[: max(0, max_length - 1)]
            token_ids.append(self.eos_token_id)
            token_ids = token_ids[:max_length]
            attention = [1] * len(token_ids)
            while len(token_ids) < max_length:
                token_ids.append(self.pad_token_id)
                attention.append(0)
            rows.append(token_ids)
            masks.append(attention)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def load_tokenizer(settings: DataSettings, vocab_size: int | None = None) -> TokenizerLike:
    """Load a trained local tokenizer or a bootstrap T5 tokenizer."""

    if settings.tokenizer_name == "internal-byte":
        return ByteSmokeTokenizer(vocab_size or 32_128)
    if settings.tokenizer_path:
        tokenizer = T5Tokenizer(vocab_file=settings.tokenizer_path, extra_ids=100)
    else:
        tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_name, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class TextStreamDataset(IterableDataset[dict[str, str]]):
    """Thin iterable wrapper around Hugging Face streaming datasets."""

    def __init__(self, settings: DataSettings) -> None:
        super().__init__()
        self.settings = settings

    def __iter__(self) -> Iterator[dict[str, str]]:
        if self.settings.dataset == "synthetic":
            samples = (
                "turn on the kitchen light",
                "set the living room thermostat to 21 degrees",
                "reject command because the garage fan does not exist",
                "turn off the bedroom lamp and close the blinds",
            )
            count = 0
            while self.settings.max_samples is None or count < self.settings.max_samples:
                yield {"text": samples[count % len(samples)]}
                count += 1
            return

        from datasets import load_dataset

        dataset = load_dataset(
            self.settings.dataset,
            name=self.settings.subset,
            split=self.settings.split,
            streaming=self.settings.streaming,
        )
        count = 0
        for row in dataset:
            text = row.get(self.settings.text_field)
            if not text:
                continue
            yield {"text": str(text)}
            count += 1
            if self.settings.max_samples is not None and count >= self.settings.max_samples:
                break


@dataclass
class JepaCollator:
    """Tokenize text and create masked source batches."""

    tokenizer: TokenizerLike
    source_length: int
    target_length: int
    mask_fraction: float = 0.15
    mean_span_length: int = 3

    def __call__(self, rows: list[dict[str, str]]) -> JepaBatch:
        texts = [row["text"] for row in rows]
        source = self.tokenizer(
            texts,
            max_length=self.source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer(
            texts,
            max_length=self.target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        decoder_input_ids = _shift_right(
            labels,
            decoder_start_token_id=self.tokenizer.pad_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        mask_token_id = self.tokenizer.unk_token_id
        if mask_token_id is None:
            mask_token_id = self.tokenizer.pad_token_id
        masked = span_mask_batch(
            source["input_ids"],
            source["attention_mask"],
            mask_token_id=mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            mask_fraction=self.mask_fraction,
            mean_span_length=self.mean_span_length,
        )

        return JepaBatch(
            input_ids=source["input_ids"],
            attention_mask=source["attention_mask"],
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            masked_input_ids=masked.input_ids,
            masked_attention_mask=masked.attention_mask,
            masked_positions=masked.masked_positions,
            masked_position_mask=masked.masked_position_mask,
        )


def _shift_right(
    labels: torch.Tensor,
    decoder_start_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 0] = decoder_start_token_id
    shifted[:, 1:] = labels[:, :-1]
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


def move_batch_to_device(batch: JepaBatch, device: torch.device) -> JepaBatch:
    return JepaBatch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        labels=batch.labels.to(device),
        decoder_input_ids=batch.decoder_input_ids.to(device),
        masked_input_ids=batch.masked_input_ids.to(device),
        masked_attention_mask=batch.masked_attention_mask.to(device),
        masked_positions=batch.masked_positions.to(device),
        masked_position_mask=batch.masked_position_mask.to(device),
    )


def build_dataloader(
    settings: DataSettings,
    tokenizer: TokenizerLike,
    source_length: int,
    target_length: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    dataset: Iterable[dict[str, str]] = TextStreamDataset(settings)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=JepaCollator(tokenizer, source_length, target_length),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
