"""Dataset and collator utilities for JEPA-SLM training."""

from __future__ import annotations

import glob
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer, T5Tokenizer

from .config import DataSettings
from .masking import span_mask_batch
from .modeling import JepaBatch
from .text_cleaning import clean_text

__all__ = [
    "ByteSmokeTokenizer",
    "JepaCollator",
    "PackedCollator",
    "PackedTokenDataset",
    "TextStreamDataset",
    "build_dataloader",
    "clean_text",
    "load_tokenizer",
    "move_batch_to_device",
]


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
    supports_dynamic_padding = False

    def __init__(self, vocab_size: int) -> None:
        if vocab_size < 259:
            raise ValueError("ByteSmokeTokenizer requires vocab_size >= 259.")
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [3 + byte for byte in text.encode("utf-8", errors="replace")]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def __call__(
        self,
        texts: list[str],
        max_length: int,
        padding: str,
        truncation: bool,
        return_tensors: str,
        pad_to_multiple_of: int | None = None,
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
    use_fast = getattr(settings, "tokenizer_use_fast", True)
    if settings.tokenizer_path:
        # T5Tokenizer is the slow SentencePiece tokenizer; prefer the fast
        # variant when available for a large throughput win.
        if use_fast:
            try:
                from transformers import T5TokenizerFast

                tokenizer = T5TokenizerFast(vocab_file=settings.tokenizer_path, extra_ids=100)
            except Exception:  # noqa: BLE001 - fall back to the slow tokenizer
                tokenizer = T5Tokenizer(vocab_file=settings.tokenizer_path, extra_ids=100)
        else:
            tokenizer = T5Tokenizer(vocab_file=settings.tokenizer_path, extra_ids=100)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_name, use_fast=use_fast)
        except Exception:  # noqa: BLE001 - some repos only ship a slow tokenizer
            tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_name, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class TextStreamDataset(IterableDataset[dict[str, str]]):
    """Thin iterable wrapper around Hugging Face streaming datasets.

    Shards the stream across DDP ranks and DataLoader workers so each consumes a
    disjoint slice. ``settings.skip_examples`` skips the head of each shard, used
    on resume so a restarted streaming run does not replay early data.
    """

    def __init__(self, settings: DataSettings, rank: int = 0, world_size: int = 1) -> None:
        super().__init__()
        self.settings = settings
        self.rank = max(0, int(rank))
        self.world_size = max(1, int(world_size))

    def _shard_and_clean(
        self, raw: Iterable[object], node_sharded: bool
    ) -> Iterator[dict[str, str]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        if node_sharded:
            # The HF stream is already split by node; only shard across workers.
            shard_index, num_shards = worker_id, num_workers
        else:
            shard_index = self.rank * num_workers + worker_id
            num_shards = self.world_size * num_workers

        # skip_examples is a PER-RANK count. Each worker runs __iter__ in its own
        # process, so split the skip across this rank's workers; the union of
        # per-worker skips then equals (within < num_workers) the per-rank skip.
        # Without this, every worker would skip the full count -> N x over-skip.
        rank_skip = max(0, int(self.settings.skip_examples))
        skip = rank_skip // num_workers
        if worker_id < rank_skip % num_workers:
            skip += 1
        max_samples = self.settings.max_samples
        shard_pos = 0
        emitted = 0
        for index, value in enumerate(raw):
            if num_shards > 1 and index % num_shards != shard_index:
                continue
            text = clean_text(
                value,
                normalize=self.settings.normalize_text,
                min_chars=self.settings.min_chars,
                max_chars=self.settings.max_chars,
            )
            if text is None:
                continue
            if shard_pos < skip:
                shard_pos += 1
                continue
            shard_pos += 1
            yield {"text": text}
            emitted += 1
            if max_samples is not None and emitted >= max_samples:
                return

    def _synthetic_raw(self) -> Iterator[str]:
        samples = (
            "turn on the kitchen light",
            "set the living room thermostat to 21 degrees",
            "reject command because the garage fan does not exist",
            "turn off the bedroom lamp and close the blinds",
        )
        count = 0
        while True:
            yield samples[count % len(samples)]
            count += 1

    def __iter__(self) -> Iterator[dict[str, str]]:
        if self.settings.dataset == "synthetic":
            yield from self._shard_and_clean(self._synthetic_raw(), node_sharded=False)
            return

        if _is_local_dataset(self.settings.dataset):
            raw = _iter_local_texts(self.settings.dataset, self.settings.text_field)
            yield from self._shard_and_clean(raw, node_sharded=False)
            return

        from datasets import load_dataset

        dataset = load_dataset(
            self.settings.dataset,
            name=self.settings.subset,
            split=self.settings.split,
            streaming=self.settings.streaming,
        )
        node_sharded = False
        if self.world_size > 1:
            try:
                from datasets.distributed import split_dataset_by_node

                dataset = split_dataset_by_node(dataset, rank=self.rank, world_size=self.world_size)
                node_sharded = True
            except Exception:  # noqa: BLE001 - fall back to index-stride sharding
                node_sharded = False

        raw = (row.get(self.settings.text_field) for row in dataset)
        yield from self._shard_and_clean(raw, node_sharded=node_sharded)


def _is_local_dataset(dataset: str) -> bool:
    return any(char in dataset for char in "*?[") or Path(dataset).exists()


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_local_texts(dataset: str, text_field: str) -> Iterator[str]:
    paths = [Path(path) for path in sorted(glob.glob(dataset))] or [Path(dataset)]
    for path in paths:
        with _open_text(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if path.suffix == ".jsonl" or path.name.endswith(".jsonl.gz"):
                    value = json.loads(line).get(text_field)
                    if value:
                        yield str(value)
                else:
                    yield line


@dataclass
class JepaCollator:
    """Tokenize text and create masked source batches."""

    tokenizer: TokenizerLike
    source_length: int
    target_length: int
    mask_fraction: float = 0.15
    mean_span_length: int = 3
    dynamic_padding: bool = True
    pad_to_multiple_of: int = 8

    def _tokenize(self, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
        use_dynamic = self.dynamic_padding and getattr(
            self.tokenizer, "supports_dynamic_padding", True
        )
        if use_dynamic:
            return self.tokenizer(
                texts,
                max_length=max_length,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __call__(self, rows: list[dict[str, str]]) -> JepaBatch:
        texts = [row["text"] for row in rows]
        source = self._tokenize(texts, self.source_length)
        target = self._tokenize(texts, self.target_length)
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


class PackedTokenDataset(IterableDataset[dict[str, list]]):
    """Greedy token-level packing over the (sharded) text stream.

    Wraps :class:`TextStreamDataset` so sharding/skip behavior is inherited, then
    concatenates document token ids (EOS-separated) and emits fixed-length
    ``source_length`` blocks. Eliminates padding waste and yields fully static
    shapes (compile-friendly). The trailing partial block is dropped.
    """

    def __init__(
        self,
        settings: DataSettings,
        tokenizer: TokenizerLike,
        source_length: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.text_dataset = TextStreamDataset(settings, rank=rank, world_size=world_size)
        self.tokenizer = tokenizer
        self.source_length = source_length
        eos = getattr(tokenizer, "eos_token_id", None)
        self.separator_id = eos if eos is not None else tokenizer.pad_token_id

    def __iter__(self) -> Iterator[dict[str, list]]:
        buffer: list[int] = []
        for row in self.text_dataset:
            ids = self.tokenizer.encode(row["text"], add_special_tokens=False)
            if not ids:
                continue
            buffer.extend(ids)
            buffer.append(self.separator_id)
            while len(buffer) >= self.source_length:
                block = buffer[: self.source_length]
                del buffer[: self.source_length]
                yield {"input_ids": block}


@dataclass
class PackedCollator:
    """Build a JepaBatch from pre-packed fixed-length token blocks."""

    tokenizer: TokenizerLike
    source_length: int
    target_length: int
    mask_fraction: float = 0.15
    mean_span_length: int = 3

    def __call__(self, rows: list[dict[str, list]]) -> JepaBatch:
        input_ids = torch.tensor([row["input_ids"] for row in rows], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids[:, : self.target_length].clone()
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
            input_ids,
            attention_mask,
            mask_token_id=mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            mask_fraction=self.mask_fraction,
            mean_span_length=self.mean_span_length,
        )
        return JepaBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            masked_input_ids=masked.input_ids,
            masked_attention_mask=masked.attention_mask,
            masked_positions=masked.masked_positions,
            masked_position_mask=masked.masked_position_mask,
        )


def move_batch_to_device(
    batch: JepaBatch, device: torch.device, non_blocking: bool = False
) -> JepaBatch:
    def to(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device, non_blocking=non_blocking)

    return JepaBatch(
        input_ids=to(batch.input_ids),
        attention_mask=to(batch.attention_mask),
        labels=to(batch.labels),
        decoder_input_ids=to(batch.decoder_input_ids),
        masked_input_ids=to(batch.masked_input_ids),
        masked_attention_mask=to(batch.masked_attention_mask),
        masked_positions=to(batch.masked_positions),
        masked_position_mask=to(batch.masked_position_mask),
    )


def build_dataloader(
    settings: DataSettings,
    tokenizer: TokenizerLike,
    source_length: int,
    target_length: int,
    batch_size: int,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    pin_memory: bool | None = None,
    dynamic_padding: bool = True,
    pad_to_multiple_of: int = 8,
    sequence_packing: bool = False,
) -> DataLoader:
    dataset: Iterable[object]
    collate_fn: object
    if sequence_packing:
        dataset = PackedTokenDataset(
            settings, tokenizer, source_length, rank=rank, world_size=world_size
        )
        collate_fn = PackedCollator(tokenizer, source_length, target_length)
    else:
        dataset = TextStreamDataset(settings, rank=rank, world_size=world_size)
        collate_fn = JepaCollator(
            tokenizer,
            source_length,
            target_length,
            dynamic_padding=dynamic_padding,
            pad_to_multiple_of=pad_to_multiple_of,
        )
    if pin_memory is None:
        pin_memory = True
    # Pinned host memory only helps CUDA H2D transfers; avoid the MPS/CPU warning.
    pin_memory = bool(pin_memory) and torch.cuda.is_available()
    loader_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)
