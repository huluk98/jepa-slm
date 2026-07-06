"""End-to-end smoke + sharding tests.

These guard the regressions surfaced in the optimization audit:
* the training loop must complete a *logged* step (the step-0 logging crash),
* checkpoint/resume must round-trip, and
* the streaming dataset must shard disjointly across DDP ranks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from jepa_slm.config import (
    BatchingSettings,
    DataSettings,
    JepaSettings,
    ModelShape,
    OptimizerSettings,
    RuntimeSettings,
    TrainingConfig,
)
from jepa_slm.data import ByteSmokeTokenizer, PackedTokenDataset, TextStreamDataset
from jepa_slm.trainer import loss_is_bad, train


def _smoke_config(
    tmp_path: Path,
    max_steps: int = 3,
    resume_from: str | None = None,
    max_loss: float = 0.0,
    divergence_patience: int = 5,
    sequence_packing: bool = False,
    eval_every_steps: int = 0,
    eval_dataset: str | None = None,
) -> TrainingConfig:
    return TrainingConfig(
        model=ModelShape(
            d_model=32,
            encoder_layers=2,
            decoder_layers=2,
            d_ff=64,
            attention_heads=4,
            vocab_size=320,
            predictor_width=32,
            predictor_layers=1,
        ),
        jepa=JepaSettings(
            predictor_width=32,
            predictor_layers=1,
            top_k_target_layers=2,
            lambda_peak=0.25,
            lambda_warmup_fraction=0.3,
            vicreg_variance_weight=0.04,
            vicreg_covariance_weight=0.01,
        ),
        data=DataSettings(
            dataset="synthetic",
            tokenizer_name="internal-byte",
            max_samples=200,
            eval_dataset=eval_dataset,
        ),
        batching=BatchingSettings(
            source_length=24,
            target_length=24,
            per_gpu_micro_batch_sequences=2,
            gradient_accumulation_steps=2,
            sequence_packing=sequence_packing,
        ),
        optimizer=OptimizerSettings(learning_rate=5e-4, weight_decay=0.01, warmup_fraction=0.3),
        runtime=RuntimeSettings(
            output_dir=str(tmp_path / "smoke"),
            max_steps=max_steps,
            save_every_steps=max_steps,
            log_every_steps=1,  # forces the step-0 logging path that used to crash
            precision="fp32",
            compile=False,
            gradient_checkpointing=False,
            seed=7,
            resume_from=resume_from,
            max_loss=max_loss,
            divergence_patience=divergence_patience,
            eval_every_steps=eval_every_steps,
            eval_max_batches=2,
        ),
    )


def test_training_runs_past_first_log_step_and_checkpoints(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, max_steps=3)
    train(config)  # must not raise (regression guard for the step-0 log crash)

    final = tmp_path / "smoke" / "step-00000003" / "trainer_state.pt"
    assert final.exists()
    state = torch.load(final, map_location="cpu", weights_only=False)
    assert state["step"] == 3
    assert "rng" in state
    assert state["samples_consumed"] > 0


def test_param_storage_dtype_resolves() -> None:
    from dataclasses import replace

    from jepa_slm.trainer import param_storage_dtype

    base = _smoke_config(Path("."))
    assert param_storage_dtype(base) is None  # fp32 default -> keep fp32 master
    bf16 = replace(base, runtime=replace(base.runtime, param_dtype="bf16"))
    assert param_storage_dtype(bf16) is torch.bfloat16


def test_training_runs_with_bf16_param_storage(tmp_path: Path) -> None:
    from dataclasses import replace

    config = _smoke_config(tmp_path, max_steps=2)
    config = replace(
        config,
        runtime=replace(config.runtime, param_dtype="bf16", precision="bf16"),
    )
    train(config)  # bf16 weight/optimizer storage must train end-to-end without error
    assert (tmp_path / "smoke" / "step-00000002" / "trainer_state.pt").exists()


def test_training_resumes_from_checkpoint(tmp_path: Path) -> None:
    train(_smoke_config(tmp_path, max_steps=2))
    resume = str(tmp_path / "smoke" / "step-00000002")
    # Resuming must restore the step and run to completion without error.
    train(_smoke_config(tmp_path, max_steps=4, resume_from=resume))
    assert (tmp_path / "smoke" / "step-00000004" / "trainer_state.pt").exists()


def _write_distinct_shard(tmp_path: Path, n: int = 24) -> str:
    import json

    shard = tmp_path / "clean-00000.jsonl"
    shard.write_text(
        "\n".join(json.dumps({"text": f"doc number {i}"}) for i in range(n)) + "\n",
        encoding="utf-8",
    )
    return str(tmp_path / "clean-*.jsonl")


def test_packed_dataset_emits_fixed_length_blocks() -> None:
    settings = DataSettings(dataset="synthetic", max_samples=50, normalize_text=True)
    tok = ByteSmokeTokenizer(320)
    blocks = list(PackedTokenDataset(settings, tok, source_length=16))

    assert len(blocks) > 1
    # Every emitted block is exactly source_length and carries no padding.
    assert all(len(b["input_ids"]) == 16 for b in blocks)
    assert all(tok.pad_token_id not in b["input_ids"] for b in blocks)


def test_training_runs_with_sequence_packing(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, max_steps=3, sequence_packing=True)
    train(config)  # packed path must train end-to-end without error
    assert (tmp_path / "smoke" / "step-00000003" / "trainer_state.pt").exists()


def test_training_logs_throughput(tmp_path: Path, capsys) -> None:
    train(_smoke_config(tmp_path, max_steps=2))
    out = capsys.readouterr().out
    assert "'tok_per_s'" in out  # tokens/sec is reported every log step


def test_eval_ce_runs_when_eval_dataset_set(tmp_path: Path, capsys) -> None:
    config = _smoke_config(
        tmp_path, max_steps=2, eval_every_steps=1, eval_dataset="synthetic"
    )
    train(config)
    out = capsys.readouterr().out
    assert "'eval_ce'" in out  # periodic validation-CE is logged


def test_loss_is_bad_detects_nonfinite_and_threshold() -> None:
    assert loss_is_bad(float("nan"), 0.0) is True
    assert loss_is_bad(float("inf"), 0.0) is True
    assert loss_is_bad(5.0, 0.0) is False  # no threshold -> only non-finite is bad
    assert loss_is_bad(5.0, 10.0) is False
    assert loss_is_bad(12.0, 10.0) is True  # over threshold


def test_divergence_guard_stops_training(tmp_path: Path, capsys) -> None:
    # max_loss far below the real loss -> every step is "bad"; with patience 2 the
    # run must stop after 2 consecutive bad steps and save a last-good checkpoint.
    config = _smoke_config(tmp_path, max_steps=20, max_loss=0.0001, divergence_patience=2)
    train(config)

    out = capsys.readouterr().out
    assert "stop loss triggered" in out
    # No optimizer step ever succeeds, so it stops at step 0 (not max_steps=20).
    assert (tmp_path / "smoke" / "step-00000000" / "trainer_state.pt").exists()


def test_stream_dataset_shards_disjointly_and_completely(tmp_path: Path) -> None:
    glob = _write_distinct_shard(tmp_path, 24)
    settings = DataSettings(dataset=glob, normalize_text=True)
    rank0 = [r["text"] for r in TextStreamDataset(settings, rank=0, world_size=2)]
    rank1 = [r["text"] for r in TextStreamDataset(settings, rank=1, world_size=2)]

    # Disjoint shards whose union is the full corpus, with no duplicates.
    assert set(rank0).isdisjoint(rank1)
    assert sorted(rank0 + rank1) == sorted(f"doc number {i}" for i in range(24))
    assert len(rank0) == 12 and len(rank1) == 12


def test_skip_examples_skips_exactly_the_head(tmp_path: Path) -> None:
    glob = _write_distinct_shard(tmp_path, 20)
    settings = DataSettings(dataset=glob, normalize_text=True, skip_examples=5)
    got = [r["text"] for r in TextStreamDataset(settings, rank=0, world_size=1)]

    assert got == [f"doc number {i}" for i in range(5, 20)]


def test_skip_examples_is_split_across_workers(tmp_path: Path, monkeypatch) -> None:
    # Simulate a 2-worker DataLoader: each worker runs __iter__ independently.
    # The per-rank skip of 8 must split (4 + 4), NOT apply 8 in each worker.
    import jepa_slm.data as data_mod

    glob = _write_distinct_shard(tmp_path, 40)
    settings = DataSettings(dataset=glob, normalize_text=True, skip_examples=8)

    class FakeWorker:
        def __init__(self, wid: int) -> None:
            self.id = wid
            self.num_workers = 2

    totals = 0
    for wid in (0, 1):
        monkeypatch.setattr(data_mod, "get_worker_info", lambda w=wid: FakeWorker(w))
        totals += len(list(TextStreamDataset(settings, rank=0, world_size=1)))
    # World-size-1, 40 docs, skip 8 -> 32 should survive across both workers.
    assert totals == 32


class _FakeWorker:
    def __init__(self, wid: int, num_workers: int = 2) -> None:
        self.id = wid
        self.num_workers = num_workers


def test_worker_self_sharded_streams_are_not_strided_again(monkeypatch) -> None:
    # HF streaming datasets (datasets>=2.8) already split their shards across
    # DataLoader workers inside __iter__. When that stream is also node-split
    # (split_dataset_by_node), _shard_and_clean must NOT stride it a second
    # time - the old behavior silently dropped (num_workers-1)/num_workers of
    # every rank's data.
    import jepa_slm.data as data_mod

    settings = DataSettings(dataset="synthetic", normalize_text=True)
    dataset = TextStreamDataset(settings, rank=0, world_size=4)
    docs = [f"doc number {i}" for i in range(10)]

    monkeypatch.setattr(data_mod, "get_worker_info", lambda: _FakeWorker(0))
    got = [
        r["text"]
        for r in dataset._shard_and_clean(docs, node_sharded=True, worker_sharded=True)
    ]
    assert got == docs  # the worker's slice passes through unfiltered


def test_worker_sharded_only_stream_strides_by_rank(monkeypatch) -> None:
    # split_dataset_by_node failed but the stream still self-shards by worker:
    # the only remaining split to apply is across ranks.
    import jepa_slm.data as data_mod

    docs = [f"doc number {i}" for i in range(10)]
    monkeypatch.setattr(data_mod, "get_worker_info", lambda: _FakeWorker(0))

    settings = DataSettings(dataset="synthetic", normalize_text=True)
    per_rank = [
        [
            r["text"]
            for r in TextStreamDataset(settings, rank=rank, world_size=2)._shard_and_clean(
                docs, node_sharded=False, worker_sharded=True
            )
        ]
        for rank in (0, 1)
    ]
    assert set(per_rank[0]).isdisjoint(per_rank[1])
    assert sorted(per_rank[0] + per_rank[1]) == docs


def test_packed_skip_examples_skips_blocks_not_documents() -> None:
    # In packed mode samples_consumed counts fixed-length blocks; resume must
    # skip exactly that many blocks, not that many documents.
    settings = DataSettings(dataset="synthetic", max_samples=50, normalize_text=True)
    tok = ByteSmokeTokenizer(320)
    baseline = list(PackedTokenDataset(settings, tok, source_length=16))

    skipped = list(
        PackedTokenDataset(
            DataSettings(
                dataset="synthetic", max_samples=50, normalize_text=True, skip_examples=3
            ),
            tok,
            source_length=16,
        )
    )
    assert skipped[0]["input_ids"] == baseline[3]["input_ids"]


def test_mask_token_prefers_sentinel_over_unk() -> None:
    from jepa_slm.data import _resolve_mask_token_id

    class SentinelTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        unk_token_id = 2

        def convert_tokens_to_ids(self, token: str) -> int:
            return 32099 if token == "<extra_id_0>" else self.unk_token_id

    class NoSentinelTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        unk_token_id = 2

        def convert_tokens_to_ids(self, token: str) -> int:
            return self.unk_token_id  # unknown strings come back as unk

    # A real sentinel wins; otherwise fall back to unk (never a text token).
    assert _resolve_mask_token_id(SentinelTokenizer()) == 32099
    assert _resolve_mask_token_id(NoSentinelTokenizer()) == 2
    assert _resolve_mask_token_id(ByteSmokeTokenizer(320)) == 2
