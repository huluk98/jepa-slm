from pathlib import Path

from jepa_slm.config import load_training_config


def test_launcher_data_source_classification() -> None:
    # run_training.sh decides "stream vs clean-first" from these resolved values.
    streaming = load_training_config(Path("configs/train_h20_8gpu.yaml"))
    assert streaming.data.dataset == "HuggingFaceFW/fineweb-edu"
    assert not any(ch in streaming.data.dataset for ch in "*?[")  # not a local glob

    local = load_training_config(Path("configs/train_h20_4gpu.yaml"))
    assert "*" in local.data.dataset  # local cleaned-shard glob -> needs cleaning


def test_load_full_h20_config_resolves_nested_model() -> None:
    config = load_training_config(Path("configs/train_h20_8gpu.yaml"))

    assert config.model.d_model == 768
    assert config.model.decoder_layers == 10
    assert config.jepa.target_encoder == "ema_encoder_only"
    assert config.batching.source_length == 512
    assert config.data.dataset == "HuggingFaceFW/fineweb-edu"


def test_load_4gpu_h20_config_preserves_stop_file() -> None:
    config = load_training_config(Path("configs/train_h20_4gpu.yaml"))

    assert config.batching.per_gpu_micro_batch_sequences == 64
    assert config.runtime.stop_file == "outputs/jepa-slm-h20-4gpu/STOP"
    assert config.runtime.save_on_stop is True
    # The full-memory config keeps fp32 master weights (param_dtype default).
    assert config.runtime.param_dtype == "fp32"


def test_load_4gpu_6gb_config_is_streaming_bf16_and_tiny_microbatch() -> None:
    # The tight 6 GiB config must: store weights in bf16 (shrinks optimizer
    # state), stream fineweb-edu (no local prep), and use micro-batch 1 with
    # gradient accumulation so it fits a ~6 GiB-per-GPU slice on GPUs 0-3.
    config = load_training_config(Path("configs/train_h20_4gpu_6gb.yaml"))

    assert config.runtime.param_dtype == "bf16"
    assert config.runtime.precision == "bf16"
    assert config.runtime.compile is False
    assert config.runtime.gradient_checkpointing is True
    assert config.batching.per_gpu_micro_batch_sequences == 1
    assert config.batching.gradient_accumulation_steps == 16
    assert config.batching.sequence_packing is True  # full 512-tok blocks, no waste
    assert config.data.dataset == "HuggingFaceFW/fineweb-edu"  # streamed, not a glob
    assert config.data.streaming is True
    assert not any(ch in config.data.dataset for ch in "*?[")
    assert config.runtime.empty_cache_steps == 50
    # ~1 epoch of sample-10BT (~11.5 B source tokens) with bounded checkpoint disk.
    assert config.runtime.max_steps == 350000
    assert config.runtime.keep_last_checkpoints == 5
    # Same flagship 0.2B model shape as the full-memory config.
    assert config.model.d_model == 768


def test_performance_and_distributed_blocks_are_parsed() -> None:
    config = load_training_config(Path("configs/train_h20_8gpu.yaml"))

    # performance: block is now wired through (was previously dead config).
    assert config.performance.num_workers == 8
    assert config.performance.prefetch_factor == 4
    assert config.performance.persistent_workers is True
    # distributed: knobs feed DDP construction.
    assert config.distributed.static_graph is True
    assert config.distributed.find_unused_parameters is False
    # matmul / tf32 from hardware: block.
    assert config.runtime.matmul_precision == "high"
    assert config.runtime.allow_tf32 is True
    # dynamic padding and a non-smoke token budget.
    assert config.batching.dynamic_padding is True
    assert config.runtime.max_steps >= 10_000


def test_objective_block_wires_final_phase_and_vicreg() -> None:
    config = load_training_config(Path("configs/train_h20_8gpu.yaml"))

    assert config.jepa.lambda_final_weight == 0.05
    assert config.jepa.vicreg_variance_weight > 0
    assert config.jepa.predictor_full_context is True
