"""Configuration dataclasses and YAML loading helpers for JEPA-SLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelShape:
    """Transformer dimensions for a small encoder-decoder model."""

    family: str = "t5_ul2_encoder_decoder"
    target_trainable_params: int = 200_000_000
    d_model: int = 768
    encoder_layers: int = 10
    decoder_layers: int = 10
    d_ff: int = 3072
    attention_heads: int = 12
    vocab_size: int = 32_128
    position_encoding: str = "relative_bias"
    ffn_activation: str = "gelu"
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


@dataclass(frozen=True)
class JepaSettings:
    """JEPA-specific objective settings."""

    enabled: bool = True
    target_encoder: str = "ema_encoder_only"
    predictor_width: int = 512
    predictor_layers: int = 3
    top_k_target_layers: int = 4
    normalize_targets: bool = True
    latent_loss: str = "smooth_l1_or_mse"
    # Predictor sees the full (contextualized) student encoder sequence and a
    # learned mask marker at masked positions, then reads out predictions at the
    # masked slots. This is the I-JEPA/data2vec contract; the legacy path only
    # fed the gathered masked vectors and is kept for ablation via this flag.
    predictor_full_context: bool = True
    lambda_peak: float = 0.25
    lambda_warmup_fraction: float = 0.10
    # Final "CE polish" phase: lambda decays from peak to lambda_final_weight
    # over the last lambda_final_phase_fraction of training.
    lambda_final_weight: float = 0.05
    lambda_final_phase_fraction: float = 0.20
    # Optional VICReg-style collapse guard on the predicted/student latents.
    # Off by default (data2vec relies on EMA + target normalization); enable in
    # production configs for an explicit anti-collapse penalty.
    vicreg_variance_weight: float = 0.0
    vicreg_covariance_weight: float = 0.0
    vicreg_variance_gamma: float = 1.0
    ema_tau_start: float = 0.99
    ema_tau_end: float = 0.9995


@dataclass(frozen=True)
class DataSettings:
    """Dataset and tokenizer settings for the training entrypoint."""

    dataset: str = "HuggingFaceFW/fineweb-edu"
    subset: str | None = "sample-10BT"
    split: str = "train"
    text_field: str = "text"
    tokenizer_path: str | None = None
    tokenizer_name: str = "google-t5/t5-small"
    tokenizer_use_fast: bool = True
    streaming: bool = True
    max_samples: int | None = None
    normalize_text: bool = True
    min_chars: int = 0
    max_chars: int | None = None
    # Number of (post-clean) examples to skip at the head of each rank's shard.
    # Used on resume so a restarted streaming run does not replay early data.
    skip_examples: int = 0
    # Optional held-out source for periodic validation-CE (path/glob or HF name).
    # When set together with runtime.eval_every_steps > 0, the trainer logs eval_ce.
    eval_dataset: str | None = None


@dataclass(frozen=True)
class RuntimeSettings:
    """Runtime training knobs."""

    output_dir: str = "outputs/jepa-slm"
    max_steps: int = 1_000
    save_every_steps: int = 1_000
    resume_from: str | None = None
    stop_file: str | None = None
    save_on_stop: bool = True
    log_every_steps: int = 10
    eval_every_steps: int = 0
    eval_max_batches: int = 50
    seed: int = 13
    precision: str = "bf16"
    # Storage dtype for the *weights* and therefore the optimizer state. The
    # default "fp32" keeps fp32 master weights with bf16/fp16 autocast compute
    # (best numerics). Set to "bf16" to also store the weights, gradients, and
    # AdamW moments in bf16 -- this roughly halves the static optimizer-state
    # footprint (~16 -> ~8 bytes/param), which is what lets the 0.2B model train
    # inside a tight (e.g. 6 GiB) per-GPU memory budget. Trade-off: bf16 master
    # weights have ~3 decimal digits of precision, so very small updates can be
    # lost; prefer fp32 storage whenever the memory budget allows.
    param_dtype: str = "fp32"
    compile: bool = False
    gradient_checkpointing: bool = True
    device: str = "auto"
    # Float32 matmul precision and TF32 toggles (honored at trainer startup).
    matmul_precision: str = "high"
    allow_tf32: bool = True
    # Attention backend for the T5 stack: "auto" tries sdpa then falls back to
    # eager (T5 relative-position bias is incompatible with FlashAttention-2).
    attn_implementation: str = "auto"
    # Call torch.cuda.empty_cache() every N optimizer steps when > 0.
    empty_cache_steps: int = 0
    # Divergence guard ("stop loss"): skip an optimizer step whose loss is
    # non-finite (NaN/Inf) or, when max_loss > 0, exceeds max_loss. After
    # divergence_patience consecutive bad steps, stop training gracefully and
    # save the last good checkpoint. Protects bf16 runs (no GradScaler) from a
    # single NaN permanently corrupting the weights.
    abort_on_nonfinite: bool = True
    max_loss: float = 0.0
    divergence_patience: int = 5


@dataclass(frozen=True)
class BatchingSettings:
    """Batch and sequence lengths."""

    source_length: int = 512
    target_length: int = 256
    per_gpu_micro_batch_sequences: int = 8
    gradient_accumulation_steps: int = 1
    # Pad each batch to the longest member (rounded up to pad_to_multiple_of)
    # instead of always padding to source_length/target_length.
    dynamic_padding: bool = True
    pad_to_multiple_of: int = 8
    # Greedily concatenate cleaned documents up to ~source_length tokens before
    # tokenizing, to cut padding waste. Falls back to per-document otherwise.
    sequence_packing: bool = False


@dataclass(frozen=True)
class OptimizerSettings:
    """Optimizer settings."""

    learning_rate: float = 4e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    warmup_fraction: float = 0.02


@dataclass(frozen=True)
class PerformanceSettings:
    """DataLoader / input-pipeline throughput knobs."""

    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True


@dataclass(frozen=True)
class DistributedSettings:
    """DDP construction knobs."""

    backend: str = "nccl"
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = True
    compile: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Fully resolved training config used by the pipeline."""

    model: ModelShape = field(default_factory=ModelShape)
    jepa: JepaSettings = field(default_factory=JepaSettings)
    data: DataSettings = field(default_factory=DataSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    batching: BatchingSettings = field(default_factory=BatchingSettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    distributed: DistributedSettings = field(default_factory=DistributedSettings)


def _filter_dataclass_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    fields = getattr(cls, "__dataclass_fields__", {})
    return {key: value for key, value in values.items() if key in fields}


def _source_target_lengths(training: dict[str, Any], batching: dict[str, Any]) -> tuple[int, int]:
    source_length = (
        batching.get("source_length")
        or training.get("source_length")
        or training.get("source_length_pretrain")
        or BatchingSettings.source_length
    )
    target_length = (
        batching.get("target_length")
        or training.get("target_length")
        or training.get("target_length_pretrain")
        or BatchingSettings.target_length
    )
    return int(source_length), int(target_length)


def _parse_simple_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"null", "none", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_yaml_scalar(part) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        # ponytail: simple nested mappings only; install PyYAML for full YAML syntax.
        root: dict[str, Any] = {}
        stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.rstrip()
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            key, separator, value = line.strip().partition(":")
            if not separator:
                raise ValueError(f"Unsupported YAML line in {path}: {raw_line}")
            while indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if value.strip():
                parent[key] = _parse_simple_yaml_scalar(value)
            else:
                parent[key] = {}
                stack.append((indent, parent[key]))
        return root

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_training_config(path: Path | str) -> TrainingConfig:
    """Load model/training YAML into a typed training config.

    The repository has both compact model configs and full launch configs. This
    loader accepts either shape and fills in conservative defaults.
    """

    config_path = Path(path)
    raw = _load_yaml(config_path)

    model_raw = dict(raw.get("model", {}))
    jepa_raw = dict(raw.get("jepa", {}))
    objective_raw = dict(raw.get("objective", {}))
    training_raw = dict(raw.get("training", {}))
    batching_raw = dict(raw.get("batching", {}))
    optimizer_raw = dict(raw.get("optimizer", {}))
    runtime_raw = dict(raw.get("runtime", {}))
    data_raw = dict(raw.get("data", {}))
    hardware_raw = dict(raw.get("hardware", {}))
    distributed_raw = dict(raw.get("distributed", {}))
    performance_raw = dict(raw.get("performance", {}))

    nested_model_path = model_raw.get("config_path")
    if nested_model_path:
        nested_path = Path(nested_model_path)
        if not nested_path.is_absolute() and not nested_path.exists():
            nested_path = config_path.parent / nested_path
        nested = load_training_config(nested_path)
        model = nested.model
        jepa = nested.jepa
    else:
        model = ModelShape(**_filter_dataclass_kwargs(ModelShape, model_raw))
        if "predictor_width" not in jepa_raw and model.predictor_width:
            jepa_raw["predictor_width"] = model.predictor_width
        if "predictor_layers" not in jepa_raw and model.predictor_layers:
            jepa_raw["predictor_layers"] = model.predictor_layers
        jepa = JepaSettings(**_filter_dataclass_kwargs(JepaSettings, jepa_raw))

    if objective_raw:
        jepa = JepaSettings(
            **{
                **jepa.__dict__,
                "lambda_peak": objective_raw.get("jepa_weight_peak", jepa.lambda_peak),
                "lambda_warmup_fraction": objective_raw.get(
                    "jepa_warmup_fraction", jepa.lambda_warmup_fraction
                ),
                "lambda_final_weight": objective_raw.get(
                    "jepa_final_phase_weight", jepa.lambda_final_weight
                ),
                "lambda_final_phase_fraction": objective_raw.get(
                    "jepa_final_phase_fraction", jepa.lambda_final_phase_fraction
                ),
                "vicreg_variance_weight": objective_raw.get(
                    "vicreg_variance_weight", jepa.vicreg_variance_weight
                ),
                "vicreg_covariance_weight": objective_raw.get(
                    "vicreg_covariance_weight", jepa.vicreg_covariance_weight
                ),
                "vicreg_variance_gamma": objective_raw.get(
                    "vicreg_variance_gamma", jepa.vicreg_variance_gamma
                ),
                "ema_tau_start": objective_raw.get("ema_tau_start", jepa.ema_tau_start),
                "ema_tau_end": objective_raw.get("ema_tau_end", jepa.ema_tau_end),
            }
        )

    source_length, target_length = _source_target_lengths(training_raw, batching_raw)
    batching = BatchingSettings(
        source_length=source_length,
        target_length=target_length,
        per_gpu_micro_batch_sequences=int(
            batching_raw.get(
                "per_gpu_micro_batch_sequences",
                BatchingSettings.per_gpu_micro_batch_sequences,
            )
        ),
        gradient_accumulation_steps=int(
            batching_raw.get(
                "gradient_accumulation_steps",
                BatchingSettings.gradient_accumulation_steps,
            )
        ),
        dynamic_padding=bool(
            batching_raw.get("dynamic_padding", BatchingSettings.dynamic_padding)
        ),
        pad_to_multiple_of=int(
            batching_raw.get("pad_to_multiple_of", BatchingSettings.pad_to_multiple_of)
        ),
        sequence_packing=bool(
            batching_raw.get(
                "sequence_packing",
                performance_raw.get("sequence_packing", BatchingSettings.sequence_packing),
            )
        ),
    )

    if "pilot_dataset" in data_raw and "dataset" not in data_raw:
        data_raw["dataset"] = data_raw["pilot_dataset"]
    if "pilot_subset" in data_raw and "subset" not in data_raw:
        data_raw["subset"] = data_raw["pilot_subset"]
    data = DataSettings(**_filter_dataclass_kwargs(DataSettings, data_raw))

    runtime = RuntimeSettings(
        **_filter_dataclass_kwargs(
            RuntimeSettings,
            {
                **runtime_raw,
                "precision": hardware_raw.get("precision", runtime_raw.get("precision", "bf16")),
                "param_dtype": hardware_raw.get(
                    "param_dtype", runtime_raw.get("param_dtype", "fp32")
                ),
                "compile": distributed_raw.get("compile", runtime_raw.get("compile", False)),
                "gradient_checkpointing": training_raw.get(
                    "gradient_checkpointing",
                    runtime_raw.get("gradient_checkpointing", True),
                ),
                "matmul_precision": hardware_raw.get(
                    "matmul_precision", runtime_raw.get("matmul_precision", "high")
                ),
                "allow_tf32": hardware_raw.get(
                    "allow_tf32", runtime_raw.get("allow_tf32", True)
                ),
                "attn_implementation": hardware_raw.get(
                    "attention_kernel",
                    performance_raw.get(
                        "attention_kernel", runtime_raw.get("attn_implementation", "auto")
                    ),
                ),
                "empty_cache_steps": performance_raw.get(
                    "empty_cache_steps", runtime_raw.get("empty_cache_steps", 0)
                ),
            },
        )
    )

    if "betas" in optimizer_raw:
        optimizer_raw["betas"] = tuple(optimizer_raw["betas"])
    optimizer = OptimizerSettings(**_filter_dataclass_kwargs(OptimizerSettings, optimizer_raw))

    performance = PerformanceSettings(
        num_workers=int(
            performance_raw.get(
                "dataloader_num_workers_per_rank",
                performance_raw.get("num_workers", PerformanceSettings.num_workers),
            )
        ),
        prefetch_factor=int(
            performance_raw.get(
                "dataloader_prefetch_factor",
                performance_raw.get("prefetch_factor", PerformanceSettings.prefetch_factor),
            )
        ),
        persistent_workers=bool(
            performance_raw.get("persistent_workers", PerformanceSettings.persistent_workers)
        ),
        pin_memory=bool(performance_raw.get("pin_memory", PerformanceSettings.pin_memory)),
    )

    distributed = DistributedSettings(
        backend=str(distributed_raw.get("backend", DistributedSettings.backend)),
        find_unused_parameters=bool(
            distributed_raw.get("find_unused_parameters", DistributedSettings.find_unused_parameters)
        ),
        gradient_as_bucket_view=bool(
            distributed_raw.get("gradient_as_bucket_view", DistributedSettings.gradient_as_bucket_view)
        ),
        static_graph=bool(distributed_raw.get("static_graph", DistributedSettings.static_graph)),
        compile=bool(distributed_raw.get("compile", runtime.compile)),
    )

    return TrainingConfig(
        model=model,
        jepa=jepa,
        data=data,
        runtime=runtime,
        batching=batching,
        optimizer=optimizer,
        performance=performance,
        distributed=distributed,
    )
