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
    lambda_peak: float = 0.25
    lambda_warmup_fraction: float = 0.10
    cross_attention_jepa_weight: float = 0.0
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
    streaming: bool = True
    max_samples: int | None = None
    normalize_text: bool = True
    min_chars: int = 0
    max_chars: int | None = None


@dataclass(frozen=True)
class RuntimeSettings:
    """Runtime training knobs."""

    output_dir: str = "outputs/jepa-slm"
    max_steps: int = 1_000
    save_every_steps: int = 1_000
    log_every_steps: int = 10
    eval_every_steps: int = 0
    seed: int = 13
    precision: str = "bf16"
    compile: bool = False
    gradient_checkpointing: bool = True
    device: str = "auto"


@dataclass(frozen=True)
class BatchingSettings:
    """Batch and sequence lengths."""

    source_length: int = 512
    target_length: int = 256
    per_gpu_micro_batch_sequences: int = 8
    gradient_accumulation_steps: int = 1


@dataclass(frozen=True)
class OptimizerSettings:
    """Optimizer settings."""

    learning_rate: float = 4e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    warmup_fraction: float = 0.02


@dataclass(frozen=True)
class TrainingConfig:
    """Fully resolved training config used by the pipeline."""

    model: ModelShape = field(default_factory=ModelShape)
    jepa: JepaSettings = field(default_factory=JepaSettings)
    data: DataSettings = field(default_factory=DataSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    batching: BatchingSettings = field(default_factory=BatchingSettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)


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
                "compile": distributed_raw.get("compile", runtime_raw.get("compile", False)),
                "gradient_checkpointing": training_raw.get(
                    "gradient_checkpointing",
                    runtime_raw.get("gradient_checkpointing", True),
                ),
            },
        )
    )

    if "betas" in optimizer_raw:
        optimizer_raw["betas"] = tuple(optimizer_raw["betas"])
    optimizer = OptimizerSettings(**_filter_dataclass_kwargs(OptimizerSettings, optimizer_raw))

    return TrainingConfig(
        model=model,
        jepa=jepa,
        data=data,
        runtime=runtime,
        batching=batching,
        optimizer=optimizer,
    )
