"""Executable training loop for JEPA-augmented encoder-decoder models."""

from __future__ import annotations

import math
import os
import random
import signal
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW

from .config import TrainingConfig
from .data import build_dataloader, load_tokenizer, move_batch_to_device
from .modeling import JepaEncoderDecoder, assert_encoder_only_jepa_contract
from .objectives import ema_tau, linear_ramp

_STOP_REQUESTED = False


def request_stop(signum: int, _frame: object) -> None:
    """Ask the training loop to stop after the current optimizer step."""

    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"Received signal {signum}; stopping after the current step.", flush=True)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def select_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def set_seed(seed: int, rank: int = 0) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def precision_dtype(precision: str) -> torch.dtype | None:
    if precision.lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision.lower() in {"fp16", "float16"}:
        return torch.float16
    return None


def build_optimizer(model: JepaEncoderDecoder, config: TrainingConfig) -> AdamW:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return AdamW(
        trainable_params,
        lr=config.optimizer.learning_rate,
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay,
        fused=torch.cuda.is_available(),
    )


def lr_scale(step: int, max_steps: int, warmup_fraction: float) -> float:
    warmup_steps = max(1, int(max_steps * warmup_fraction))
    if step < warmup_steps:
        return max(1e-8, step / warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: JepaEncoderDecoder | DistributedDataParallel,
    optimizer: AdamW,
    config: TrainingConfig,
) -> None:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    checkpoint_dir = output_dir / f"step-{step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
        },
        checkpoint_dir / "trainer_state.pt",
    )


def checkpoint_state_path(path: Path) -> Path:
    if path.is_dir():
        return path / "trainer_state.pt"
    return path


def load_checkpoint(
    path: Path,
    model: JepaEncoderDecoder | DistributedDataParallel,
    optimizer: AdamW,
    device: torch.device,
) -> int:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    state = torch.load(checkpoint_state_path(path), map_location=device)
    raw_model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return int(state["step"])


def should_stop(stop_file: str | None, device: torch.device) -> bool:
    local_stop = _STOP_REQUESTED or bool(stop_file and Path(stop_file).exists())
    if dist.is_available() and dist.is_initialized():
        flag = torch.tensor(int(local_stop), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())
    return local_stop


def train(config: TrainingConfig) -> None:
    install_signal_handlers()
    rank, local_rank, world_size = setup_distributed()
    set_seed(config.runtime.seed, rank)
    device = select_device(local_rank)
    dtype = precision_dtype(config.runtime.precision)

    tokenizer = load_tokenizer(config.data, vocab_size=config.model.vocab_size)
    dataloader = build_dataloader(
        config.data,
        tokenizer,
        source_length=config.batching.source_length,
        target_length=config.batching.target_length,
        batch_size=config.batching.per_gpu_micro_batch_sequences,
    )

    model = JepaEncoderDecoder(
        config.model,
        config.jepa,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
    )
    assert_encoder_only_jepa_contract(model)
    if config.runtime.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    if config.runtime.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    train_model: JepaEncoderDecoder | DistributedDataParallel = model
    if world_size > 1:
        train_model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    raw_model = train_model.module if isinstance(train_model, DistributedDataParallel) else train_model
    optimizer = build_optimizer(raw_model, config)
    step = 0
    if config.runtime.resume_from:
        step = load_checkpoint(Path(config.runtime.resume_from), raw_model, optimizer, device)
        if rank == 0:
            print(f"Resumed from {config.runtime.resume_from} at step {step}.", flush=True)

    output_dir = Path(config.runtime.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    max_steps = config.runtime.max_steps
    accumulation = config.batching.gradient_accumulation_steps
    warmup_steps = max(1, int(max_steps * config.jepa.lambda_warmup_fraction))
    stop_training = False
    optimizer.zero_grad(set_to_none=True)

    try:
        while step < max_steps and not stop_training:
            for batch_index, batch in enumerate(dataloader):
                micro_step = batch_index % accumulation
                batch = move_batch_to_device(batch, device)
                jepa_weight = linear_ramp(step, warmup_steps, config.jepa.lambda_peak)

                with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                    output = train_model(batch, jepa_weight=jepa_weight)
                    loss = output.loss / accumulation

                loss.backward()
                if micro_step != accumulation - 1:
                    continue

                if config.optimizer.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        raw_model.parameters(), config.optimizer.grad_clip_norm
                    )
                lr_factor = lr_scale(step, max_steps, config.optimizer.warmup_fraction)
                for group in optimizer.param_groups:
                    group["lr"] = config.optimizer.learning_rate * lr_factor
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                tau = ema_tau(step, max_steps, config.jepa.ema_tau_start, config.jepa.ema_tau_end)
                raw_model.update_ema_encoder(tau)

                if rank == 0 and step % config.runtime.log_every_steps == 0:
                    print(
                        {
                            "step": step,
                            "loss": round(float(output.loss.detach().cpu()), 5),
                            "ce_loss": round(float(output.ce_loss.detach().cpu()), 5),
                            "jepa_loss": round(float(output.jepa_loss.detach().cpu()), 5),
                            "cross_attention_jepa_loss": round(
                                float(output.cross_attention_jepa_loss.detach().cpu()), 5
                            ),
                            "jepa_weight": round(jepa_weight, 5),
                            "encoder_var": round(float(output.encoder_state_variance.cpu()), 5),
                            "predictor_var": round(float(output.predictor_state_variance.cpu()), 5),
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                        flush=True,
                    )

                step += 1
                if (
                    rank == 0
                    and config.runtime.save_every_steps > 0
                    and step % config.runtime.save_every_steps == 0
                ):
                    save_checkpoint(output_dir, step, train_model, optimizer, config)
                stop_training = should_stop(config.runtime.stop_file, device)
                if step >= max_steps or stop_training:
                    break
    finally:
        if rank == 0 and (not stop_training or config.runtime.save_on_stop):
            save_checkpoint(output_dir, step, train_model, optimizer, config)
            if stop_training:
                print(f"Stop requested; saved checkpoint at step {step}.", flush=True)
        cleanup_distributed()
