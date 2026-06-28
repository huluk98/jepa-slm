"""Executable training loop for JEPA-augmented encoder-decoder models."""

from __future__ import annotations

import math
import os
import random
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW

from .config import TrainingConfig
from .data import build_dataloader, load_tokenizer, move_batch_to_device
from .modeling import JepaEncoderDecoder, assert_encoder_only_jepa_contract
from .objectives import ema_tau, jepa_lambda_schedule

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


def setup_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        chosen = backend if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=chosen)
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


def configure_matmul_precision(config: TrainingConfig) -> None:
    """Enable TF32 / set float32 matmul precision when requested."""

    if config.runtime.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    precision = (config.runtime.matmul_precision or "high").lower()
    if precision in {"highest", "high", "medium"}:
        try:
            torch.set_float32_matmul_precision(precision)
        except Exception:  # noqa: BLE001 - older torch may not support all values
            pass


def precision_dtype(precision: str) -> torch.dtype | None:
    if precision.lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision.lower() in {"fp16", "float16"}:
        return torch.float16
    return None


def build_optimizer(model: JepaEncoderDecoder, config: TrainingConfig) -> AdamW:
    """AdamW with decoupled weight decay applied only to matmul weights.

    Biases, LayerNorm/RMSNorm gains, the relative-position-bias tables, and the
    (tied) token embeddings are excluded from weight decay, following the
    GPT/LLaMA/T5x convention.
    """

    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        is_no_decay = (
            param.ndim < 2
            or "norm" in lowered
            or "embed" in lowered
            or "relative_attention_bias" in lowered
            or name.endswith("mask_marker")
        )
        (no_decay_params if is_no_decay else decay_params).append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.optimizer.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(
        param_groups,
        lr=config.optimizer.learning_rate,
        betas=config.optimizer.betas,
        fused=torch.cuda.is_available(),
    )


def lr_scale(step: int, max_steps: int, warmup_fraction: float) -> float:
    warmup_steps = max(1, int(max_steps * warmup_fraction))
    if step < warmup_steps:
        return max(1e-8, step / warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, object] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:  # noqa: BLE001 - device count mismatch on resume
            pass


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: JepaEncoderDecoder | DistributedDataParallel,
    optimizer: AdamW,
    config: TrainingConfig,
    samples_consumed: int = 0,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
    save_rng: bool = True,
) -> None:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    checkpoint_dir = output_dir / f"step-{step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "step": step,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "samples_consumed": samples_consumed,
    }
    if scaler is not None and scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()
    if save_rng:
        payload["rng"] = _rng_state()
    torch.save(payload, checkpoint_dir / "trainer_state.pt")


def checkpoint_state_path(path: Path) -> Path:
    if path.is_dir():
        return path / "trainer_state.pt"
    return path


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:  # noqa: BLE001 - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def load_checkpoint(
    path: Path,
    model: JepaEncoderDecoder | DistributedDataParallel,
    optimizer: AdamW,
    device: torch.device,
    scaler=None,
    restore_rng: bool = True,
) -> int:
    """Restore model/optimizer (and RNG/scaler) state. Returns the step."""
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    state = torch.load(checkpoint_state_path(path), map_location=device, weights_only=False)
    raw_model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    if restore_rng:
        _restore_rng_state(state.get("rng"))
    return int(state["step"])


def _global_flag(local: bool, device: torch.device) -> bool:
    """Reduce a boolean across ranks with MAX (any-rank-true)."""
    if dist.is_available() and dist.is_initialized():
        flag = torch.tensor(int(local), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())
    return local


def should_stop(stop_file: str | None, device: torch.device) -> bool:
    local_stop = _STOP_REQUESTED or bool(stop_file and Path(stop_file).exists())
    return _global_flag(local_stop, device)


def loss_is_bad(loss_value: float, max_loss: float) -> bool:
    """A loss is 'bad' if it is non-finite, or exceeds max_loss when set."""
    if not math.isfinite(loss_value):
        return True
    return max_loss > 0.0 and loss_value > max_loss


@torch.no_grad()
def run_eval(
    model: JepaEncoderDecoder,
    eval_loader,
    device: torch.device,
    dtype: torch.dtype | None,
    max_batches: int,
) -> float | None:
    """Mean validation cross-entropy over up to max_batches, averaged across ranks.

    Uses the unwrapped model (no DDP gradient sync). Returns None if no batches.
    """
    was_training = model.training
    model.eval()
    total = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    try:
        for index, batch in enumerate(eval_loader):
            if index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                output = model(batch, jepa_weight=0.0)
            total = total + output.ce_loss.detach().float()
            count = count + 1
    finally:
        if was_training:
            model.train()
            # Keep the EMA target encoder frozen in eval (architectural contract);
            # model.train() above would otherwise flip it back to train mode.
            ema = getattr(model, "ema_encoder", None)
            if ema is not None:
                ema.eval()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    if count.item() == 0:
        return None
    return float((total / count).item())


def train(config: TrainingConfig) -> None:
    install_signal_handlers()
    rank, local_rank, world_size = setup_distributed(config.distributed.backend)
    set_seed(config.runtime.seed, rank)
    configure_matmul_precision(config)
    device = select_device(local_rank)
    dtype = precision_dtype(config.runtime.precision)
    use_grad_scaler = dtype is torch.float16 and device.type == "cuda"
    if (
        rank == 0
        and dtype is torch.float16
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        print(
            "[jepa-slm] precision=fp16 on a bf16-capable GPU: prefer precision=bf16 "
            "(wider dynamic range, no loss scaling, same Tensor Core throughput).",
            flush=True,
        )

    tokenizer = load_tokenizer(config.data, vocab_size=config.model.vocab_size)

    if config.batching.sequence_packing and rank == 0:
        print(
            "[jepa-slm] sequence packing enabled: emitting fully-packed "
            f"{config.batching.source_length}-token blocks.",
            flush=True,
        )

    # On resume, skip the data each rank already consumed so a restarted
    # streaming run does not replay the head of the corpus.
    resume_samples = 0
    if config.runtime.resume_from:
        try:
            ckpt = torch.load(
                checkpoint_state_path(Path(config.runtime.resume_from)),
                map_location="cpu",
                weights_only=False,
            )
            resume_samples = int(ckpt.get("samples_consumed", 0))
        except Exception:  # noqa: BLE001 - best-effort; fall back to no skip
            resume_samples = 0
    if resume_samples:
        config = _with_skip_examples(config, resume_samples // max(1, world_size))

    dataloader = build_dataloader(
        config.data,
        tokenizer,
        source_length=config.batching.source_length,
        target_length=config.batching.target_length,
        batch_size=config.batching.per_gpu_micro_batch_sequences,
        num_workers=config.performance.num_workers,
        rank=rank,
        world_size=world_size,
        prefetch_factor=config.performance.prefetch_factor,
        persistent_workers=config.performance.persistent_workers,
        pin_memory=config.performance.pin_memory,
        dynamic_padding=config.batching.dynamic_padding,
        pad_to_multiple_of=config.batching.pad_to_multiple_of,
        sequence_packing=config.batching.sequence_packing,
    )

    eval_loader = None
    if config.runtime.eval_every_steps > 0 and config.data.eval_dataset:
        from dataclasses import replace as _replace

        eval_settings = _replace(
            config.data, dataset=config.data.eval_dataset, skip_examples=0
        )
        eval_loader = build_dataloader(
            eval_settings,
            tokenizer,
            source_length=config.batching.source_length,
            target_length=config.batching.target_length,
            batch_size=config.batching.per_gpu_micro_batch_sequences,
            num_workers=0,
            rank=rank,
            world_size=world_size,
            dynamic_padding=config.batching.dynamic_padding,
            pad_to_multiple_of=config.batching.pad_to_multiple_of,
            sequence_packing=config.batching.sequence_packing,
        )
    elif config.runtime.eval_every_steps > 0 and rank == 0:
        print(
            "[jepa-slm] eval_every_steps is set but data.eval_dataset is not; "
            "skipping validation-CE.",
            flush=True,
        )

    model = JepaEncoderDecoder(
        config.model,
        config.jepa,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        attn_implementation=config.runtime.attn_implementation,
    )
    assert_encoder_only_jepa_contract(model)
    if config.runtime.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    if config.runtime.compile and hasattr(torch, "compile"):
        # dynamic=True so variable padded lengths do not trigger recompilation.
        model = torch.compile(model, dynamic=True)

    train_model: JepaEncoderDecoder | DistributedDataParallel = model
    if world_size > 1:
        train_model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=config.distributed.find_unused_parameters,
            gradient_as_bucket_view=config.distributed.gradient_as_bucket_view,
            static_graph=config.distributed.static_graph,
        )

    raw_model = train_model.module if isinstance(train_model, DistributedDataParallel) else train_model
    optimizer = build_optimizer(raw_model, config)
    scaler = _make_grad_scaler(use_grad_scaler)

    step = 0
    samples_consumed = resume_samples
    if config.runtime.resume_from:
        step = load_checkpoint(
            Path(config.runtime.resume_from), raw_model, optimizer, device, scaler=scaler
        )
        if rank == 0:
            print(f"Resumed from {config.runtime.resume_from} at step {step}.", flush=True)

    output_dir = Path(config.runtime.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    max_steps = config.runtime.max_steps
    accumulation = max(1, config.batching.gradient_accumulation_steps)
    micro_batch = config.batching.per_gpu_micro_batch_sequences
    stop_training = False
    optimizer.zero_grad(set_to_none=True)
    micro = 0  # running micro-step counter, robust across epoch boundaries
    guard_enabled = config.runtime.abort_on_nonfinite or config.runtime.max_loss > 0
    step_bad = False  # any micro in the current accumulation window was bad
    bad_in_row = 0  # consecutive diverging optimizer steps

    # Throughput accounting (source tokens the model computed over per step).
    tokens_per_step = world_size * accumulation * micro_batch * config.batching.source_length
    tokens_seen = 0
    tokens_at_window = 0
    window_start = time.perf_counter()

    try:
        while step < max_steps and not stop_training:
            for batch in dataloader:
                batch = move_batch_to_device(
                    batch, device, non_blocking=config.performance.pin_memory
                )
                samples_consumed += micro_batch * world_size
                jepa_weight = jepa_lambda_schedule(
                    step,
                    max_steps,
                    config.jepa.lambda_warmup_fraction,
                    config.jepa.lambda_peak,
                    config.jepa.lambda_final_weight,
                    config.jepa.lambda_final_phase_fraction,
                )

                is_last_micro = micro == accumulation - 1
                # Skip the DDP all-reduce on non-final micro-steps.
                sync_context = (
                    train_model.no_sync()
                    if (not is_last_micro and isinstance(train_model, DistributedDataParallel))
                    else nullcontext()
                )
                with sync_context:
                    with torch.autocast(
                        device_type=device.type, dtype=dtype, enabled=dtype is not None
                    ):
                        output = train_model(batch, jepa_weight=jepa_weight)
                        loss = output.loss / accumulation
                    scaler.scale(loss).backward()

                if guard_enabled:
                    step_bad = step_bad or loss_is_bad(
                        float(output.loss.detach()), config.runtime.max_loss
                    )

                micro += 1
                if not is_last_micro:
                    continue
                micro = 0

                # Divergence guard ("stop loss"): if the loss went non-finite or
                # blew past max_loss, discard this step's grads (do NOT apply a
                # NaN update) and, after enough consecutive bad steps, stop.
                if guard_enabled:
                    bad = _global_flag(step_bad, device)
                    step_bad = False
                    if bad:
                        optimizer.zero_grad(set_to_none=True)
                        bad_in_row += 1
                        if rank == 0:
                            print(
                                f"[jepa-slm] diverging loss at step {step} "
                                f"({bad_in_row}/{config.runtime.divergence_patience}); "
                                "skipping update.",
                                flush=True,
                            )
                        if bad_in_row >= config.runtime.divergence_patience:
                            stop_training = True
                            if rank == 0:
                                print(
                                    f"[jepa-slm] stop loss triggered: loss diverged "
                                    f"for {bad_in_row} consecutive steps; "
                                    "saving last good checkpoint and stopping.",
                                    flush=True,
                                )
                            break
                        continue
                    bad_in_row = 0

                if config.optimizer.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        raw_model.parameters(), config.optimizer.grad_clip_norm
                    )
                lr_factor = lr_scale(step, max_steps, config.optimizer.warmup_fraction)
                for group in optimizer.param_groups:
                    group["lr"] = config.optimizer.learning_rate * lr_factor
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                tau = ema_tau(step, max_steps, config.jepa.ema_tau_start, config.jepa.ema_tau_end)
                raw_model.update_ema_encoder(tau)
                tokens_seen += tokens_per_step

                if rank == 0 and step % config.runtime.log_every_steps == 0:
                    now = time.perf_counter()
                    elapsed = now - window_start
                    tok_per_s = (tokens_seen - tokens_at_window) / elapsed if elapsed > 0 else 0.0
                    window_start = now
                    tokens_at_window = tokens_seen
                    record = {
                        "step": step,
                        "loss": round(float(output.loss.detach().cpu()), 5),
                        "ce_loss": round(float(output.ce_loss.detach().cpu()), 5),
                        "jepa_loss": round(float(output.jepa_loss.detach().cpu()), 5),
                        "vicreg_loss": round(float(output.vicreg_loss.detach().cpu()), 5),
                        "jepa_weight": round(jepa_weight, 5),
                        "encoder_std": round(float(output.encoder_repr_std.cpu()), 5),
                        "predictor_std": round(float(output.predictor_repr_std.cpu()), 5),
                        "lr": optimizer.param_groups[0]["lr"],
                        "tok_per_s": round(tok_per_s),
                    }
                    if device.type == "cuda":
                        record["gpu_mem_gb"] = round(
                            torch.cuda.max_memory_allocated(device) / 1024**3, 2
                        )
                        torch.cuda.reset_peak_memory_stats(device)
                    print(record, flush=True)

                # Periodic validation cross-entropy (collective: all ranks run it).
                if (
                    eval_loader is not None
                    and config.runtime.eval_every_steps > 0
                    and step % config.runtime.eval_every_steps == 0
                ):
                    eval_ce = run_eval(
                        raw_model, eval_loader, device, dtype, config.runtime.eval_max_batches
                    )
                    if rank == 0 and eval_ce is not None:
                        print({"step": step, "eval_ce": round(eval_ce, 5)}, flush=True)
                    # Don't let eval wall-time / memory pollute the next throughput
                    # window or the next peak-memory reading.
                    window_start = time.perf_counter()
                    tokens_at_window = tokens_seen
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)

                step += 1
                if (
                    config.runtime.empty_cache_steps > 0
                    and device.type == "cuda"
                    and step % config.runtime.empty_cache_steps == 0
                ):
                    torch.cuda.empty_cache()
                if (
                    rank == 0
                    and config.runtime.save_every_steps > 0
                    and step % config.runtime.save_every_steps == 0
                ):
                    save_checkpoint(
                        output_dir, step, train_model, optimizer, config,
                        samples_consumed=samples_consumed, scaler=scaler,
                    )
                stop_training = should_stop(config.runtime.stop_file, device)
                if step >= max_steps or stop_training:
                    break
    finally:
        if rank == 0 and (not stop_training or config.runtime.save_on_stop):
            save_checkpoint(
                output_dir, step, train_model, optimizer, config,
                samples_consumed=samples_consumed, scaler=scaler,
            )
            if stop_training:
                print(f"Stop requested; saved checkpoint at step {step}.", flush=True)
        cleanup_distributed()


def _with_skip_examples(config: TrainingConfig, skip: int) -> TrainingConfig:
    from dataclasses import replace

    return replace(config, data=replace(config.data, skip_examples=skip))
