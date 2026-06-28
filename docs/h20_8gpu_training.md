# 8x NVIDIA H20 Training Plan

## Goal

Use all 8 H20 GPUs efficiently for the 0.2B JEPA-augmented encoder-decoder model.

The default path is **DDP + BF16 + large per-rank batches**. For this model size, full FSDP or ZeRO-3 is usually counterproductive because the model already fits comfortably on each H20 and aggressive sharding increases communication.

> Attention backend: T5's relative-position bias is incompatible with
> FlashAttention-2 and PyTorch SDPA (`T5ForConditionalGeneration._supports_sdpa`
> is `False`), so the trainer uses eager attention. The `attention_kernel` config
> value is tried then falls back to eager automatically — it is not silently
> ignored. `flash_attn` is therefore optional and unused for this model.

## Assumptions

- 8 visible NVIDIA H20 GPUs.
- Hopper-class BF16 support.
- About 96 GB memory per GPU.
- Single node with working NCCL peer-to-peer or high-speed interconnect.

If the actual H20 variant differs, update `configs/train_h20_8gpu.yaml`.

## Environment

Create the environment:

```bash
conda env create -f envs/jepa-h20-cu124.yml
conda activate jepa-h20
python -m pip install --no-build-isolation -r envs/requirements-h20-optional.txt
python scripts/h20_sanity_check.py
```

Or use the helper:

```bash
bash scripts/install_h20_env.sh
```

## Launch

```bash
conda activate jepa-h20
bash scripts/launch_h20_8gpu.sh configs/train_h20_8gpu.yaml
```

This launches:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m jepa_slm.train --config configs/train_h20_8gpu.yaml
```

## Utilization Strategy

The trainer now honors the throughput knobs that were previously dead config
(see `docs/optimization_changes.md`):

- **BF16** for Tensor Core throughput and stable training (no GradScaler needed).
- **TF32 matmul** is enabled at startup (`hardware.allow_tf32`, `matmul_precision`).
- **Per-rank data sharding** is active, so 8 GPUs see 8 disjoint data slices —
  the headline `approximate_tokens_per_step` is now real, not 8x optimistic.
- **DataLoader workers** (`performance.dataloader_num_workers_per_rank: 8`),
  `prefetch_factor`, and `persistent_workers` keep the GPUs fed; the fast
  tokenizer (`tokenizer_use_fast: true`) is used.
- **Dynamic padding** (`batching.dynamic_padding: true`) pads to the longest
  member of each batch (multiple of 8), cutting padding FLOPs vs fixed 512.
- **`torch.compile`** runs with `dynamic=True`; the JEPA loss is computed over the
  full (static-shaped) sequence, so variable padded lengths do not trigger
  recompilation.
- `gradient_accumulation_steps=1` initially; if you raise it, `no_sync()` already
  skips the all-reduce on non-final micro-steps.
- Start at `64` sequences per GPU with `source_length=512`, `target_length=256`.
  The trainer uses fixed micro-batches — increase `per_gpu_micro_batch_sequences`
  manually after the first profiler run (auto batch-autotune is not implemented).
- `sequence_packing` is not implemented; the trainer logs a note and uses dynamic
  padding instead.
- Prefer DDP for 0.2B; FSDP/ZeRO is unnecessary at this size.

## What To Watch

- GPU utilization should stay high after data pipeline warmup.
- HBM usage should land near 85-92% after batch autotune.
- If utilization is low and memory is available, increase `per_gpu_micro_batch_sequences`.
- If utilization is low but memory is full, reduce padding with better packing.
- If rank-to-rank time varies, check dataloader workers, storage throughput, and NCCL topology.

## Dataset Path

Use `FineWeb-Edu sample-10BT` for the first ablation. Move to `SmolLM-Corpus` only after `CE+JEPA` beats or matches `CE-only` on validation CE and improves encoder probes.
