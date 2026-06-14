# 8x NVIDIA H20 Training Plan

## Goal

Use all 8 H20 GPUs efficiently for the 0.2B JEPA-augmented encoder-decoder model.

The default path is **DDP + BF16 + FlashAttention + large per-rank batches**. For this model size, full FSDP or ZeRO-3 is usually counterproductive because the model already fits comfortably on each H20 and aggressive sharding increases communication.

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

- Use BF16 for Tensor Core throughput and stable training.
- Use FlashAttention 2 when available.
- Keep `gradient_accumulation_steps=1` initially; H20 memory should allow large micro-batches.
- Start at `64` sequences per GPU with `source_length=512` and `target_length=256`.
- Enable batch autotune to grow toward 92% memory utilization.
- Use sequence packing so short examples do not waste attention compute.
- Prefer DDP for 0.2B. Use `configs/deepspeed_h20_zero1.json` only if optimizer state or larger variants become the bottleneck.

## What To Watch

- GPU utilization should stay high after data pipeline warmup.
- HBM usage should land near 85-92% after batch autotune.
- If utilization is low and memory is available, increase `per_gpu_micro_batch_sequences`.
- If utilization is low but memory is full, reduce padding with better packing.
- If rank-to-rank time varies, check dataloader workers, storage throughput, and NCCL topology.

## Dataset Path

Use `FineWeb-Edu sample-10BT` for the first ablation. Move to `SmolLM-Corpus` only after `CE+JEPA` beats or matches `CE-only` on validation CE and improves encoder probes.
