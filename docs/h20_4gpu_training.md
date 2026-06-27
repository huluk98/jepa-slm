# 4x NVIDIA H20 Training

Use this on one node with 4 visible H20 GPUs, each around `97871 MiB`.

Launch:

```bash
bash scripts/launch_h20_4gpu.sh
```

Stop safely:

```bash
touch outputs/jepa-slm-h20-4gpu/STOP
```

The trainer checks the stop file after each completed optimizer step. When it
sees the file, all DDP ranks stop together and rank 0 saves a checkpoint.

Resume:

```bash
bash scripts/launch_h20_4gpu.sh \
  configs/train_h20_4gpu.yaml \
  outputs/jepa-slm-h20-4gpu/step-00000250
```

Defaults:

```yaml
gpus_per_node: 4
per_gpu_micro_batch_sequences: 64
global_batch_sequences: 256
source_length: 512
target_length: 256
precision: bf16
save_every_steps: 250
```

The config trains from:

```text
data/clean/fineweb-edu-sample10bt-100k/clean-*.jsonl
```
