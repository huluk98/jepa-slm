# JEPA-SLM

Research and verification tools for training a small encoder-decoder language model with a JEPA-style auxiliary representation objective.

The recommended architecture is:

```text
UL2/T5-style encoder-decoder + encoder-side JEPA/data2vec auxiliary loss
```

The key rule is simple: keep token-level cross entropy for generation, and add JEPA only as a train-time representation objective for the encoder.

## Quick Start

Generate the architecture verification report:

```bash
python3 scripts/jepa_verifier_agent.py
```

Generate the theory-backed architecture research report:

```bash
python3 scripts/jepa_research_agent.py
```

Generate the dataset recommendation report:

```bash
python3 scripts/dataset_recommender_agent.py
```

Generate the tokenizer recommendation report:

```bash
python3 scripts/tokenizer_recommender_agent.py
```

Generate the combined tokenizer and model-shape search report:

```bash
python3 scripts/model_design_search_agent.py
```

Read the corpus, cleaning, tokenizer, architecture, and optimizer recipe:

```text
docs/corpus_cleaning_and_0_2b_recipe.md
```

Prepare cleaned local JSONL shards:

```bash
python scripts/prepare_clean_corpus.py --max-docs 1000000 --overwrite
```

Train from cleaned shards:

```bash
PYTHONPATH=src python -m jepa_slm.train --config configs/train_clean_local.yaml
```

### One-command training

`scripts/run_training.sh` is the single entry point: it checks the env, prepares
cleaned shards **only if the config needs local data and they are missing**, then
launches distributed training. The default 8-GPU config streams its corpus, so no
data step is required.

```bash
# 8x H20, streaming corpus — no local data prep needed:
bash scripts/run_training.sh                      # = configs/train_h20_8gpu.yaml

# A config that reads local cleaned shards — auto-cleans them first if absent:
bash scripts/run_training.sh configs/train_h20_4gpu.yaml

# Preview what it will do without running anything:
DRY_RUN=1 bash scripts/run_training.sh
```

Generate the smaller HomeBench-style command model verification report:

```bash
python3 scripts/small_model_verification_agent.py
```

Generate the activation and position-encoding recommendation report:

```bash
python3 scripts/activation_position_agent.py
```

Check an 8x H20 node:

```bash
python3 scripts/h20_sanity_check.py
```

Launch 4x H20 DDP training (pins `CUDA_VISIBLE_DEVICES=0,1,2,3`, `nproc=4`):

```bash
bash scripts/launch_h20_4gpu.sh
```

Stop that run safely:

```bash
touch outputs/jepa-slm-h20-4gpu/STOP
```

### Tight per-GPU memory (e.g. ~6 GiB on a shared node)

`configs/train_h20_4gpu_6gb.yaml` trains the full 0.2B model on GPUs 0-3 inside a
~6 GiB-per-GPU slice. It stores weights + AdamW state in **bf16** (`param_dtype:
bf16`, halving the static optimizer footprint), uses micro-batch 1 with
gradient accumulation, keeps gradient checkpointing on, disables `torch.compile`,
and **streams** fineweb-edu (no local data prep). Estimated peak ~3.5-3.7 GiB/GPU.

```bash
bash scripts/launch_h20_4gpu.sh configs/train_h20_4gpu_6gb.yaml
# stop it safely:
touch outputs/jepa-slm-h20-4gpu-6gb/STOP
```

bf16 weight storage trades a little numerical precision for the memory win; once
you have the full 96 GiB H20s back, use the full-memory config below instead.

### Full memory (~96 GiB/GPU), large corpus, choose GPUs

`configs/train_h20_4gpu_96gb.yaml` is the full-throughput config: fp32 master
weights, micro-batch 128 (global batch 512), `torch.compile` on, gradient
checkpointing off (peak ~34 GiB/GPU), and it **streams the full fineweb-edu**
(`subset: null`) for a large token budget — `max_steps: 1,150,000` ≈ **~301 B
source tokens** (well-trained 0.2B; cf. SmolLM-360M @ 600 B). It pins GPUs via
`runtime.cuda_visible_devices: "4,5,6,7"`, and the launcher derives `nproc` from
that list:

```bash
bash scripts/launch_h20_4gpu.sh configs/train_h20_4gpu_96gb.yaml   # uses GPUs 4-7
# override the GPUs ad hoc:
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/launch_h20_4gpu.sh configs/train_h20_4gpu_96gb.yaml
# stop it safely:
touch outputs/jepa-slm-h20-4gpu-96gb/STOP
```

The cosine LR anneals to 0 at `max_steps`, so set the full token budget up front
(raise `max_steps` for more — still < 1 epoch of fineweb-edu).

Resume a stopped 4x H20 run:

```bash
bash scripts/launch_h20_4gpu.sh configs/train_h20_4gpu.yaml outputs/jepa-slm-h20-4gpu/step-00000250
```

Run a one-step offline smoke train:

```bash
PYTHONPATH=src python -m jepa_slm.train --config configs/train_tiny_smoke.yaml
```

Export the repo-ready 0.2B JEPA-T5 build config:

```bash
python scripts/build_0_2b_manifest.py
```

Run the predictor-context ablation (legacy gathered-only predictor) by setting:

```yaml
jepa:
  predictor_full_context: false
```

The default (`true`) lets the predictor attend over the full contextualized
student sequence with a learned mask marker, matching the I-JEPA/data2vec
contract. See `docs/optimization_changes.md` for the full list of training
optimizations and the audit findings they resolve.

## Main Recommendation

Architecture:

- `d_model=768`
- `encoder_layers=10`
- `decoder_layers=10`
- `d_ff=3072`
- `attention_heads=12`
- JEPA predictor: `512 width x 3 layers`
- EMA target: encoder-only

Smaller command-focused option:

- `d_model=512`
- `encoder_layers=12`
- `decoder_layers=10`
- `d_ff=2048`
- `attention_heads=8`
- JEPA predictor: `384 width x 2 layers`
- Trainable params: about `100M`

Objective:

```text
L_total = L_CE + lambda_jepa(t) * L_JEPA
```

Dataset:

- Pilot: `HuggingFaceFW/fineweb-edu`, subset `sample-10BT`
- Main: `HuggingFaceTB/smollm-corpus`
- Scale-up: `HuggingFaceFW/fineweb-edu`, subset `sample-100BT` or larger

## Reports

- `reports/jepa_verification_report.md`
- `reports/jepa_slm_research_report.md`
- `reports/jepa_slm_research_ranking.json`
- `reports/dataset_recommendation.md`
- `reports/tokenizer_recommendation.md`
- `reports/model_design_search_report.md`
- `reports/small_model_verification_report.md`
- `reports/activation_position_report.md`

## Training Pipeline

See `docs/training_pipeline.md` for the executable CE+JEPA trainer, smoke test,
and current implementation boundaries.

## Configs

- `configs/model_0_2b.yaml`
- `configs/model_0_2b_swiglu_ablation.yaml`
- `configs/model_homebench_0_1b.yaml`
- `configs/datasets.yaml`
- `configs/tokenizer.yaml`
- `configs/train_tiny_smoke.yaml`
- `configs/train_h20_8gpu.yaml`
- `configs/train_h20_4gpu.yaml`
- `configs/train_clean_local.yaml`

## Tokenizer

Recommended tokenizer:

```text
T5-style SentencePiece Unigram
32000 base SentencePiece slots + 100 sentinel tokens
32100 active tokenizer IDs
32128 model embedding rows
```

Train it with:

```bash
python scripts/train_sentencepiece_tokenizer.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --subset sample-10BT \
  --output-dir artifacts/tokenizer \
  --vocab-size 32000 \
  --extra-ids 100
```

## H20 Environment (CUDA 12.4)

`requirements.txt` is the complete runtime package set, pinned for the CUDA 12.4
H20 stack. Two supported install paths:

**conda (recommended on the GPU node — pulls CUDA 12.4 + NCCL via conda):**

```bash
bash scripts/install_h20_env.sh        # creates the 'jepa-h20' env from envs/jepa-h20-cu124.yml
conda activate jepa-h20
python scripts/h20_sanity_check.py
```

**pure pip into an existing CUDA 12.4 env:**

```bash
pip install -r requirements.txt        # the cu124 extra-index resolves torch to the CUDA 12.4 wheels
python scripts/h20_sanity_check.py
```

The trainer is plain torch DDP + AdamW, so `flash-attn`, `transformer-engine`,
and `deepspeed` are **not** required (T5's relative-position bias is incompatible
with FlashAttention-2; the trainer uses sdpa/eager attention). The first two
remain available for unrelated experiments:

```bash
python -m pip install --no-build-isolation -r envs/requirements-h20-optional.txt
```

Launch the 8-GPU training scaffold:

```bash
bash scripts/launch_h20_8gpu.sh configs/train_h20_8gpu.yaml
```

The training entrypoint now enforces the intended contract:

```text
clean source -> EMA encoder -> latent target
masked source -> student encoder -> JEPA predictor -> latent loss
masked source -> encoder-decoder -> token CE loss
```

Decoded tokens are never fed back into JEPA target construction.

See `docs/h20_8gpu_training.md` for the utilization plan.

## Useful Knobs

```bash
python3 scripts/jepa_verifier_agent.py \
  --target-params 200000000 \
  --vram-gb 24 \
  --source-len 512 \
  --target-len 256 \
  --micro-batch 8
```

The verifier estimates parameter count, JEPA predictor overhead, EMA target storage, rough VRAM, and whether the proposed training objective still supports generation.
