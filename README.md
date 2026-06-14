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

Check an 8x H20 node:

```bash
python3 scripts/h20_sanity_check.py
```

## Main Recommendation

Architecture:

- `d_model=768`
- `encoder_layers=10`
- `decoder_layers=10`
- `d_ff=3072`
- `attention_heads=12`
- JEPA predictor: `512 width x 3 layers`
- EMA target: encoder-only

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

## Configs

- `configs/model_0_2b.yaml`
- `configs/datasets.yaml`
- `configs/tokenizer.yaml`
- `configs/train_h20_8gpu.yaml`
- `configs/deepspeed_h20_zero1.json`

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

## 8x H20 Environment

Create the training environment:

```bash
conda env create -f envs/jepa-h20-cu124.yml
conda activate jepa-h20
python -m pip install --no-build-isolation -r envs/requirements-h20-optional.txt
python scripts/h20_sanity_check.py
```

Launch the 8-GPU training scaffold:

```bash
bash scripts/launch_h20_8gpu.sh configs/train_h20_8gpu.yaml
```

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
