# Corpus, Cleaning, And 0.2B Training Recipe

## Material Passport

- Origin: ARS experiment plan
- Verification status: analyzed
- Scope: first serious JEPA + T5-style encoder-decoder run

## Corpus

Use the existing staged path:

1. Tokenizer sample: `HuggingFaceFW/fineweb-edu`, subset `sample-10BT`.
2. Pilot pretrain: `FineWeb-Edu sample-10BT`.
3. Main pretrain: `HuggingFaceTB/smollm-corpus`-style mixture.
4. Polish: small teacher-distilled instruction set.

Recommended main mix:

```yaml
fineweb-edu-dedup: 0.55
cosmopedia-v2: 0.20
math: 0.10
code: 0.10
instruction-distill: 0.05
```

Token budget:

- Pilot: `10B` tokens.
- Main useful run: `50B` tokens.
- Stretch run: `100B` tokens, only if validation improves.

## Cleaning

Use already-filtered open corpora where possible. The executable trainer now applies cheap document hygiene:

```yaml
normalize_text: true
min_chars: 200
max_chars: 20000
```

Process:

1. HTML unescape.
2. Unicode NFKC normalization.
3. Control-character removal.
4. Whitespace collapse.
5. Drop documents shorter than `200` characters.
6. Truncate documents longer than `20000` characters.

Do not build a custom Common Crawl cleaner until the prepared corpora fail. FineWeb-Edu and SmolLM-Corpus already buy most of the quality lift.

Current local cleaned starter corpus:

```text
data/clean/fineweb-edu-sample10bt-100k/clean-*.jsonl
100000 documents
426194823 cleaned characters
```

The cleaner writes `*.jsonl.tmp` files first and only renames complete shards to
`*.jsonl`, so an interrupted clean does not feed broken rows to training.
The tracked checksum manifest is `reports/clean_corpus_manifest.json`.

Create local cleaned shards:

```bash
python scripts/prepare_clean_corpus.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --subset sample-10BT \
  --output-dir data/clean/fineweb-edu-sample10bt \
  --max-docs 1000000 \
  --min-chars 200 \
  --max-chars 20000 \
  --overwrite
```

Then train from those shards:

```bash
PYTHONPATH=src python -m jepa_slm.train --config configs/train_clean_local.yaml
```

For a full pilot, increase `--max-docs` or pass `--max-docs 0` to stream until the source ends.

## Tokenizer

Use:

```yaml
train_base_vocab_size: 32000
extra_sentinel_tokens: 100
active_tokenizer_size: 32100
embedding_vocab_size: 32128
```

Why this is the correct size:

- T5 span corruption needs sentinel tokens.
- `32128` keeps T5-compatible embedding rows.
- Larger 49k vocabularies steal too many parameters from a 0.2B encoder-decoder.

## Model Shape

Use the current 0.2B config:

```yaml
d_model: 768
encoder_layers: 10
decoder_layers: 10
d_ff: 3072
attention_heads: 12
head_dim: 64
vocab_size: 32128
```

This is the best first point because it keeps T5-base width/head geometry while reducing depth enough to leave budget for JEPA predictor parameters.

Close alternative:

```yaml
encoder_layers: 12
decoder_layers: 8
```

Use the alternative only if encoder probes matter more than generation quality.

## Optimizer

Use AdamW:

```yaml
learning_rate: 0.0004
betas: [0.9, 0.95]
weight_decay: 0.1
grad_clip_norm: 1.0
scheduler: cosine
warmup_fraction: 0.02
precision: bf16
```

JEPA schedule:

```yaml
jepa_weight_peak: 0.25
jepa_warmup_fraction: 0.10
jepa_final_phase_weight: 0.05
ema_tau_start: 0.99
ema_tau_end: 0.9995
```

For BF16, use `torch.autocast`; a GradScaler is only needed for FP16. The repo trainer already uses BF16 autocast and fused AdamW when CUDA is available.
