# Tokenizer Recommendation For JEPA-SLM

Generated: 2026-06-14 20:48

## Recommendation

Use **T5-style SentencePiece Unigram**.

Current repo status: the model config previously only assumed a numeric vocabulary size. The tokenizer should be made explicit as:

```text
32000 base SentencePiece Unigram vocabulary slots
+ 100 T5-style sentinel tokens: <extra_id_0> ... <extra_id_99>
= 32100 active tokenizer IDs
rounded/padded to 32128 model embedding rows
```

The recommended model config value is therefore:

```yaml
vocab_size: 32128
```

## Ranking

| candidate | score | vocab |
| --- | --- | --- |
| T5-style SentencePiece Unigram | 95.0 | 32000 base SPM slots + 100 sentinels; 32128 model embedding rows |
| Byte-level BPE | 82.0 | 32000-50000 merges |
| Reuse an existing T5 tokenizer | 78.0 | 32128 model rows in common HF configs |
| Byte/character model | 62.0 | 256 bytes plus specials |

## Why This Tokenizer

T5/UL2-style training depends on span corruption. The decoder target is not the whole original text; it is a sequence of sentinel tokens and missing spans. That means sentinel tokens are not optional implementation details. They are part of the objective.

SentencePiece Unigram is the closest fit because it trains from raw text, handles whitespace internally, and is the tokenizer family used by T5-style models. For this project, byte fallback is enabled so web/code/tutorial text does not collapse into `<unk>` when unusual characters appear.

The 32000 SentencePiece slots include SentencePiece meta pieces and, when enabled, byte-fallback pieces. The sentinel tokens are added by the T5 tokenizer wrapper rather than learned inside the SentencePiece model.

## How To Train

Pilot tokenizer training:

```bash
conda activate jepa-h20
python scripts/train_sentencepiece_tokenizer.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --subset sample-10BT \
  --text-field text \
  --output-dir artifacts/tokenizer \
  --vocab-size 32000 \
  --extra-ids 100 \
  --max-docs 5000000
```

Main tokenizer training should sample the same proportions as the main corpus:

```text
80% fineweb-edu-dedup
15% cosmopedia-v2
5% python-edu
```

## Validation Gates

- Unknown-token rate should be near zero when byte fallback is enabled.
- Median tokens per document should not inflate badly versus an existing T5 tokenizer.
- Sentinel IDs must round-trip exactly.
- `len(tokenizer)` should expose the active tokenizer size, while model embeddings should use `32128`.
- Re-run the parameter verifier after changing vocab size.

## Candidate Notes

### T5-style SentencePiece Unigram
Strengths:
- Native fit for T5/UL2 span corruption.
- Supports sentinel tokens for denoising targets.
- Works directly from raw text and is stable for encoder-decoder training.
- The 32128 embedding size matches common T5 configs.
Risks:
- Less code-specialized than a large byte-level BPE.
- Needs care around sentinel token ordering.

### Byte-level BPE
Strengths:
- Robust for arbitrary web/code bytes.
- Popular for decoder-only LMs.
Risks:
- Less natural for T5 sentinel-span corruption.
- Can produce longer sequences for some educational prose.

### Reuse an existing T5 tokenizer
Strengths:
- Fastest way to start ablations.
- Compatible with many seq2seq libraries.
Risks:
- Not trained on SmolLM-Corpus/FineWeb-Edu distribution.
- Less ideal if code/tutorial data is a serious part of the mixture.

### Byte/character model
Strengths:
- No unknown tokens.
- Simple and multilingual-friendly.
Risks:
- Much longer sequences at 0.2B scale.
- Higher attention cost hurts H20 throughput for this model.

