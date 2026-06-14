# Tokenizer Training

## Recommendation

Use a **T5-style SentencePiece Unigram tokenizer**.

The project should use:

```text
32000 base SentencePiece vocabulary slots
100 sentinel tokens: <extra_id_0> ... <extra_id_99>
32100 active tokenizer IDs
32128 model embedding rows
```

The `32128` value follows the common T5-style model config pattern: a 32k base SentencePiece model plus sentinels, with model embedding rows padded to a multiple of 128. The 32k SentencePiece value includes SentencePiece meta pieces such as `<pad>`, `</s>`, `<unk>`, and byte-fallback pieces when byte fallback is enabled.

## Why Not A Plain 32k Tokenizer

The model is trained with span corruption and JEPA over masked source positions. Span corruption needs sentinel tokens so the decoder can emit a compact target like:

```text
<extra_id_0> missing span A <extra_id_1> missing span B </s>
```

Without sentinels, the tokenizer would be adequate for normal CE training but incomplete for T5/UL2 denoising.

## Train Command

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

The output directory will contain:

- `jepa_slm_spm.model`
- `jepa_slm_spm.vocab`
- `tokenizer_metadata.json`
- `tokenizer_corpus.txt`

Do not commit the corpus or tokenizer artifacts unless intentionally versioning a small tokenizer release.

Load the tokenizer in a T5-style stack with 100 extra sentinel IDs, then keep model embeddings at 32128 rows:

```python
from transformers import T5Tokenizer

tokenizer = T5Tokenizer(
    vocab_file="artifacts/tokenizer/jepa_slm_spm.model",
    extra_ids=100,
)
```

## Main-Corpus Tokenizer

For the final tokenizer, sample from the same distribution as training:

| subset | share |
| --- | --- |
| `fineweb-edu-dedup` | 80% |
| `cosmopedia-v2` | 15% |
| `python-edu` | 5% |

The pilot tokenizer from FineWeb-Edu `sample-10BT` is good enough for ablations, but the final tokenizer should see the SmolLM-Corpus mixture if Python/code tutorials remain in the training data.

## Validation

- Sentinel tokens round-trip exactly.
- Unknown-token rate is near zero with byte fallback.
- Average sequence length is close to or better than a baseline T5 tokenizer.
- The model config uses `vocab_size: 32128`.
- The tokenizer metadata records `base_sentencepiece_vocab_size=32000`, `extra_sentinel_tokens=100`, `active_tokenizer_size=32100`, and `embedding_vocab_size=32128`.
