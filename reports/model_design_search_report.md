# JEPA-SLM Model Design Search

Generated: 2026-06-14 20:54

## Final Recommendation

Use this tokenizer:

```text
T5-style SentencePiece Unigram
base SentencePiece vocabulary: 32000
T5 sentinel tokens: 100
active tokenizer IDs: 32100
model embedding rows: 32128
```

Use this model shape:

```yaml
d_model: 768
encoder_layers: 10
decoder_layers: 10
d_ff: 3072
attention_heads: 12
attention_head_dim: 64
vocab_size: 32128
trainable_params: 200.3M
```

This is the best practical point for a 0.2B JEPA-augmented encoder-decoder because it keeps the clean T5-base width/head/FFN ratios while reducing depth from 12+12 to fit the JEPA predictor and EMA encoder budget.

## Tokenizer Search

| tokenizer | embedding vocab | score | rationale |
| --- | --- | --- | --- |
| T5-style SentencePiece Unigram | 32128 | 96.0 | Best fit for span corruption because sentinels are first-class tokens and the model remains T5-compatible. |
| Byte-level BPE with sentinels | 32896 | 84.0 | Very robust for arbitrary bytes and code-heavy text. |
| SmolLM-style larger BPE | 49280 | 80.0 | Strong precedent for decoder-only small LMs trained on SmolLM-Corpus. |
| Byte/character tokenizer | 384 | 58.0 | No unknown tokens and simple multilingual behavior. |

Decision: choose the T5-style SentencePiece Unigram tokenizer. The larger SmolLM-style 49152-token BPE is attractive for decoder-only small LMs, but at `d_model=768` it adds about 13M tied embedding parameters compared with `32128`. For this encoder-decoder, that steals too much capacity from layers.

## Architecture Search

| d_model | enc | dec | d_ff | heads | head_dim | params | score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 768 | 10 | 10 | 3072 | 12 | 64 | 200.3M | 99.9 |
| 768 | 12 | 8 | 3072 | 12 | 64 | 195.6M | 98.5 |
| 768 | 12 | 10 | 3072 | 12 | 64 | 214.5M | 94.9 |
| 768 | 10 | 10 | 3072 | 8 | 96 | 200.3M | 94.4 |
| 768 | 12 | 10 | 2816 | 12 | 64 | 205.8M | 94.0 |
| 768 | 14 | 8 | 3072 | 12 | 64 | 209.8M | 93.6 |
| 768 | 10 | 8 | 3072 | 12 | 64 | 181.4M | 93.5 |
| 768 | 10 | 10 | 2816 | 12 | 64 | 192.4M | 93.4 |
| 768 | 12 | 8 | 3072 | 8 | 96 | 195.6M | 93.0 |
| 768 | 14 | 8 | 2816 | 12 | 64 | 201.1M | 92.6 |

Decision: choose `d_model=768`, `encoder_layers=10`, `decoder_layers=10`, `d_ff=3072`, `attention_heads=12`.

Close alternative: `d_model=768`, `encoder_layers=12`, `decoder_layers=8`, `d_ff=3072`, `attention_heads=12`. Use it if encoder representations matter more than generation quality. The balanced 10/10 split is safer for a first full run because the decoder still has enough capacity for span-corruption and seq2seq generation.

## Evidence Anchors

- Hugging Face T5 config defaults and public T5-base use `vocab_size=32128`.
- T5-base uses `d_model=768`, `d_ff=3072`, and `num_heads=12`.
- T5 tokenizer APIs expose `extra_ids=100` sentinel tokens and are based on Unigram.
- SentencePiece trains subword models directly from raw sentences, which fits FineWeb-Edu and SmolLM-Corpus streaming.
- SmolLM uses a 49152-token tokenizer for decoder-only SLMs and reports strong results, but its larger vocab is a worse parameter trade for this 0.2B encoder-decoder.
- MobileLLM shows that architecture choices matter at sub-billion scale and favors efficient depth, embedding sharing, and attention efficiency.

## Practical Training Note

Keep this as the first serious training run:

```text
T5/UL2 span corruption CE + encoder-side JEPA auxiliary loss
lambda_jepa warmup to 0.25
tied embeddings
encoder-only EMA target
```

If the CE-only baseline beats CE+JEPA after equal tokens, test the 12-encoder/8-decoder alternative before changing tokenizer size.
