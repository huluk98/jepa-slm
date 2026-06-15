# JEPA Encoder-Decoder Activation And Position Report

Generated: 2026-06-15 11:33

## Recommendation

Use **SwiGLU** for a custom implementation, but keep the first positional baseline as **T5 relative position bias**.

RoPE is **not** a tokenizer choice. Keep the tokenizer as T5-style SentencePiece Unigram with `vocab_size: 32128`; evaluate RoPE only as a positional-attention ablation.

Recommended activation settings:

```yaml
# 0.2B shape
ffn_activation: swiglu
d_ff: 2048

# 0.1B HomeBench-style shape
ffn_activation: swiglu
d_ff: 1344
```

If you stay with the current non-gated FFN, use `GELU` and keep `d_ff: 3072` for the 0.2B model and `d_ff: 2048` for the 0.1B model.

## Activation Ranking

| activation | score | parameter rule | recommendation |
| --- | --- | --- | --- |
| SwiGLU | 94.0 | Use about 2/3 of the non-gated d_ff to keep parameter count stable. | Use for the custom JEPA-SLM implementation, with adjusted d_ff. |
| GEGLU | 90.0 | Same gated parameter rule as SwiGLU. | Use if T5 compatibility matters more than modern LLM convention. |
| GELU | 82.0 | Non-gated FFN; current d_ff values are already sized for it. | Keep as the baseline if you want minimal implementation risk. |
| Plain SiLU | 68.0 | Non-gated FFN; same parameter shape as GELU. | Do not use as the main FFN activation unless testing a control ablation. |

### SwiGLU

Pros:

- Best modern default for custom LLM-style Transformer blocks.
- Gating improves FFN expressivity and token-wise feature selection.
- SiLU's smoothness is useful inside the gate without using plain SiLU everywhere.
- Works well with pre-norm/RMSNorm-style small models.

Cons:

- Adds a second up projection, so keeping the same d_ff increases parameters and FLOPs.
- Slightly less plug-and-play with vanilla Hugging Face T5 code than GELU/GEGLU.
- Can produce larger activations than GELU, so BF16 training should keep clipping/scale monitoring.

### GEGLU

Pros:

- Strong T5-family choice; easy to justify for encoder-decoder models.
- Gated like SwiGLU while staying closer to T5.1.1 conventions.
- Good compatibility story if starting from T5-style code.

Cons:

- Slightly less common than SwiGLU in newer decoder-only LLM stacks.
- Still needs d_ff reduction for fair parameter matching.

### GELU

Pros:

- Simple, stable, and widely implemented.
- Good baseline for ablations because parameter accounting is straightforward.
- Lower activation outlier risk than gated SiLU variants.

Cons:

- No gate, so it is usually less expressive per layer than GEGLU/SwiGLU.
- Less aligned with current small-LLM architecture practice.

### Plain SiLU

Pros:

- Smooth and cheap.
- Useful as the nonlinear component inside SwiGLU.

Cons:

- Plain SiLU is not the part that usually gives modern LLM gains; the gate is.
- Less benchmark precedent than GELU or gated GLU variants for encoder-decoder pretraining.

## Position Encoding Ranking

| position method | score | recommendation |
| --- | --- | --- |
| T5 relative position bias | 92.0 | Use for the first serious JEPA+encoder-decoder run. |
| RoPE in encoder and decoder self-attention only | 86.0 | Run as the RoPE ablation after the relative-bias baseline. |
| Full RoPE including cross-attention | 72.0 | Research ablation only, not first-run architecture. |
| Learned absolute positions | 58.0 | Avoid unless you need the simplest possible debugging baseline. |

## RoPE Decision

Use relative position bias first. It is simpler, T5-native, and safer for encoder-decoder cross-attention. Then run a RoPE ablation in encoder self-attention and decoder self-attention only.

Do not call RoPE a tokenizer improvement. The tokenizer decides token IDs; RoPE changes how attention represents token positions after tokenization.

## JEPA-Specific Notes

- The JEPA target encoder and student encoder should use the same positional scheme.
- Relative bias is safer for masked-span target alignment because it is already standard in T5-style bidirectional encoders.
- SwiGLU/GEGLU may improve the semantic quality of encoder states, but the latent loss can amplify activation outliers. Track representation norm, variance, and BF16 overflow/NaN counters.
- If SwiGLU destabilizes early training, fall back to GEGLU or GELU before changing the tokenizer.

## First Ablation Matrix

| run | activation | position | d_ff 0.2B | d_ff 0.1B | purpose |
| --- | --- | --- | --- | --- | --- |
| A | GELU | relative bias | 3072 | 2048 | compatibility baseline |
| B | SwiGLU | relative bias | 2048 | 1344 | recommended custom run |
| C | GEGLU | relative bias | 2048 | 1344 | T5-family gated run |
| D | SwiGLU | RoPE self-attn only | 2048 | 1344 | position ablation |

## Evidence Anchors

- GLU variant work found that gated variants such as GEGLU/SwiGLU improve Transformer FFN quality over ReLU/GELU-style baselines.
- PaLM-style modern LLM stacks use SwiGLU, which is a strong precedent for custom dense LLM blocks.
- T5-small and T5-base configs use encoder-decoder relative attention buckets, which supports relative bias as the compatibility baseline.
- RoFormer introduced RoPE as a rotary positional method that encodes relative position behavior in attention, but that is separate from tokenization.
