# Smaller JEPA-SLM Verification For HomeBench-Style Commands

Generated: 2026-06-14 21:02

## Verdict

Yes, the 0.2B design can be made smaller for command-heavy smart-home workloads. The best first smaller target is **about 100M trainable parameters**, not 60M.

Recommended small shape:

```yaml
d_model: 512
encoder_layers: 12
decoder_layers: 10
d_ff: 2048
attention_heads: 8
attention_head_dim: 64
vocab_size: 32128
predictor_width: 384
predictor_layers: 2
trainable_params: 100.3M
stored_params_with_ema: 138.1M
estimated_peak_vram_per_gpu: 4.3 GB
```

## Target Sweep

| tier | d_model | enc | dec | d_ff | heads | head_dim | params | score | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 160M safer general small | 640 | 14 | 10 | 2560 | 10 | 64 | 160.6M | 89.8 | feasible |
| 125M safe command model | 576 | 12 | 10 | 2304 | 9 | 64 | 124.9M | 90.0 | feasible |
| 100M recommended command model | 512 | 12 | 10 | 2048 | 8 | 64 | 100.3M | 94.8 | recommended |
| 80M aggressive edge model | 512 | 8 | 8 | 2048 | 8 | 64 | 79.2M | 91.3 | possible with cautions |
| 60M minimum parser | 384 | 12 | 10 | 1536 | 6 | 64 | 60.0M | 86.0 | not recommended as first run |

## Checks For Recommended 100M Shape

| check | status | detail |
| --- | --- | --- |
| parameter budget | OK | 100.3M trainable vs target 100.0M |
| predictor overhead | OK | predictor is 4.1% of the base encoder-decoder |
| generation objective | OK | token CE is retained, so the model remains generative |
| collapse controls | OK | EMA targets plus normalized top-layer latent prediction are configured |
| encoder emphasis | OK | encoder capacity is at least decoder capacity, which suits representation pretraining |
| rough VRAM | OK | estimated peak 4.3 GB on a 24.0 GB budget |
| EMA target scope | OK | EMA target is encoder-only and has no optimizer state |

## Why 100M Is The Right Small Point

- It keeps the T5-small width family: `d_model=512`, `d_ff=2048`, `8` heads, and `64`-dim heads.
- It adds depth over T5-small so the encoder has enough room for JEPA representation learning and the decoder still has 10 layers for structured generation.
- It keeps the existing `32128` tokenizer instead of shrinking vocabulary. For command tasks, losing sentinel/tokenizer compatibility is not worth saving a few embedding rows.
- It reduces trainable params from about 200M to about 100M while keeping the predictor overhead under control.

## HomeBench Risk Assessment

HomeBench-style tasks are not just normal single-device commands. The hard slice is invalid multi-device instructions, where the model must reject, repair, or split commands correctly. For that slice, architecture size is less important than the output contract:

1. Generate a structured action list or an explicit rejection, never free text.
2. Use constrained decoding against the device/action schema.
3. Add a deterministic verifier after the model for room, device, capability, and state checks.
4. Train separate slices for valid single-device, invalid single-device, valid multi-device, and invalid multi-device instructions.
5. Gate release on exact-match action success and invalid-instruction rejection rate.

## Smaller Options

- **125M** is safer if invalid multi-device handling is the product-critical metric.
- **100M** is the recommended first small run.
- **80M** is plausible for a narrow command grammar with constrained decoding.
- **60M** is a parser-sized model, not a robust smart-home assistant by itself.

## Relevant Work Anchors

- HomeBench shows that smart-home evaluation must include valid and invalid instructions across single and multiple devices.
- HomeBench reports that invalid multi-device instructions remain difficult even for strong LLMs, so rejection and verification logic are mandatory.
- T5-small establishes that `d_model=512`, `d_ff=2048`, and `8` attention heads are a proven small encoder-decoder shape.
- MobileLLM argues that sub-billion models benefit from careful architecture design rather than just naive shrinking.
- Newer smart-home benchmarks emphasize executable or verifiable environments, which supports pairing the 100M model with a deterministic command verifier.
