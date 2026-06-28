# Training Optimization Pass

This documents the optimization/correctness fixes applied to the executable
trainer (`src/jepa_slm/`) and configs, and how the JEPA and T5 components are
combined. Each item maps to a finding from the optimization audit.

## How JEPA and T5 are combined (best-of-both)

| Component | Source | Role |
|-----------|--------|------|
| Encoder-decoder + span corruption + relative-position bias | T5 / UL2 | Generative contract: decoder reconstructs target text from the masked source. Token cross-entropy stays primary. |
| Gated-GELU FFN, tied embeddings, RMSNorm | T5 v1.1 / UL2 | Backbone capacity. (True SwiGLU is not native to T5; `swiglu`/`geglu` map to gated-gelu with a logged note.) |
| EMA target encoder (stop-gradient) | data2vec / I-JEPA | Stable latent targets from the *clean* source; never sees decoder output. |
| Top-k layer-averaged, affine-free normalized targets | data2vec | Richer, collapse-resistant targets. |
| Full-context predictor with a learned mask marker | I-JEPA | Predicts masked-target latents by attending over visible context — not just refining masked slots. |
| Smooth-L1 latent loss + 3-phase λ schedule | data2vec + UL2 | Auxiliary signal that ramps up, plateaus, then decays for a CE-polish finish. |
| Optional VICReg variance/covariance penalty + per-dim std monitor | VICReg | Explicit anti-collapse guard on top of EMA + target normalization. |

The training contract per step:

```text
clean source   -> EMA encoder      -> top-k mean -> norm -> latent target (stop-grad)
masked source  -> student encoder  -> predictor (full context + mask marker) -> latent loss (masked positions)
masked source  -> encoder-decoder  -> token cross-entropy (primary objective)
L = L_CE + lambda_jepa(t) * L_JEPA + vicreg
```

## Correctness fixes (training could not run / silently wrong)

- **Step-0 logging crash** — the loop read `output.cross_attention_jepa_loss`,
  a field that never existed (`AttributeError` on the first logged step). Removed;
  logging now reports `vicreg_loss`, `encoder_std`, `predictor_std`. A smoke test
  (`tests/test_training_smoke.py`) now runs past the first log step so this class
  of bug cannot regress.
- **No per-rank data sharding** — every DDP rank streamed identical data.
  `TextStreamDataset` now shards by `(rank, world_size)` and by DataLoader worker
  (`split_dataset_by_node` for HF streaming, index-stride otherwise).
- **Gradient-accumulation boundary** — the micro-step counter was keyed on
  `batch_index`, which resets each epoch. Replaced with a persistent counter.
- **Incomplete resume** — checkpoints now persist RNG state, GradScaler state, and
  `samples_consumed`; on resume the stream skips already-consumed examples instead
  of replaying the head of the corpus.
- **fp16 divergence** — `torch.autocast` was used with no `GradScaler`. A
  `GradScaler` is now enabled automatically when `precision: fp16`.

## Throughput / efficiency fixes

- **DataLoader workers** — `num_workers`/`prefetch_factor`/`persistent_workers`
  from the `performance:` block are now honored (were hardcoded to 0 / unparsed).
- **Fast tokenizer** — `tokenizer_use_fast: true` uses the Rust tokenizer.
- **Dynamic padding** — pad to the longest member of a batch (multiple of 8)
  instead of always padding to `source_length`/`target_length`.
- **DDP gradient sync** — `model.no_sync()` skips the all-reduce on non-final
  accumulation micro-steps; `static_graph`/`find_unused_parameters`/
  `gradient_as_bucket_view` come from the `distributed:` block.
- **torch.compile** — compiled with `dynamic=True`, and the JEPA loss is computed
  mask-weighted over the full (static-shaped) sequence, removing the per-batch
  recompilation churn caused by the old variable-width gather.
- **TF32 / matmul precision** — `allow_tf32` and `matmul_precision` are applied at
  startup.
- **Non-blocking H2D copies** + CUDA-gated `pin_memory`.
- **`empty_cache_steps`** is honored when > 0.

## Precision (bf16 vs fp16 vs int)

Use **bf16** on H20 (Hopper). It has the same exponent range as fp32, so no loss
scaling is needed, with full Tensor Core throughput. **fp16** has a narrow range,
requires a GradScaler (now present), and offers no speed gain on Hopper — only
useful on pre-Ampere GPUs. **Integer formats (int8/int4)** are for *quantized
inference* after training, not for the forward/backward of pretraining. All
configs use bf16; the trainer prints a recommendation if `precision: fp16` is set
on a bf16-capable GPU.

## Divergence guard ("stop loss")

`runtime.abort_on_nonfinite` (default true), `runtime.max_loss` (default 0 = off),
and `runtime.divergence_patience` (default 5) add a safety stop: an optimizer step
whose loss is NaN/Inf — or above `max_loss` when set — is **discarded** (grads
zeroed, no update applied, so a single NaN cannot permanently corrupt bf16
weights). After `divergence_patience` consecutive bad steps the run stops
gracefully and saves the last *good* checkpoint. The bad-loss flag is reduced
across DDP ranks so all ranks skip/stop together.

## Optimizer / schedule fixes

- **Weight-decay param grouping** — decay applies only to ≥2-D matmul weights;
  biases, LayerNorm/RMSNorm gains, relative-position-bias tables, embeddings, and
  the predictor mask marker are excluded (GPT/LLaMA/T5x convention).
- **3-phase JEPA λ schedule** (`jepa_lambda_schedule`) — warmup → plateau →
  final-phase decay to `jepa_final_phase_weight` (the documented CE-polish phase).
- **Token budget** — `train_h20_8gpu.yaml` `max_steps` raised from `1000`
  (~0.4B tokens, ~10x under Chinchilla) to `50000` (~19.6B tokens).

## JEPA methodology fixes

- **Predictor context** — now attends over the full student sequence with a
  learned mask marker (was: only the gathered masked vectors).
- **Collapse guard** — optional VICReg variance/covariance penalty plus a
  per-dimension std monitor (the old monitor was a single global scalar that could
  not detect low-rank collapse).

## Removed dead config

- `configs/deepspeed_h20_zero1.json` (never wired; DDP is correct for 0.2B).
- `batch_autotune` block (unimplemented).
- `cross_attention_jepa_weight` knob (unwired); the ablation is now
  `predictor_full_context: false`.
- `attention_kernel: flash_attention_2_*` now resolves through a real fallback
  (T5's relative-position bias is incompatible with FlashAttention-2 and SDPA, so
  it falls back to eager rather than being silently ignored).
