# JEPA-SLM Research Agent Report

Generated: 2026-06-14 20:38

## Bottom Line

The best small-language-model training architecture to add JEPA to is:

**UL2/T5 Encoder-Decoder + Encoder JEPA Auxiliary**

Use JEPA as an **encoder-side auxiliary latent prediction objective**, not as a replacement for token-level language modeling. In plain English: keep the normal seq2seq training that teaches the model to speak, and add a JEPA/data2vec-style loss that teaches the encoder to build better internal world-state representations of text.

Recommended 0.2B shape:

- Base: T5/UL2-style encoder-decoder.
- Width: `d_model=704-768`.
- Depth: `10-12` encoder layers and `8-10` decoder layers.
- FFN: GEGLU/SwiGLU with `d_ff` around `2.5x-4x d_model`.
- Attention: relative position bias or RoPE-style relative handling.
- Predictor: 2-3 narrow Transformer/MLP blocks, about `0.5x-0.7x d_model`.
- EMA target: encoder-only copy, no gradients, no optimizer state.
- Inference: discard EMA target and predictor.

## Candidate Ranking

| candidate | weighted score | generation | latent fit | efficiency | theory |
| --- | --- | --- | --- | --- | --- |
| UL2/T5 Encoder-Decoder + Encoder JEPA Auxiliary | 93.6 | 5 | 5 | 4 | 5 |
| Plain T5 Span-Corruption + Encoder JEPA Auxiliary | 87.8 | 5 | 4.5 | 4 | 4 |
| BART Denoising + Encoder JEPA Auxiliary | 82.1 | 5 | 4 | 3.5 | 4 |
| Decoder-Only SLM + Latent Self-Distillation Head | 68.9 | 5 | 2.5 | 4 | 2.5 |
| Pure Text JEPA Encoder, Decoder Added Later | 63.4 | 1.5 | 5 | 3 | 4 |

## Mathematical Objective

Let `x` be the source sequence, `x_tilde = mask(x)` the corrupted source, `y` the target text, `f_theta` the student encoder, `f_bar` the EMA target encoder, `g_phi` the predictor, and `D_theta` the decoder.

The standard encoder-decoder language-modeling term is:

```text
L_CE(theta) = - sum_t log p_theta(y_t | y_<t, f_theta(x_tilde))
```

The JEPA/data2vec term is:

```text
z_i = stopgrad(norm(mean_top_k(f_bar(x))_i))
h = f_theta(x_tilde)
z_hat_i = norm(g_phi(h, pos_i))

L_JEPA(theta, phi) = (1 / |M|) sum_{i in M} || z_hat_i - z_i ||_2^2
```

The EMA target update is:

```text
bar_theta <- tau * bar_theta + (1 - tau) * theta_encoder
```

The total objective:

```text
L_total = L_CE + lambda_jepa(t) * L_JEPA + beta * L_var + gamma * L_cov
```

In the default recipe, `L_var` and `L_cov` are first used as monitors. Add them as small penalties only if representation variance starts to collapse.

Variance monitor:

```text
L_var = (1 / d) sum_j max(0, sigma_min - std_batch_time(z_j))
```

Covariance monitor:

```text
L_cov = (1 / d) sum_{i != j} Cov(z)_ij^2
```

## Why This Should Improve A Small Model

### 1. Token CE is sparse supervision for the encoder

In a normal encoder-decoder, the encoder learns only through decoder token loss. That signal is useful, but it is filtered through the decoder's immediate need to predict surface tokens. At small scale, this encourages shortcuts.

The JEPA term adds a direct representation objective: from partial text, predict the clean full-context hidden state. That pushes the encoder toward contextual, paraphrase-stable, semantic features.

### 2. Latent prediction is a better target than raw reconstruction for representation quality

Raw reconstruction asks the model to recover every token detail. Latent prediction asks it to recover the hidden state that the teacher encoder builds from the clean input. The target is still grounded in the text, but it is less obsessed with surface form.

This is the data2vec/JEPAs core move: learn by predicting contextualized representations, not just vocabulary items or pixels.

### 3. The predictor creates a useful bottleneck

If the encoder directly had to match target states, it could contort its hidden space around the auxiliary task. A narrow predictor lets the encoder stay useful for the decoder while the predictor handles the latent matching geometry.

### 4. EMA makes the target stable

The target encoder changes slowly. That makes the latent target less noisy than using the current student as its own target. This is the same stability logic behind mean-teacher/self-distillation methods.

### 5. UL2-style objective mixing protects generation

Pure JEPA does not train the decoder to emit text. UL2/T5-style CE preserves generation, while JEPA improves the hidden states feeding the decoder. The best version is a staged mixture:

```text
phase 1: CE span corruption + JEPA, lambda_jepa ramp 0 -> 0.1
phase 2: CE span corruption + prefix/seq2seq tasks + JEPA, lambda_jepa 0.1 -> 0.25
phase 3: reduce JEPA to 0.05-0.15 or disable for final CE polish
```

## Best Training Loop

```python
for batch in train_loader:
    source_ids, labels = batch["source_ids"], batch["labels"]
    masked_source, masked_positions = span_mask(source_ids)

    with torch.no_grad():
        teacher_layers = ema_encoder(source_ids, output_hidden_states=True)
        target = mean_top_k(teacher_layers, k=4)
        target = layer_norm_no_affine(target)
        target = gather(target, masked_positions)

    student_hidden = model.encoder(masked_source)
    predicted = predictor(student_hidden, masked_positions)
    jepa_loss = mse_or_smooth_l1(l2_normalize(predicted), l2_normalize(target))

    logits = model(
        input_ids=masked_source,
        decoder_input_ids=shift_right(labels),
    ).logits
    ce_loss = cross_entropy(logits, labels)

    loss = ce_loss + lambda_jepa * jepa_loss
    loss.backward()
    optimizer.step()
    update_ema(ema_encoder, model.encoder, tau=ema_tau)
```

## Recommended Ablation

Run these before committing to full training:

1. `CE-only`: normal T5/UL2 span corruption.
2. `CE + JEPA-0.10`: latent loss ramped to 0.10.
3. `CE + JEPA-0.25`: latent loss ramped to 0.25.
4. Optional `CE + JEPA + var/cov`: only if diagnostics show collapse risk.

Success gates:

- Validation CE is no worse than 1-2 percent after warmup.
- Encoder representation variance remains healthy.
- Mean cosine similarity between unrelated examples does not approach 1.
- Encoder probes improve: retrieval, NLI, classification, or semantic similarity.
- Downstream seq2seq metrics match or beat CE-only at equal compute.

## Failure Modes

- **Pure JEPA for generation:** bad idea; it does not teach text emission.
- **Too much lambda:** can improve hidden matching while hurting token quality.
- **Oversized predictor:** the predictor becomes a second model and hides weak encoder learning.
- **Full encoder-decoder EMA:** wasteful at 0.2B; use encoder-only EMA.
- **No diagnostics:** falling JEPA loss can coexist with collapsed representations.

## Source Ledger

| key | year | evidence type | source |
| --- | --- | --- | --- |
| t5 | 2019 | large empirical architecture study | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) |
| bart | 2019 | seq2seq denoising pretraining | [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) |
| ul2 | 2022 | objective-mixture architecture study | [UL2: Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131) |
| data2vec | 2022 | cross-modal latent prediction | [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/abs/2202.03555) |
| data2vec2 | 2022 | efficiency improvement | [Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language](https://arxiv.org/abs/2212.07525) |
| ijepa | 2023 | canonical JEPA implementation | [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) |
| vicreg | 2021 | collapse prevention theory | [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906) |
| barlow | 2021 | redundancy-reduction objective | [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) |
| spactor | 2024 | small/efficient T5 hybrid objective | [SpacTor-T5: Pre-training T5 Models with Span Corruption and Replaced Token Detection](https://arxiv.org/abs/2401.13160) |
| varjepa | 2026 | probabilistic theory | [Var-JEPA: A Variational Formulation of the Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2603.20111) |

## Source Synthesis

- T5 and BART establish that encoder-decoder denoising is the reliable base for small generative seq2seq models.
- UL2 suggests objective mixing is better than betting on a single pretraining mode.
- data2vec is the strongest text-relevant precedent for predicting contextual latent representations from masked inputs.
- I-JEPA provides the architecture pattern: context encoder, EMA target encoder, predictor, and latent loss.
- VICReg and Barlow Twins explain the collapse problem and give variance/covariance diagnostics.
- SpacTor-T5 supports staged auxiliary objectives for efficient small-model pretraining.
- Var-JEPA connects JEPA to a latent-variable/ELBO interpretation, which supports treating JEPA as a principled latent prediction term rather than a bag of heuristics.
