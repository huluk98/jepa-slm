# JEPA Encoder-Decoder Verification Report

Generated: 2026-06-14 20:48

## Verdict

The verifier rates a JEPA-style auxiliary objective for a 200.0M encoder-decoder model as **feasible** under the current assumptions.

Best candidate: **d768-e10-d10-p512x3**

| field | value |
| --- | --- |
| d_model | 768 |
| encoder layers | 10 |
| decoder layers | 10 |
| d_ff | 3072 |
| attention heads | 12 |
| predictor | 512 width x 3 layers |
| trainable params | 200.3M |
| stored params incl. EMA | 271.2M |
| estimated peak VRAM | 6.2 GB |

## Candidate Loop Results

| candidate | trainable | encoder | decoder | predictor | VRAM | score | checks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d768-e10-d10-p512x3 | 200.3M | 70.9M | 94.5M | 10.2M | 6.2 GB | 99.9 | 7 OK / 0 WARN / 0 FAIL |
| d832-e12-d6-p512x3 | 203.4M | 99.8M | 66.5M | 10.3M | 6.4 GB | 98.6 | 7 OK / 0 WARN / 0 FAIL |
| d704-e12-d12-p512x2 | 196.4M | 71.5M | 95.3M | 7.0M | 6.2 GB | 98.6 | 7 OK / 0 WARN / 0 FAIL |
| d768-e12-d8-p512x3 | 195.6M | 85.1M | 75.6M | 10.2M | 6.2 GB | 98.2 | 7 OK / 0 WARN / 0 FAIL |
| d704-e14-d10-p512x2 | 192.5M | 83.4M | 79.4M | 7.0M | 6.2 GB | 97.0 | 7 OK / 0 WARN / 0 FAIL |
| d832-e8-d8-p512x3 | 192.3M | 66.5M | 88.7M | 10.3M | 5.9 GB | 96.9 | 7 OK / 0 WARN / 0 FAIL |
| d704-e14-d12-p512x2 | 208.4M | 83.4M | 95.3M | 7.0M | 6.5 GB | 96.7 | 7 OK / 0 WARN / 0 FAIL |
| d832-e10-d8-p512x3 | 208.9M | 83.2M | 88.7M | 10.3M | 6.4 GB | 96.4 | 7 OK / 0 WARN / 0 FAIL |

## Verification Checks For Recommended Candidate

| check | status | detail |
| --- | --- | --- |
| parameter budget | OK | 200.3M trainable vs target 200.0M |
| predictor overhead | OK | predictor is 5.4% of the base encoder-decoder |
| generation objective | OK | token CE is retained, so the model remains generative |
| collapse controls | OK | EMA targets plus normalized top-layer latent prediction are configured |
| encoder emphasis | OK | encoder capacity is at least decoder capacity, which suits representation pretraining |
| rough VRAM | OK | estimated peak 6.2 GB on a 24.0 GB budget |
| EMA target scope | OK | EMA target is encoder-only and has no optimizer state |

## Recommended Training Shape

Train it as an encoder-decoder model with JEPA as an auxiliary representation objective, not as a pure JEPA model.

1. Build a normal encoder-decoder Transformer with tied token embeddings.
2. Keep a standard token-level denoising or seq2seq cross-entropy loss.
3. Add an EMA target encoder. The target encoder sees the unmasked source sequence.
4. The student encoder sees a masked/corrupted source sequence.
5. A compact predictor receives student encoder states plus mask-position queries and predicts the EMA target encoder states for masked source spans.
6. Normalize target states and compute latent loss on masked positions only.
7. Optimize `total_loss = CE + lambda_jepa * latent_loss`, with `lambda_jepa` ramped from 0 to about 0.25 over early training.
8. Update the target encoder with EMA after each optimizer step.

Pseudo-code:

```python
for batch in train_loader:
    source, labels = batch["source_ids"], batch["labels"]
    masked_source, masked_positions = span_mask(source)

    with torch.no_grad():
        target_states = ema_encoder(source, output_hidden_states=True)
        target = normalize(mean_top_k(target_states, k=4))
        target = gather_positions(target, masked_positions)

    student_states = model.encoder(masked_source)
    pred = predictor(student_states, masked_positions)
    latent_loss = smooth_l1_or_mse(normalize(pred), target)

    decoder_logits = model.decoder(labels[:, :-1], encoder_states=student_states)
    ce_loss = cross_entropy(decoder_logits, labels[:, 1:])

    loss = ce_loss + lambda_jepa * latent_loss
    loss.backward()
    optimizer.step()
    update_ema(ema_encoder, model.encoder)
```

## Why This Architecture Is Worth Training

- **Small models need representation help.** A 0.2B encoder-decoder has limited capacity. Latent prediction pressures the encoder to learn contextual, semantic states instead of spending all learning signal on next-token surface form.
- **The decoder still needs token supervision.** JEPA predicts hidden representations, not text. Keeping CE is what makes the model useful for generation, translation, summarization, or instruction-style seq2seq work.
- **EMA targets stabilize the objective.** The predictor learns against a slowly moving teacher, which reduces representation collapse and avoids the student chasing its own current noise.
- **A small predictor is the right bottleneck.** The predictor absorbs the latent prediction task so the encoder does not have to contort itself into directly reconstructing target states.
- **Encoder-only EMA keeps this plausible at 0.2B.** A full encoder-decoder teacher would be wasteful. Copying only the encoder keeps the extra storage meaningful but not absurd.
- **The latent objective is cheaper than reconstruction-heavy alternatives.** You predict normalized hidden states for masked spans instead of reconstructing every token/pixel/detail.

## What Would Make It Unwise

- Training with only JEPA loss while expecting text generation.
- Using a large predictor that becomes a second model.
- Copying the full encoder-decoder as the EMA target.
- Training on too little text; the CE-only baseline may simply be easier and more reliable.
- No collapse monitoring. A falling latent loss is not enough if representation variance also collapses.

## Empirical Verification Plan

Run a short ablation before full training:

1. CE-only baseline for the same tokens, batch size, and optimizer.
2. CE + JEPA with `lambda_jepa` ramped to 0.1.
3. CE + JEPA with `lambda_jepa` ramped to 0.25.

Track these gates:

- Validation CE/perplexity should not regress by more than 1-2 percent after warmup.
- JEPA loss should fall while target representation variance stays non-zero.
- Mean pairwise cosine similarity between unrelated examples should not approach 1.
- Linear probe, retrieval, or classification performance on encoder states should improve over CE-only.
- Downstream seq2seq metric should match or beat CE-only after equal compute.

## Assumptions

- Vocabulary size: 32128
- Source length: 512
- Target length: 256
- Micro-batch size: 8
- VRAM budget: 24.0 GB
- Gradient checkpointing: True
- Parameter target: 200.0M
- Target tolerance: 12%

The VRAM estimate is intentionally rough. Treat it as an early warning system, not a replacement for a real profiler.
