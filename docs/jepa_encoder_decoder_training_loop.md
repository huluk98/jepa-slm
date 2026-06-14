# JEPA Encoder-Decoder Training Loop

This project is currently a design sandbox. The safest way to use JEPA for a small encoder-decoder model is as an auxiliary latent-prediction loss, not as the only objective.

## Core Loop

```python
for batch in train_loader:
    source_ids = batch["source_ids"]
    labels = batch["labels"]

    masked_source_ids, masked_positions = span_mask(source_ids)

    with torch.no_grad():
        teacher_hidden = ema_encoder(source_ids, output_hidden_states=True)
        target = mean_top_k_layers(teacher_hidden, k=4)
        target = layer_norm_without_affine(target)
        target = gather_positions(target, masked_positions)

    student_hidden = model.encoder(masked_source_ids)
    predicted_target = predictor(student_hidden, masked_positions)
    jepa_loss = smooth_l1_loss(normalize(predicted_target), target)

    logits = model(
        input_ids=masked_source_ids,
        decoder_input_ids=shift_right(labels),
    ).logits
    ce_loss = cross_entropy(logits, labels)

    loss = ce_loss + lambda_jepa * jepa_loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    update_ema(ema_encoder, model.encoder, tau=current_ema_tau)
```

## Why Not Pure JEPA

JEPA predicts representations. It does not directly train a decoder to produce text. For an encoder-decoder language model, the token loss is the contract that keeps the model generative. The JEPA loss should make the encoder states better; the CE loss should keep the model useful.

## Verification Gates

- Compare against a CE-only baseline at equal tokens and equal optimizer settings.
- Monitor latent variance so the model does not collapse to constant hidden states.
- Monitor validation CE so the auxiliary loss does not damage generation.
- Check an encoder-state probe or retrieval task; that is where JEPA should help first.
- Keep predictor parameters below about 8-10 percent of the base model.
- Keep the EMA target encoder-only unless there is a clear reason to distill decoder states too.

## Practical Starting Recipe

- Target model: about 200M trainable parameters.
- Encoder layers: at least as many as decoder layers.
- Predictor: 2-3 layers, narrower than `d_model`.
- JEPA loss: SmoothL1 or MSE over normalized latent states.
- `lambda_jepa`: ramp from 0 to 0.1-0.25 during warmup.
- EMA tau: start around 0.99 and rise toward 0.999+.
- Masking: span masking over source tokens, with targets gathered only at masked positions.
