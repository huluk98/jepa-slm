# Executable Training Pipeline

The training path now implements the repository's core JEPA contract:

```text
clean source -> EMA encoder -> latent target
masked source -> student encoder -> predictor -> JEPA latent loss
masked source -> encoder-decoder -> CE token loss
```

Decoded tokens are never fed back into the JEPA target path.

## Main Entry Point

```bash
PYTHONPATH=src python -m jepa_slm.train --config configs/train_tiny_smoke.yaml
```

For the H20 launch:

```bash
conda activate jepa-h20
bash scripts/launch_h20_8gpu.sh configs/train_h20_8gpu.yaml
```

## Modules

- `src/jepa_slm/config.py`: loads compact model configs and H20 launch configs.
- `src/jepa_slm/masking.py`: masks source spans and returns source positions for JEPA.
- `src/jepa_slm/modeling.py`: wraps T5 with an encoder-only EMA target and JEPA predictor.
- `src/jepa_slm/data.py`: Hugging Face streaming data, tokenizer loading, and offline smoke data.
- `src/jepa_slm/trainer.py`: DDP/BF16 training loop, optimizer, EMA updates, checkpoints.
- `src/jepa_slm/train.py`: CLI entrypoint.

## Verification

Run the dependency-light tests in base Python:

```bash
PYTHONPATH=src python3 -m pytest -q
```

Run the full contract tests in an environment with PyTorch and Transformers:

```bash
conda run -n encoder-decoder-prune env PYTHONPATH=src python -m pytest -q
```

Run a one-step offline train:

```bash
conda run -n encoder-decoder-prune env PYTHONPATH=src \
  python -m jepa_slm.train --config configs/train_tiny_smoke.yaml
```

## Current Boundaries

- The executable trainer supports fixed batch sizes, not automatic batch autotune.
- Sequence packing is not wired into the trainer yet.
- The production tokenizer should be trained with `scripts/train_sentencepiece_tokenizer.py`;
  the internal byte tokenizer exists only for offline smoke tests.
- The current Hugging Face T5 backend supports the CE+JEPA contract. The SwiGLU
  ablation config is a design target; if using stock T5 modules, GEGLU/GELU are
  the directly supported FFN paths.
