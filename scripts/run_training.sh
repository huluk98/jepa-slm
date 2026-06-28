#!/usr/bin/env bash
# One-command JEPA-SLM training launcher.
#
#   bash scripts/run_training.sh [CONFIG] [NPROC]
#
# Defaults to the 8x H20 config. It:
#   1. checks the Python env (torch importable),
#   2. resolves the config's data source — if it is a local cleaned-shard glob
#      that is missing, it builds the shards first (the "cleaning" step); if it
#      streams (e.g. fineweb-edu) or is synthetic, no local prep is needed,
#   3. launches distributed training via torchrun.
#
# Useful overrides (env vars):
#   PYTHON=...            python to use (default: python)
#   NPROC=...             GPUs per node (default: auto from nvidia-smi)
#   CLEAN_DATASET=...     HF dataset to clean from   (default HuggingFaceFW/fineweb-edu)
#   CLEAN_SUBSET=...      subset                     (default from config / sample-10BT)
#   CLEAN_MAX_DOCS=...    docs to clean              (default 100000; 0 = all)
#   DRY_RUN=1            print the prepare/torchrun commands instead of running them
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${1:-configs/train_h20_8gpu.yaml}"
PYBIN="${PYTHON:-python}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

# NCCL / Tensor-Core env (mirrors launch_h20_8gpu.sh).
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }

# 1) env check
if ! "$PYBIN" -c "import torch" >/dev/null 2>&1; then
  echo "ERROR: '$PYBIN' cannot import torch. Set up the env first:" >&2
  echo "       bash scripts/install_h20_env.sh   (or activate your conda env)" >&2
  exit 1
fi

# 2) resolve the data source from the config
eval "$("$PYBIN" scripts/launch_preflight.py "$CONFIG")"
echo ">>> config=$CONFIG  data=$DATASET  local=$IS_LOCAL  packing=$SEQUENCE_PACKING  max_steps=$MAX_STEPS"

# 3) GPU count
if [[ -n "${NPROC:-}" ]]; then
  nproc="$NPROC"
elif command -v nvidia-smi >/dev/null 2>&1; then
  nproc="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 1)"
else
  nproc=1
fi
[[ "${nproc:-0}" -ge 1 ]] || nproc=1

# 4) ensure data is present (clean if a local glob is missing)
if [[ "$IS_LOCAL" == "1" && "$GLOB_COUNT" == "0" ]]; then
  echo ">>> No cleaned shards match '$DATASET' — building them now into '$OUTPUT_DIR'"
  PREP=("$PYBIN" scripts/prepare_clean_corpus.py
        --dataset "${CLEAN_DATASET:-HuggingFaceFW/fineweb-edu}"
        --subset "${CLEAN_SUBSET:-$SUBSET}"
        --output-dir "$OUTPUT_DIR"
        --max-docs "${CLEAN_MAX_DOCS:-100000}"
        --min-chars 200 --max-chars 20000 --overwrite)
  if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "DRY_RUN prepare: ${PREP[*]}"; else "${PREP[@]}"; fi
elif [[ "$IS_LOCAL" == "1" ]]; then
  echo ">>> Found $GLOB_COUNT cleaned shard(s) for '$DATASET' — skipping cleaning."
else
  echo ">>> Streaming/synthetic source '$DATASET' — cleaned on-the-fly, no local prep needed."
fi
if [[ -n "$EVAL_DATASET" ]]; then echo ">>> Validation-CE enabled on: $EVAL_DATASET"; fi

# 5) launch
LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node="$nproc" -m jepa_slm.train --config "$CONFIG")
echo ">>> Launching: ${LAUNCH[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "DRY_RUN launch (not executed)"; exit 0; fi
exec "${LAUNCH[@]}"
