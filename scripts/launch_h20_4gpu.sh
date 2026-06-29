#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train_h20_4gpu.yaml}"
RESUME_FROM="${2:-${JEPA_RESUME_FROM:-}}"

# Resolve the graceful-stop file the *trainer* actually watches (config
# runtime.stop_file) so the path we clear/announce matches the chosen config --
# otherwise a stale STOP at the config's path could instantly halt the run, and
# the printed "touch ..." hint would be wrong. The config loader is torch-free.
if [[ -n "${JEPA_STOP_FILE:-}" ]]; then
  STOP_FILE="${JEPA_STOP_FILE}"
else
  STOP_FILE="$(PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" "${PYTHON:-python}" - "${CONFIG}" <<'PY' 2>/dev/null || true
import sys
from jepa_slm.config import load_training_config
print(load_training_config(sys.argv[1]).runtime.stop_file or "")
PY
)"
  STOP_FILE="${STOP_FILE:-outputs/jepa-slm-h20-4gpu/STOP}"
fi

rm -f "${STOP_FILE}"
mkdir -p "$(dirname "${STOP_FILE}")"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

echo "Launching 4-GPU H20 DDP with ${CONFIG}"
echo "Graceful stop: touch ${STOP_FILE}"
if [[ -n "${RESUME_FROM}" ]]; then
  echo "Resume checkpoint: ${RESUME_FROM}"
fi

ARGS=(--config "${CONFIG}")
if [[ -n "${RESUME_FROM}" ]]; then
  ARGS+=(--resume-from "${RESUME_FROM}")
fi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  -m jepa_slm.train \
  "${ARGS[@]}"
