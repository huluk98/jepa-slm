#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train_h20_4gpu.yaml}"
RESUME_FROM="${2:-${JEPA_RESUME_FROM:-}}"
STOP_FILE="${JEPA_STOP_FILE:-outputs/jepa-slm-h20-4gpu/STOP}"

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
