#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train_h20_8gpu.yaml}"
RESUME_FROM="${2:-${JEPA_RESUME_FROM:-}}"

# Resolve config-derived launch settings (STOP file the trainer watches, GPU
# pin) with one torch-free call; see launch_h20_4gpu.sh for details.
CFG_STOP=""
CFG_DEVICES=""
CFG_RESOLVED=0
eval "$(PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" "${PYTHON:-python}" - "${CONFIG}" <<'PY' || true
import shlex, sys
from jepa_slm.config import load_training_config
c = load_training_config(sys.argv[1])
stop = c.runtime.stop_file
devices = c.runtime.cuda_visible_devices
print(f"CFG_STOP={shlex.quote('' if stop is None else str(stop))}")
print(f"CFG_DEVICES={shlex.quote('' if devices is None else str(devices))}")
print("CFG_RESOLVED=1")  # printed last: only reached when the load succeeded
PY
)"
if [[ "${CFG_RESOLVED}" != "1" ]]; then
  echo "WARNING: could not resolve launch settings from ${CONFIG} (bad config or python env?); using defaults." >&2
fi

# Clear a stale STOP file so a previous graceful stop cannot instantly halt
# this run at step 0.
STOP_FILE="${JEPA_STOP_FILE:-${CFG_STOP:-}}"
if [[ -n "${STOP_FILE}" ]]; then
  rm -f "${STOP_FILE}"
  mkdir -p "$(dirname "${STOP_FILE}")"
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Precedence: env override > config runtime.cuda_visible_devices > all 8.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CFG_DEVICES:-0,1,2,3,4,5,6,7}}"
# Derive nproc from the number of visible devices so a restricted GPU list
# (e.g. sharing the box) does not spawn ranks onto nonexistent ordinals.
NPROC="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
[[ "${NPROC}" =~ ^[0-9]+$ && "${NPROC}" -ge 1 ]] || NPROC=8
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

echo "Launching ${NPROC}-GPU H20 DDP with ${CONFIG}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
if [[ -n "${STOP_FILE}" ]]; then
  echo "Graceful stop: touch ${STOP_FILE}"
fi
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
  --nproc_per_node="${NPROC}" \
  -m jepa_slm.train \
  "${ARGS[@]}"
