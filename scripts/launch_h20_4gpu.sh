#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train_h20_4gpu.yaml}"
RESUME_FROM="${2:-${JEPA_RESUME_FROM:-}}"

# Resolve config-derived launch settings with one torch-free call: the STOP
# file the *trainer* actually watches (runtime.stop_file) and the GPU list the
# config requests (runtime.cuda_visible_devices). Matching the STOP path avoids a
# stale STOP instantly halting the run; the GPU list lets a config pin e.g. 4-7.
CFG_STOP=""
CFG_DEVICES=""
CFG_RESOLVED=0
eval "$(PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" "${PYTHON:-python}" - "${CONFIG}" <<'PY' || true
import shlex, sys
from jepa_slm.config import load_training_config
c = load_training_config(sys.argv[1])
stop = c.runtime.stop_file
devices = c.runtime.cuda_visible_devices
# str() + explicit None checks: a YAML `cuda_visible_devices: 7` arrives as an
# int (shlex.quote would raise) and `0` is falsy — both must survive.
print(f"CFG_STOP={shlex.quote('' if stop is None else str(stop))}")
print(f"CFG_DEVICES={shlex.quote('' if devices is None else str(devices))}")
print("CFG_RESOLVED=1")  # printed last: only reached when the load succeeded
PY
)"
if [[ "${CFG_RESOLVED}" != "1" ]]; then
  echo "WARNING: could not resolve launch settings from ${CONFIG} (bad config or python env?);" >&2
  echo "         falling back to GPUs 0,1,2,3 and the default STOP path." >&2
fi

STOP_FILE="${JEPA_STOP_FILE:-${CFG_STOP:-outputs/jepa-slm-h20-4gpu/STOP}}"
rm -f "${STOP_FILE}"
mkdir -p "$(dirname "${STOP_FILE}")"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Precedence: env override > config runtime.cuda_visible_devices > 0,1,2,3.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CFG_DEVICES:-0,1,2,3}}"
# Derive nproc from the number of visible devices so 4,5,6,7 (or any list) works.
NPROC="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
[[ "${NPROC}" =~ ^[0-9]+$ && "${NPROC}" -ge 1 ]] || NPROC=4
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
  --nproc_per_node="${NPROC}" \
  -m jepa_slm.train \
  "${ARGS[@]}"
