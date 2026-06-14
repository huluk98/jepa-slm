#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-jepa-h20}"
ENV_FILE="${ENV_FILE:-envs/jepa-h20-cu124.yml}"

conda env create -f "${ENV_FILE}" || conda env update -n "${ENV_NAME}" -f "${ENV_FILE}"

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install --no-build-isolation -r envs/requirements-h20-optional.txt

python scripts/h20_sanity_check.py
