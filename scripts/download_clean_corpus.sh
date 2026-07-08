#!/usr/bin/env bash
# Download the preprocessed fineweb-edu sample-10BT corpus from GitHub Releases.
#
#   bash scripts/download_clean_corpus.sh [TAG] [DEST_DIR]
#
# Fetches the gzipped cleaned shards (plus metadata + a local t5-small
# tokenizer bundle), verifies checksums, and decompresses into DEST_DIR so
# configs/train_clean_local_full.yaml works as-is. Useful when the training
# box cannot reach HuggingFace (the shards were prepared elsewhere with
# scripts/prepare_clean_corpus.py).
#
# Env overrides:
#   KEEP_GZ=1     keep .jsonl.gz instead of decompressing (the dataloader can
#                 read .gz directly; point data.dataset at 'clean-*.jsonl.gz')
#   REPO=...      GitHub repo (default huluk98/jepa-slm)
set -euo pipefail

TAG="${1:-corpus-10bt-v1}"
DEST="${2:-data/clean/fineweb-edu-sample10bt-full}"
REPO="${REPO:-huluk98/jepa-slm}"
BASE="https://github.com/${REPO}/releases/download/${TAG}"

command -v curl >/dev/null || { echo "ERROR: curl is required" >&2; exit 1; }
mkdir -p "${DEST}"
cd "${DEST}"

echo ">>> Fetching checksum list from ${BASE}"
curl -fsSL -o SHA256SUMS "${BASE}/SHA256SUMS"

# Everything listed in SHA256SUMS except the tokenizer bundle lands in DEST.
assets=$(awk '{print $2}' SHA256SUMS)
total=$(wc -l < SHA256SUMS | tr -d ' ')
n=0
for asset in ${assets}; do
  n=$((n + 1))
  if [[ -f "${asset}" ]]; then
    echo ">>> [${n}/${total}] ${asset} already present — skipping download"
  else
    echo ">>> [${n}/${total}] ${asset}"
    # -C - resumes a partially downloaded asset after an interruption.
    curl -fL --retry 5 --retry-delay 5 -C - -o "${asset}.part" "${BASE}/${asset}"
    mv "${asset}.part" "${asset}"
  fi
done

echo ">>> Verifying checksums"
if command -v sha256sum >/dev/null; then
  sha256sum -c SHA256SUMS
else
  shasum -a 256 -c SHA256SUMS  # macOS
fi

# Unpack the offline tokenizer bundle inside DEST.
if [[ -f t5-small-tokenizer.tar.gz ]]; then
  tar -xzf t5-small-tokenizer.tar.gz
  echo ">>> Local tokenizer at $(pwd)/t5-small-tokenizer (use when HF is"
  echo "    unreachable: set data.tokenizer_name to that path in your config)"
fi

if [[ "${KEEP_GZ:-0}" != "1" ]]; then
  echo ">>> Decompressing shards (KEEP_GZ=1 skips this; the dataloader reads .gz too)"
  ls clean-*.jsonl.gz | xargs -P "$(nproc 2>/dev/null || sysctl -n hw.ncpu)" -I{} gunzip -f {}
fi

echo ">>> Done. Corpus in $(pwd)"
echo ">>> Train with: bash scripts/run_training.sh configs/train_clean_local_full.yaml"
