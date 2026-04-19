#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_DIR="${REPO_ROOT}/demo_data/loops_cluster/input"
OUTPUT_DIR="${REPO_ROOT}/demo_outputs/loops_cluster"

mkdir -p "${OUTPUT_DIR}"

python "${REPO_ROOT}/scripts/run_loops_cluster.py" cluster-only \
  --data-dir "${INPUT_DIR}" \
  --out "${OUTPUT_DIR}" \
  --k 2 \
  --square 4

echo "Demo output written to: ${OUTPUT_DIR}"
