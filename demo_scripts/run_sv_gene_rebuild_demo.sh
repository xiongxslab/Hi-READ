#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ROOT="${REPO_ROOT}/demo_data/sv_gene_disease_analysis"
OUTPUT_DIR="${REPO_ROOT}/demo_outputs/sv_gene_disease_analysis/analysis_min1"

python "${REPO_ROOT}/workflows/sv_gene_disease_analysis/scripts/step1_rebuild_analysis_tables.py" \
  --run-root "${RUN_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --raw-subdir min_sv1

echo "Rebuilt analysis tables under: ${OUTPUT_DIR}"
