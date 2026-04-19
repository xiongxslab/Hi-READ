#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export HIREAD_GRAM_BASE_DIR="${REPO_ROOT}/demo_data/gram_peaks_annotation/peaks"
export HIREAD_ANNOTATION_DIR="${REPO_ROOT}/demo_data/common_annotations"
export HIREAD_GRAM_OUTPUT_DIR="${REPO_ROOT}/demo_outputs/gram_smoke"

mkdir -p "${HIREAD_GRAM_OUTPUT_DIR}"

echo "Running GRAM annotation smoke demo"
echo "  peaks: ${HIREAD_GRAM_BASE_DIR}"
echo "  annotations: ${HIREAD_ANNOTATION_DIR}"
echo "  outputs: ${HIREAD_GRAM_OUTPUT_DIR}"

Rscript "${REPO_ROOT}/workflows/gram_peaks_annotation/genomic_annotation_analysis.R"
Rscript "${REPO_ROOT}/workflows/gram_peaks_annotation/chromhmm_ccre_heatmap_stackplot.R"

echo
echo "Generated outputs:"
find "${HIREAD_GRAM_OUTPUT_DIR}" -maxdepth 4 -type f | sort
