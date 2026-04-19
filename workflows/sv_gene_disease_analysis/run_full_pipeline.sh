#!/bin/bash
#
# Full SV-Gene-Disease Analysis Pipeline
#
# This script runs all 11 steps to reproduce the key figures:
# - NATURE_sharedGOID_examples_panel_a_seizure_disorder_barplot.pdf
# - NATURE_sharedGOID_examples_panel_d_epilepsy_barplot.pdf
# - a_seizure_disorder_CTCF.pdf (delta-gene track)
# - d_epilepsy_NANOG.pdf (delta-gene track)
#
# Prerequisites:
# - Raw GO results in per-TF directories (CTCF/, KLF4/, NANOG/, OCT4/, Rad21/)
# - Disease annotation files (matched_deletions.xlsx, duplication_with_clinvar.xlsx)
# - Disease category lists in disease_lists/
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_DIR="${SCRIPT_DIR}"
RUN_ROOT="${HIREAD_SV_RUN_ROOT:-${WORKFLOW_DIR}}"
SOURCE_ROOT="${HIREAD_SV_SOURCE_ROOT:-${RUN_ROOT}/sv_perturbation_analysis}"
OUTPUT_DIR="${HIREAD_SV_OUTPUT_DIR:-${RUN_ROOT}/analysis_min1}"
SCRIPTS_DIR="${WORKFLOW_DIR}/scripts"
RAW_SUBDIR="${HIREAD_SV_RAW_SUBDIR:-min_sv1}"

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "SV-Gene-Disease Analysis Pipeline"
echo "=========================================="
echo ""
echo "Run root: ${RUN_ROOT}"
echo "Source root: ${SOURCE_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Step 0: Snapshot raw GO results
echo "Step 0: Snapshot raw GO results from per-TF directories..."
python ${SCRIPTS_DIR}/step0_snapshot_go_results.py \
  --source-root "${SOURCE_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --out-subdir "${RAW_SUBDIR}"
echo "✓ Step 0 complete"
echo ""

# Step 1: Rebuild analysis tables
echo "Step 1: Rebuild core analysis tables..."
python ${SCRIPTS_DIR}/step1_rebuild_analysis_tables.py \
  --run-root "${RUN_ROOT}" \
  --raw-subdir "${RAW_SUBDIR}"
echo "✓ Step 1 complete (expect 1023 sig rows, 786 agg pairs)"
echo ""

# Step 2: Select correspondence diseases
echo "Step 2: Select correspondence diseases..."
python ${SCRIPTS_DIR}/step2_select_correspondence_diseases.py \
  --sig-pairs "${OUTPUT_DIR}/min1_all_sig_pairs_agg.csv" \
  --output-dir "${OUTPUT_DIR}/correspondence" \
  --min-tfs 2
echo "✓ Step 2 complete"
echo ""

# Step 3: Make shared GO bubble panels
echo "Step 3: Generate shared GO bubble panels..."
Rscript ${SCRIPTS_DIR}/step3_make_shared_bubble_panels.R \
  --run-root "${RUN_ROOT}"
echo "✓ Step 3 complete"
echo ""

# Step 4: Make brain-shared bubble panels
echo "Step 4: Generate brain-shared bubble panels..."
Rscript ${SCRIPTS_DIR}/step4_make_brain_shared_bubble_panels.R \
  --run-root "${RUN_ROOT}"
echo "✓ Step 4 complete"
echo ""

# Step 5: Make brain-shared with terms
echo "Step 5: Generate brain-shared panels with shared terms..."
Rscript ${SCRIPTS_DIR}/step5_make_brain_shared_with_terms.R \
  --run-root "${RUN_ROOT}"
echo "✓ Step 5 complete"
echo ""

# Step 6: Make examples bubble plots
echo "Step 6: Generate examples bubble plots..."
Rscript ${SCRIPTS_DIR}/step6_make_examples_bubble_plots.R \
  --run-root "${RUN_ROOT}"
echo "✓ Step 6 complete"
echo ""

# Step 7: Find shared GOID diseases
echo "Step 7: Find diseases with shared GO IDs across TFs..."
python ${SCRIPTS_DIR}/step7_find_shared_goid_diseases.py \
  --run_root "${RUN_ROOT}" \
  --bucket brain \
  --tfs CTCF,Rad21 \
  --out_csv "${OUTPUT_DIR}/shared_goid/shared_goid_candidates_CTCF_Rad21_brain.csv"
python ${SCRIPTS_DIR}/step7_find_shared_goid_diseases.py \
  --run_root "${RUN_ROOT}" \
  --bucket brain \
  --tfs KLF4,OCT4,NANOG \
  --out_csv "${OUTPUT_DIR}/shared_goid/shared_goid_candidates_KLF4_OCT4_NANOG_brain.csv"
echo "✓ Step 7 complete"
echo ""

# Step 8: Make 4-panel shared examples
echo "Step 8: Generate 4-panel shared GOID examples..."
Rscript ${SCRIPTS_DIR}/step8_make_4panel_shared_examples.R \
  --run-root "${RUN_ROOT}"
echo "✓ Step 8 complete"
echo ""

# Step 9: Plot GO barplots (KEY FIGURES)
echo "Step 9: Generate GO barplots for key examples..."
Rscript ${SCRIPTS_DIR}/step9_plot_go_barplots.R \
  --run-root "${RUN_ROOT}"
echo "✓ Step 9 complete"
echo "  → NATURE_sharedGOID_examples_panel_a_seizure_disorder_barplot.pdf"
echo "  → NATURE_sharedGOID_examples_panel_d_epilepsy_barplot.pdf"
echo ""

# Step 10: Summarize gene lists
echo "Step 10: Summarize gene lists with delta rankings..."
python ${SCRIPTS_DIR}/step10_summarize_gene_lists.py \
  --run-root "${RUN_ROOT}" \
  --score-root "${SOURCE_ROOT}" \
  --out-dir "${OUTPUT_DIR}/correspondence_high_conf_min1/shared_sig_panels/gene_lists_sharedGOID_examples"
echo "✓ Step 10 complete"
echo ""

# Step 11: Plot delta-gene tracks (KEY FIGURES)
echo "Step 11: Generate delta-gene track plots..."
Rscript ${SCRIPTS_DIR}/step11_plot_delta_gene_tracks.R \
  --run-root "${RUN_ROOT}" \
  --sv-root "${SOURCE_ROOT}" \
  --output-dir "${OUTPUT_DIR}/correspondence_high_conf_min1/shared_sig_panels/example_3d_delta_gene_tracks"
echo "✓ Step 11 complete"
echo "  → a_seizure_disorder_CTCF.pdf"
echo "  → d_epilepsy_NANOG.pdf"
echo ""

echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Key outputs:"
echo "  1. ${OUTPUT_DIR}/correspondence_high_conf_min1/shared_sig_panels/NATURE_sharedGOID_examples_panel_a_seizure_disorder_barplot.pdf"
echo "  2. ${OUTPUT_DIR}/correspondence_high_conf_min1/shared_sig_panels/NATURE_sharedGOID_examples_panel_d_epilepsy_barplot.pdf"
echo "  3. ${OUTPUT_DIR}/correspondence_high_conf_min1/shared_sig_panels/example_3d_delta_gene_tracks/a_seizure_disorder_CTCF.pdf"
echo "  4. ${OUTPUT_DIR}/correspondence_high_conf_min1/shared_sig_panels/example_3d_delta_gene_tracks/d_epilepsy_NANOG.pdf"
echo ""
echo "Verification:"
echo "  - Check min1_all_sig_rows_hbi.csv has 1023 rows"
echo "  - Check min1_all_sig_pairs_agg.csv has 786 rows"
echo ""
