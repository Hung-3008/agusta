#!/usr/bin/env bash
# =====================================================================
# Table 4 Ablation Runner — Modality Coverage
# Factorial ablation over V, A, L, AV modality groups:
#   Row 1: V only        (vision: 5 backbones)
#   Row 2: A only        (audio:  1 backbone)
#   Row 3: L only        (language: 3 backbones)
#   Row 4: V + A         (6 backbones)
#   Row 5: V + L         (8 backbones)
#   Row 6: A + L         (4 backbones)
#   Row 7: V + A + L     (9 backbones)
#   Row 8: All (ours)    (V+A+L+AV: 10 backbones)
#
# All use: DiT-L (24b/1024d/16h), full SISFM, 4 subjects, 80 epochs
#
# Usage:
#   bash src/scripts/run_table4_modality.sh
#   bash src/scripts/run_table4_modality.sh --fast_dev_run
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN="${PROJECT_ROOT}/src/train_brainflow.py"
CFG_DIR="${PROJECT_ROOT}/src/configs/table4_modality_coverage"
EXTRA_ARGS="${@}"

cd "${PROJECT_ROOT}"

ROWS=(
    "v_only:V only"
    "a_only:A only"
    "l_only:L only"
    "v_a:V + A"
    "v_l:V + L"
    "a_l:A + L"
    "v_a_l:V + A + L"
    "all_ours:All (ours)"
)

for entry in "${ROWS[@]}"; do
    key="${entry%%:*}"
    label="${entry##*:}"
    echo ""
    echo "============================================================"
    echo " Table 4 — ${label}"
    echo "============================================================"
    conda run -n base python "${TRAIN}" \
        --config "${CFG_DIR}/${key}.yaml" \
        ${EXTRA_ARGS}
done

echo ""
echo "All Table 4 modality ablations complete."
echo "Results in outputs/table4_modality/{v_only,a_only,l_only,v_a,v_l,a_l,v_a_l,all_ours}/"
