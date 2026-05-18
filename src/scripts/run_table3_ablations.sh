#!/usr/bin/env bash
# =====================================================================
# Table 3 Ablation Runner — Generative vs. Deterministic Encoding
# Runs three formulations of the same DiT-X Small architecture:
#   Row 1: Regression DiT-X  (MSE loss, no flow matching)
#   Row 2: CFM + Gaussian    (standard N(0,I) source)
#   Row 3: CFM + SISFM       (stimulus-conditioned source, ours)
#
# Usage:
#   bash src/scripts/run_table3_ablations.sh
#   bash src/scripts/run_table3_ablations.sh --fast_dev_run   # smoke test
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN="${PROJECT_ROOT}/src/train_brainflow.py"
EXTRA_ARGS="${@}"

cd "${PROJECT_ROOT}"

echo "============================================================"
echo " Table 3 Ablation: Row 1 — Regression DiT-X"
echo "============================================================"
conda run -n base python "${TRAIN}" \
    --config src/configs/table3_generative/regression_ditx.yaml \
    ${EXTRA_ARGS}

echo ""
echo "============================================================"
echo " Table 3 Ablation: Row 2 — CFM + Gaussian Source"
echo "============================================================"
conda run -n base python "${TRAIN}" \
    --config src/configs/table3_generative/cfm_gaussian.yaml \
    ${EXTRA_ARGS}

echo ""
echo "============================================================"
echo " Table 3 Ablation: Row 3 — CFM + SISFM (ours)"
echo "============================================================"
conda run -n base python "${TRAIN}" \
    --config src/configs/table3_generative/cfm_sisfm.yaml \
    ${EXTRA_ARGS}

echo ""
echo "All Table 3 ablations complete."
echo "Results in:"
echo "  outputs/ablation_regression_ditx/"
echo "  outputs/ablation_cfm_gaussian/"
echo "  outputs/ablation_cfm_sisfm/"
