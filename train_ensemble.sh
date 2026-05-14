#!/usr/bin/env bash
# =====================================================================
# Ensemble Training — 50 seeds × 1 config → 50 independent checkpoints
# =====================================================================
#
# Usage:
#   bash train_ensemble.sh                    # Train all 50 seeds
#   bash train_ensemble.sh --resume           # Resume interrupted runs
#   bash train_ensemble.sh --dry-run          # Print commands only
#
# Output structure:
#   outputs/brainflow_ensemble50/
#   ├── seed_1001/   (best.pt, last.pt, config.yaml, history.csv)
#   ├── seed_1234/
#   ├── ...
#   └── seed_9901/
#
# Each run is fully independent (different random init + data order).
# Already-completed seeds (with best.pt) are skipped automatically
# unless --resume is passed.
# =====================================================================

set -euo pipefail

# --- Configuration ---
CONFIG="src/configs/brainflow.yaml"
OUTPUT_BASE="outputs/brainflow_ensemble50"
LOG_DIR="${OUTPUT_BASE}/logs"

# --- 50 diverse seeds (prime-like, well-separated) ---
SEEDS=(
    1001 1234 1597 1783 2003
    2221 2468 2719 2953 3137
    3301 3571 3779 4001 4219
    4441 4673 4889 5101 5347
    5501 5743 5981 6199 6421
    6637 6857 7001 7213 7459
    7687 7901 8117 8353 8573
    8807 9001 9221 9437 9661
    9901 1117 1373 1621 1879
    2131 2377 2633 2887 3163
)

# --- Parse arguments ---
RESUME_FLAG=""
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --resume)  RESUME_FLAG="--resume" ;;
        --dry-run) DRY_RUN=true ;;
        *)         echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# --- Setup ---
mkdir -p "${LOG_DIR}"

TOTAL=${#SEEDS[@]}
echo "============================================================="
echo " BrainFlow Ensemble Training"
echo " Config:  ${CONFIG}"
echo " Seeds:   ${TOTAL}"
echo " Output:  ${OUTPUT_BASE}"
echo " Resume:  ${RESUME_FLAG:-off}"
echo "============================================================="
echo ""

COMPLETED=0
SKIPPED=0
FAILED=0

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    RUN_DIR="${OUTPUT_BASE}/seed_${SEED}"
    RUN_NUM=$((i + 1))
    LOG_FILE="${LOG_DIR}/seed_${SEED}.log"

    echo "-------------------------------------------------------------"
    echo " [${RUN_NUM}/${TOTAL}] Seed: ${SEED}"
    echo "-------------------------------------------------------------"

    # Skip if already completed (has best.pt) and not resuming
    if [ -f "${RUN_DIR}/best.pt" ] && [ -z "${RESUME_FLAG}" ]; then
        echo "  ✓ Already completed (best.pt exists). Skipping."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Build command
    CMD="python src/train_brainflow.py \
        --config ${CONFIG} \
        --seed ${SEED} \
        --output-dir ${RUN_DIR} \
        ${RESUME_FLAG}"

    if $DRY_RUN; then
        echo "  [DRY RUN] ${CMD}"
        continue
    fi

    echo "  Output:  ${RUN_DIR}"
    echo "  Log:     ${LOG_FILE}"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Run training (tee to log file)
    mkdir -p "${RUN_DIR}"
    if ${CMD} 2>&1 | tee "${LOG_FILE}"; then
        echo ""
        echo "  ✓ Seed ${SEED} completed successfully."
        COMPLETED=$((COMPLETED + 1))
    else
        echo ""
        echo "  ✗ Seed ${SEED} FAILED (exit code: $?)."
        FAILED=$((FAILED + 1))
        # Continue to next seed instead of aborting
    fi

    echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
done

echo ""
echo "============================================================="
echo " Ensemble Training Summary"
echo "============================================================="
echo " Total seeds:  ${TOTAL}"
echo " Completed:    ${COMPLETED}"
echo " Skipped:      ${SKIPPED}"
echo " Failed:       ${FAILED}"
echo "============================================================="

# --- Final check: count available checkpoints ---
N_CHECKPOINTS=$(find "${OUTPUT_BASE}" -name "best.pt" 2>/dev/null | wc -l)
echo " Available checkpoints (best.pt): ${N_CHECKPOINTS}/${TOTAL}"
echo "============================================================="
