#!/bin/bash
# =============================================================================
# Train BrainFlow ensemble — N runs with different random seeds.
#
# Usage:
#   bash scripts/train_ensemble.sh                    # 5 seeds (default)
#   bash scripts/train_ensemble.sh 3                  # 3 seeds
#   bash scripts/train_ensemble.sh 5 --resume         # 5 seeds, resume all
#   N_SEEDS=7 bash scripts/train_ensemble.sh          # 7 seeds via env var
# =============================================================================

set -euo pipefail

# --- Configuration ---
N_SEEDS="${1:-${N_SEEDS:-5}}"       # number of ensemble members
CONFIG="src/configs/brainflow.yaml"
BASE_OUTPUT_DIR="outputs/brainflow_ensemble"
EXTRA_ARGS="${@:2}"                 # pass --resume, --warmstart, etc.

# Fixed seeds for reproducibility across runs (50 seeds)
# SEEDS=(
#     42    123   256   512   1024
#     2048  4096  8192  16384 32768
#     7     13    37    73    97
#     137   251   359   499   613
#     743   857   991   1109  1279
#     1429  1597  1741  1889  2039
#     2203  2371  2543  2699  2861
#     3011  3187  3343  3511  3671
#     3847  4007  4177  4337  4507
#     4673  4831  5003  5179  5351
# )

# Alternative seed set (10 seeds, no overlap with above) — uncomment to use:
SEEDS=(5501 5647 5801 5953 6101 6263 6421 6577 6733 6899)

echo "=============================================="
echo " BrainFlow Ensemble Training"
echo "  Seeds:  ${N_SEEDS}"
echo "  Config: ${CONFIG}"
echo "  Output: ${BASE_OUTPUT_DIR}/seed_*"
echo "  Extra:  ${EXTRA_ARGS:-none}"
echo "=============================================="

for i in $(seq 0 $((N_SEEDS - 1))); do
    SEED=${SEEDS[$i]}
    RUN_DIR="${BASE_OUTPUT_DIR}/seed_${SEED}"

    echo ""
    echo "----------------------------------------------"
    echo " Run $((i + 1))/${N_SEEDS} — seed=${SEED}"
    echo " Output: ${RUN_DIR}"
    echo "----------------------------------------------"

    python src/train_brainflow.py \
        --config "${CONFIG}" \
        --seed "${SEED}" \
        --output-dir "${RUN_DIR}" \
        ${EXTRA_ARGS}

    echo "✓ Seed ${SEED} finished."
done

echo ""
echo "=============================================="
echo " All ${N_SEEDS} ensemble runs completed."
echo " Results in: ${BASE_OUTPUT_DIR}/"
echo "=============================================="
