#!/bin/bash
# ==============================================================================
# Prediction-Level Ensemble: S7 evaluation with 4 medium checkpoints
# 
# Runs S7 inference on each checkpoint independently, then averages predictions.
# Config: outputs/agusta_ensemble_50_medium/config.yaml (DiT-L, per-voxel sigma)
# Batch size: 128, Stride: 5
# ==============================================================================

set -e

CONFIG="outputs/agusta_ensemble_50_medium/config.yaml"
BATCH_SIZE=128
STRIDE=10

# 4 seed checkpoints
SEEDS=(1001 1234 3137 5347)

echo "============================================================"
echo "  Prediction-Level Ensemble — S7"
echo "  Config: ${CONFIG}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Batch size: ${BATCH_SIZE}, Stride: ${STRIDE}"
echo "============================================================"

# ---- Step 1: Run inference for each seed ----
PRED_DIRS=()

for SEED in "${SEEDS[@]}"; do
    CKPT="outputs/agusta_ensemble_50_medium/best_${SEED}.pt"
    OUT_DIR="outputs/pred_medium_seed${SEED}"
    SUB_DIR="outputs/submissions/${OUT_DIR##*/}/s7"

    if [ -f "${SUB_DIR}/submission.npy" ]; then
        echo ""
        echo "[SKIP] Seed ${SEED} — already completed: ${SUB_DIR}/submission.npy"
        PRED_DIRS+=("${SUB_DIR}")
        continue
    fi

    echo ""
    echo "============================================================"
    echo "  Running S7 inference — Seed ${SEED}"
    echo "  Checkpoint: ${CKPT}"
    echo "  Output: ${OUT_DIR}"
    echo "============================================================"

    python src/evaluate_brainflow.py \
        --config "${CONFIG}" \
        --checkpoint "${CKPT}" \
        --eval_session s7 \
        --batch_size ${BATCH_SIZE} \
        --stride ${STRIDE} \
        --output_dir "${OUT_DIR}"

    PRED_DIRS+=("${SUB_DIR}")
    echo "[DONE] Seed ${SEED} — saved to ${SUB_DIR}"
done

# ---- Step 2: Ensemble predictions ----
ENSEMBLE_DIR="outputs/submissions/pred_ensemble_4seed_medium_s7/s7"

echo ""
echo "============================================================"
echo "  Ensembling predictions from ${#PRED_DIRS[@]} runs"
echo "  Output: ${ENSEMBLE_DIR}"
echo "============================================================"

python src/ensemble_predictions.py \
    --pred-dirs "${PRED_DIRS[@]}" \
    --output-dir "${ENSEMBLE_DIR}"

echo ""
echo "============================================================"
echo "  ALL DONE!"
echo "  Individual predictions:"
for d in "${PRED_DIRS[@]}"; do
    echo "    - ${d}"
done
echo "  Ensembled submission: ${ENSEMBLE_DIR}/submission.zip"
echo "============================================================"
