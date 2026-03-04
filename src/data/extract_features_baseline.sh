#!/bin/bash
# =============================================================================
# extract_features_baseline.sh
# Extract visual, audio, and language features for ALL movie splits.
#
# Feature extraction is stimulus-based (not subject-specific).
# Run this once to generate features for both TRAIN and TEST sets.
#
# Train data: Friends s1-s6 + Movie10 (bourne, figures, life, wolf)
# Test data:  Friends s7
#
# Usage:
#   cd /media/hung/data1/codes/multimodal
#   bash src/data/extract_features_baseline.sh
#   bash src/data/extract_features_baseline.sh visual   # only visual
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Which modalities to run (default: all)
# Override via: bash extract_features_baseline.sh visual
MODALITIES=("visual" "audio" "language")
if [ $# -ge 1 ]; then
    MODALITIES=("$@")
fi

# Batch size for GPU inference (default: 128)
# Override via: BATCH_SIZE=64 bash extract_features_baseline.sh
BATCH_SIZE=${BATCH_SIZE:-128}

# ----------------------------- TRAIN SPLITS ----------------------------------
# Friends: seasons 1-6 (train) and season 7 (test)
FRIENDS_TRAIN_SEASONS=("s1" "s2" "s3" "s4" "s5" "s6")
FRIENDS_TEST_SEASONS=("s7")

# Movie10: all 4 movies (train only)
MOVIE10_MOVIES=("bourne" "figures" "life" "wolf")

# =============================================================================
# Helper: run extraction and log result
# =============================================================================
run_extraction() {
    local movie_type=$1
    local stimulus_type=$2
    local modality=$3

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${modality^^}] movie_type=${movie_type}  stimulus_type=${stimulus_type}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python feature_extraction.py \
        --movie_type    "$movie_type"    \
        --stimulus_type "$stimulus_type" \
        --modality      "$modality"      \
        --batch_size    "$BATCH_SIZE"

    echo "  ✅ Done: ${movie_type}/${stimulus_type} [${modality}]"
}

# =============================================================================
# Main loop
# =============================================================================
for MODALITY in "${MODALITIES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  EXTRACTING: ${MODALITY^^} FEATURES"
    echo "============================================================"

    # --- TRAIN: Friends s1-s6 ---
    echo ""
    echo ">>> [TRAIN] Friends seasons 1-6"
    for SEASON in "${FRIENDS_TRAIN_SEASONS[@]}"; do
        run_extraction "friends" "$SEASON" "$MODALITY"
    done

    # --- TEST: Friends s7 ---
    echo ""
    echo ">>> [TEST] Friends season 7"
    for SEASON in "${FRIENDS_TEST_SEASONS[@]}"; do
        run_extraction "friends" "$SEASON" "$MODALITY"
    done

    # --- TRAIN: Movie10 ---
    echo ""
    echo ">>> [TRAIN] Movie10"
    for MOVIE in "${MOVIE10_MOVIES[@]}"; do
        run_extraction "movie10" "$MOVIE" "$MODALITY"
    done

    echo ""
    echo "✅✅ FINISHED all ${MODALITY^^} features ✅✅"
done

echo ""
echo "============================================================"
echo "  ALL FEATURE EXTRACTION COMPLETE"
echo "  Output: Data/features/raw/"
echo "============================================================"
