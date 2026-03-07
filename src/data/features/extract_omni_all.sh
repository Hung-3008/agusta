#!/bin/bash
# =============================================================================
# Extract Qwen2.5-Omni features for ALL sessions and movies.
#
# Usage:
#   bash extract_omni_all.sh                      # Full extraction
#   bash extract_omni_all.sh --no_flash_attn      # Without FlashAttention
#   bash extract_omni_all.sh --dry_run             # Test mode (1 clip each)
#
# Data overview (385 clips total):
#   Friends: s1(48), s2(48), s3(50), s4(48), s5(48), s6(50), s7(49) = 341 clips
#   Movie10: bourne(10), figures(12), life(5), wolf(17) = 44 clips
#
# Estimated time (~1.3 TRs/sec, ~8min/clip):
#   Full run: ~50 hours
#   With flash-attn on 48GB VRAM: ~35 hours
# =============================================================================

EXTRA_ARGS="${@}"  # Pass through all args (e.g. --no_flash_attn, --dry_run)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACT_SCRIPT="${SCRIPT_DIR}/extract_omni.py"

TOTAL_FAIL=0
TOTAL_DONE=0

# ---------------------------------------------------------------------------
# Helper: print GPU/memory stats
# ---------------------------------------------------------------------------
print_mem_stats() {
    echo "--- Memory Status ---"
    free -h | head -2
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total \
                    --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=',' read -r idx used free total; do
            echo "GPU $idx: ${used}MiB used / ${total}MiB total (${free}MiB free)"
        done
    fi
    echo "---------------------"
}

# ---------------------------------------------------------------------------
# Helper: extract one session/movie
# ---------------------------------------------------------------------------
extract_one() {
    local movie_type="$1"
    local stim_type="$2"
    local label="${movie_type}/${stim_type}"

    echo ""
    echo "=========================================="
    echo "  Extracting: ${label}"
    echo "  Args: ${EXTRA_ARGS}"
    echo "=========================================="

    if python "${EXTRACT_SCRIPT}" \
        --movie_type "${movie_type}" \
        --stimulus_type "${stim_type}" \
        ${EXTRA_ARGS}; then
        echo "  ✅ ${label} succeeded."
        TOTAL_DONE=$((TOTAL_DONE + 1))
    else
        echo "  ❌ ${label} FAILED (exit code $?)."
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi

    # Cleanup between sessions
    python3 -c "
import torch, gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "================================================"
echo " Qwen2.5-Omni Feature Extraction — FULL DATASET"
echo " Extra args: ${EXTRA_ARGS}"
echo "================================================"
print_mem_stats

START_TIME=$(date +%s)

# --- Friends seasons (s1-s7) ---
echo ""
echo "############################################"
echo "#           FRIENDS (7 seasons)            #"
echo "############################################"

for season in s1 s2 s3 s4 s5 s6 s7; do
    extract_one "friends" "${season}"
done

# --- Movie10 movies ---
echo ""
echo "############################################"
echo "#           MOVIE10 (4 movies)             #"
echo "############################################"

for movie in bourne figures life wolf; do
    extract_one "movie10" "${movie}"
done

# --- Summary ---
END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "================================================"
echo " EXTRACTION COMPLETE"
echo "================================================"
echo " Total sessions processed: $((TOTAL_DONE + TOTAL_FAIL))/11"
echo " ✅ Succeeded: ${TOTAL_DONE}"
if [ "$TOTAL_FAIL" -gt 0 ]; then
    echo " ❌ Failed: ${TOTAL_FAIL}"
fi
echo " Time elapsed: ${ELAPSED} minutes"
echo " Output: Data/features/omni/"
echo "================================================"
print_mem_stats

exit $TOTAL_FAIL
