#!/bin/bash
# Extract all TRIBE-based features for a given movie type and season.
# Runs each modality in a SEPARATE Python process so that when the process
# exits, ALL RAM and VRAM used by the model are fully freed by the OS.
#
# Usage:
#   bash extract_all.sh [movie_type] [stimulus_type] [device]
#   bash extract_all.sh friends s1 cuda
#   bash extract_all.sh friends s1 cuda --dry_run
#
# Memory requirements (approximate):
#   Audio  (Wav2Vec-BERT 2.0): ~2GB VRAM, ~4GB RAM
#   Text   (LLaMA 3.2 3B):    ~7GB VRAM, ~12GB RAM
#   Video  (V-JEPA 2 ViT-G):  ~20GB VRAM, ~25GB RAM
#
# Each modality runs independently — if one fails, the others still run.

# NOTE: No "set -e" here! Each modality handles errors independently.

MOVIE_TYPE="${1:-friends}"
STIMULUS_TYPE="${2:-s1}"
DEVICE="${3:-auto}"
EXTRA_ARGS="${@:4}"  # e.g. --dry_run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAIL_COUNT=0

# ---------------------------------------------------------------------------
# Helper: print memory stats
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
# Helper: clean up memory between modalities
# ---------------------------------------------------------------------------
cleanup_memory() {
    echo ""
    echo ">>> Cleaning up RAM and VRAM..."
    sync
    if sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
        echo "    Dropped OS page cache."
    else
        echo "    (Skipped drop_caches — no sudo access. This is fine.)"
    fi
    python3 -c "
import torch, gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f'    CUDA cache cleared. Allocated: {torch.cuda.memory_allocated()/1e6:.0f}MB')
" 2>/dev/null || true
    sleep 2
    print_mem_stats
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "================================================"
echo " TRIBE Feature Extraction Pipeline"
echo " Movie: ${MOVIE_TYPE} / ${STIMULUS_TYPE}"
echo " Device: ${DEVICE}"
echo " Extra args: ${EXTRA_ARGS}"
echo "================================================"
print_mem_stats

# --- 1/3: Audio (smallest model, ~2GB VRAM) ---
echo ""
echo "=== [1/3] Extracting AUDIO features (Wav2Vec-BERT 2.0) ==="
echo "    Expected: ~2GB VRAM, ~4GB RAM"
if python "${SCRIPT_DIR}/extract_audio.py" \
    --movie_type "${MOVIE_TYPE}" \
    --stimulus_type "${STIMULUS_TYPE}" \
    --device "${DEVICE}" \
    ${EXTRA_ARGS}; then
    echo "    ✅ Audio extraction succeeded."
else
    echo "    ❌ Audio extraction FAILED (exit code $?)."
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi
cleanup_memory

# --- 2/3: Text (medium model, ~7GB VRAM) ---
echo ""
echo "=== [2/3] Extracting TEXT features (LLaMA 3.2 3B) ==="
echo "    Expected: ~7GB VRAM, ~12GB RAM"
if python "${SCRIPT_DIR}/extract_text.py" \
    --movie_type "${MOVIE_TYPE}" \
    --stimulus_type "${STIMULUS_TYPE}" \
    --device "${DEVICE}" \
    ${EXTRA_ARGS}; then
    echo "    ✅ Text extraction succeeded."
else
    echo "    ❌ Text extraction FAILED (exit code $?)."
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi
cleanup_memory

# --- 3/3: Video (largest model, ~20GB VRAM) ---
echo ""
echo "=== [3/3] Extracting VIDEO features (V-JEPA 2 ViT-G) ==="
echo "    Expected: ~20GB VRAM, ~25GB RAM"
echo "    WARNING: This is the most memory-intensive step."
if python "${SCRIPT_DIR}/extract_video.py" \
    --movie_type "${MOVIE_TYPE}" \
    --stimulus_type "${STIMULUS_TYPE}" \
    --device "${DEVICE}" \
    --batch_timepoints 12 \
    ${EXTRA_ARGS}; then
    echo "    ✅ Video extraction succeeded."
else
    echo "    ❌ Video extraction FAILED (exit code $?)."
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi
cleanup_memory

# --- Summary ---
echo ""
echo "================================================"
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo " ✅ All 3 modalities extracted successfully!"
else
    echo " ⚠️  ${FAIL_COUNT}/3 modalities FAILED. Check logs above."
fi
echo " Output: Data/features/{video,audio,text}/"
echo "================================================"

exit $FAIL_COUNT
