#!/bin/bash
# Extract DINOv2 (visual) and AST (audio) features for ALL movies.
#
# Each script loads its model ONCE, then processes all 17 movie/stimulus
# combinations (friends/s1-s7, movie10/4, ood/6) in a single process.
#
# Memory (24GB VRAM):
#   DINOv2-giant (fp16): ~5GB VRAM, ~1136M params, dim=1536
#   AST:                 ~1GB VRAM, ~86M params,   dim=768
#
# Usage:
#   bash src/data/features/extract_dinov2_ast.sh             # full
#   bash src/data/features/extract_dinov2_ast.sh --dry_run   # test 1 clip/combo

EXTRA_ARGS="${@}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================"
echo " DINOv2 + AST Feature Extraction"
echo " Args: ${EXTRA_ARGS:-full run}"
echo "========================================================"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null
echo ""

# --- Stage 1: DINOv2-giant (~22GB VRAM dynamically using max_frames=4000) ---
echo "=== [1/2] DINOv2-giant (1536-D visual) ==="
python -u "${SCRIPT_DIR}/extract_dinov2.py" \
    --model_variant giant \
    --movie_type all \
    --device cuda \
    --num_temporal_pools 4 \
    ${EXTRA_ARGS}
DINOV2_EXIT=$?

# Clean VRAM between models
python3 -c "import torch,gc; gc.collect(); torch.cuda.empty_cache()" 2>/dev/null
sleep 2

# --- Stage 2: AST (~8GB VRAM with batch_trs=1024) ---
echo ""
echo "=== [2/2] AST (768-D audio) ==="
python -u "${SCRIPT_DIR}/extract_ast.py" \
    --movie_type all \
    --device cuda \
    --batch_trs 1024 \
    ${EXTRA_ARGS}
AST_EXIT=$?

# --- Summary ---
echo ""
echo "========================================================"
echo " DINOv2: exit=$DINOV2_EXIT  |  AST: exit=$AST_EXIT"
echo " Output: Data/features_npy_pooled/{dinov2_giant,ast}/"
echo "========================================================"

exit $((DINOV2_EXIT + AST_EXIT))
