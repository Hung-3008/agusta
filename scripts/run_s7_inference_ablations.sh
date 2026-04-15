#!/usr/bin/env bash
set -euo pipefail

# Run in-distribution (Friends S7) inference ablations for BrainFlow:
# - Strategy 2 only (pruned sampling)
# - Strategy 3 only (singularity avoidance)
# - Strategy 4 only (parcel-stitch ensemble; requires S6 calibration)
# - Strategy 2+3+4 combined

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_CONFIG="${BASE_CONFIG:-src/configs/brainflow.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/brainflow_seq2seq_v6_ditx/best.pt}"
DEVICE="${DEVICE:-cuda}"

# Strategy defaults (override via env vars if needed)
PRUNE_K="${PRUNE_K:-5}"
N_SEEDS="${N_SEEDS:-10}"
TIME_GRID_MAX="${TIME_GRID_MAX:-0.99}"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

make_cfg() {
  local out_dir="$1"
  local cfg_path="$2"

  awk -v out="$out_dir" '
    BEGIN { replaced=0 }
    /^output_dir:[[:space:]]*/ {
      print "output_dir: " out
      replaced=1
      next
    }
    { print }
    END {
      if (!replaced) {
        print "output_dir: " out
      }
    }
  ' "$BASE_CONFIG" > "$cfg_path"
}

run_eval() {
  local cfg_path="$1"
  local session="$2"
  shift 2
  python src/evaluate_brainflow.py \
    --config "$cfg_path" \
    --eval_session "$session" \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    "$@"
}

echo "[Info] Root:        $ROOT_DIR"
echo "[Info] Base config: $BASE_CONFIG"
echo "[Info] Checkpoint:  $CHECKPOINT"
echo "[Info] Device:      $DEVICE"

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "[Error] Config not found: $BASE_CONFIG" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[Error] Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

echo
echo "========== Experiment 1/4: Strategy 2 only (Pruned Sampling) =========="
OUT1="outputs/brainflow_seq2seq_v6_ditx_s7_id_strategy2_pruned"
CFG1="$TMP_DIR/strategy2.yaml"
make_cfg "$OUT1" "$CFG1"
run_eval "$CFG1" s7 \
  --ensemble_mode none \
  --n_seeds 1 \
  --use_pruned_sampling \
  --prune_k "$PRUNE_K" \
  --time_grid_max 1.0 \
  --no-final_jump

echo
echo "========== Experiment 2/4: Strategy 3 only (Singularity Avoidance) =========="
OUT2="outputs/brainflow_seq2seq_v6_ditx_s7_id_strategy3_singularity_avoid"
CFG2="$TMP_DIR/strategy3.yaml"
make_cfg "$OUT2" "$CFG2"
run_eval "$CFG2" s7 \
  --ensemble_mode none \
  --n_seeds 1 \
  --no-use_pruned_sampling \
  --time_grid_max "$TIME_GRID_MAX" \
  --final_jump

echo
echo "========== Experiment 3/4: Strategy 4 only (Parcel-Stitch Ensemble) =========="
OUT3="outputs/brainflow_seq2seq_v6_ditx_s7_id_strategy4_parcel_stitch"
CFG3="$TMP_DIR/strategy4.yaml"
make_cfg "$OUT3" "$CFG3"

echo "[Step] S6 calibration for Strategy 4..."
run_eval "$CFG3" s6 \
  --ensemble_mode parcel_stitch \
  --n_seeds "$N_SEEDS" \
  --no-use_pruned_sampling \
  --time_grid_max 1.0 \
  --no-final_jump

echo "[Step] S7 submission for Strategy 4..."
run_eval "$CFG3" s7 \
  --ensemble_mode parcel_stitch \
  --n_seeds "$N_SEEDS" \
  --no-use_pruned_sampling \
  --time_grid_max 1.0 \
  --no-final_jump

echo
echo "========== Experiment 4/4: Strategy 2+3+4 combined =========="
OUT4="outputs/brainflow_seq2seq_v6_ditx_s7_id_strategy234_combo"
CFG4="$TMP_DIR/strategy234.yaml"
make_cfg "$OUT4" "$CFG4"

echo "[Step] S6 calibration for Strategy 2+3+4..."
run_eval "$CFG4" s6 \
  --ensemble_mode parcel_stitch \
  --n_seeds "$N_SEEDS" \
  --use_pruned_sampling \
  --prune_k "$PRUNE_K" \
  --time_grid_max "$TIME_GRID_MAX" \
  --final_jump

echo "[Step] S7 submission for Strategy 2+3+4..."
run_eval "$CFG4" s7 \
  --ensemble_mode parcel_stitch \
  --n_seeds "$N_SEEDS" \
  --use_pruned_sampling \
  --prune_k "$PRUNE_K" \
  --time_grid_max "$TIME_GRID_MAX" \
  --final_jump

echo
echo "========== Done =========="
echo "S7 outputs (submission.npy + submission.zip) are under:"
echo "  outputs/submissions/$(basename "$OUT1")/s7"
echo "  outputs/submissions/$(basename "$OUT2")/s7"
echo "  outputs/submissions/$(basename "$OUT3")/s7"
echo "  outputs/submissions/$(basename "$OUT4")/s7"
