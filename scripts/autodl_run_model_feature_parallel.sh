#!/usr/bin/env bash
set -u

# AutoDL launcher for the Lisan model-feature sweep.
# It runs two models at a time to reduce CUDA OOM risk, writes per-model logs,
# resumes completed work, and shuts down the instance after both batches finish.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-lisan-openhgnn}"
GPU="${GPU:-0}"
N_TRIALS="${N_TRIALS:-50}"
MAX_EPOCH="${MAX_EPOCH:-200}"
PATIENCE="${PATIENCE:-20}"
SEEDS="${SEEDS:-0 1 2 3 4}"
FEATURE_MODES="${FEATURE_MODES:-A B C D E}"
DATASETS="${DATASETS:-lisan-acm lisan-dblp}"
TASKS="${TASKS:-node_classification link_prediction}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/lisan_model_feature_sweep_parallel}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
SHUTDOWN_AFTER="${SHUTDOWN_AFTER:-1}"

cd "$PROJECT_ROOT" || exit 1

mkdir -p "$LOG_DIR"
LAUNCH_LOG="$LOG_DIR/model_feature_parallel_launcher_$(date +%Y%m%d_%H%M%S).log"
ALL_LOG="$LOG_DIR/model_sweep_all.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LAUNCH_LOG"
}

fail_preflight() {
  log "PRECHECK FAILED: $*"
  log "No training launched and no shutdown requested."
  exit 1
}

[ -f "$CONDA_SH" ] || fail_preflight "Cannot find conda init script: $CONDA_SH"
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$ENV_NAME" || fail_preflight "Cannot activate conda env: $ENV_NAME"

export DGLBACKEND=pytorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

[ -f "scripts/run_lisan_model_feature_sweep.py" ] || fail_preflight "Missing scripts/run_lisan_model_feature_sweep.py"
[ -d "OpenHGNN" ] || fail_preflight "Missing OpenHGNN directory"
[ -d "OpenHGNN/openhgnn/dataset/lisan_hgb" ] || fail_preflight "Missing converted graph directory OpenHGNN/openhgnn/dataset/lisan_hgb"

python - <<'PY' || fail_preflight "Python/OpenHGNN/DGL import check failed"
import torch
import dgl
from openhgnn import Experiment

print("torch", torch.__version__)
print("dgl", dgl.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    g = dgl.graph(([0], [1])).to("cuda")
    print("dgl graph device", g.device)
print("openhgnn Experiment import ok")
PY

log "Precheck passed."
log "PROJECT_ROOT=$PROJECT_ROOT"
log "ENV_NAME=$ENV_NAME GPU=$GPU N_TRIALS=$N_TRIALS MAX_EPOCH=$MAX_EPOCH PATIENCE=$PATIENCE"
log "SEEDS=$SEEDS"
log "FEATURE_MODES=$FEATURE_MODES"
log "DATASETS=$DATASETS"
log "TASKS=$TASKS"
log "OUTPUT_ROOT=$OUTPUT_ROOT"

run_model() {
  local model="$1"
  local out="$OUTPUT_ROOT/$model"
  local log_file="$LOG_DIR/model_sweep_${model}.log"

  {
    echo "===== ${model} start: $(date) ====="
    echo "output_root=${out}"
  } | tee -a "$log_file"

  # Intentional word splitting for CLI list arguments from env strings.
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_lisan_model_feature_sweep.py \
    --models "$model" \
    --feature_modes $FEATURE_MODES \
    --datasets $DATASETS \
    --tasks $TASKS \
    --n_trials "$N_TRIALS" \
    --seeds $SEEDS \
    --max_epoch "$MAX_EPOCH" \
    --patience "$PATIENCE" \
    --gpu 0 \
    --output_root "$out" \
    --resume >> "$log_file" 2>&1

  local status=$?
  echo "===== ${model} finished: $(date), status=${status} =====" | tee -a "$log_file"
  return "$status"
}

run_batch() {
  local first="$1"
  local second="$2"

  log "Starting batch: $first + $second"
  run_model "$first" &
  local pid_first=$!
  run_model "$second" &
  local pid_second=$!

  wait "$pid_first"
  local status_first=$?
  wait "$pid_second"
  local status_second=$?

  log "Batch finished: $first=$status_first $second=$status_second"
  echo "[$(date '+%F %T')] $first=$status_first $second=$status_second" >> "$ALL_LOG"
}

run_batch HGT SimpleHGN
run_batch GCN GAT

log "All model-feature sweeps finished."
sync

if [ "$SHUTDOWN_AFTER" = "1" ]; then
  log "Shutting down AutoDL instance now."
  /usr/bin/shutdown
else
  log "SHUTDOWN_AFTER=$SHUTDOWN_AFTER, leaving instance running."
fi
