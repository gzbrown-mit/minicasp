#!/bin/bash
#SBATCH -p pi_melkin
#SBATCH --nodelist=node3616
#SBATCH --job-name=minicasp_pipeline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/gzbrown/minicasp/slurm/results/%x-%j.out

set -euo pipefail

# ------------------------------
# MiniCASP Slurm pipeline driver
# ------------------------------
# Usage:
#   sbatch slurm/run_minicasp.sbatch smoke
#   sbatch slurm/run_minicasp.sbatch train
#   sbatch slurm/run_minicasp.sbatch audit
#   sbatch slurm/run_minicasp.sbatch sweep
#   sbatch slurm/run_minicasp.sbatch all
#
# Optional override of key paths:
#   sbatch --export=ALL,CONDA_ENV=minicasp,RUN_TAG=myrun \
#          slurm/run_minicasp.sbatch sweep
# Optional override of execution commands (required if minicasp.scripts is absent):
#   sbatch --export=ALL,STEP0_CMD="python /path/to/step0_templates.py ...",\
#TRAIN_CMD="python /path/to/step1_train.py ...",AUDIT_CMD="python /path/to/step2_audit.py ...",\
#SWEEP_CMD="python /path/to/train_audit_sweep.py ..." slurm/run_minicasp.sbatch train
#
# Notes:
# - This script can auto-use local scripts in ./scripts if present.
# - Otherwise provide STEP0_CMD/TRAIN_CMD/AUDIT_CMD/SWEEP_CMD via --export.

STEP="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/gzbrown/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-minicasp}"
REPO_DIR="${REPO_DIR:-$DEFAULT_REPO_DIR}"
RUN_TAG="${RUN_TAG:-$(date +%y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-$REPO_DIR/results/runs/${RUN_TAG}_sweep}"

PAIRS_CACHE="${PAIRS_CACHE:-$REPO_DIR/results/templates/pairs_r1_min5.jsonl.gz}"
TEMPLATES_CACHE="${TEMPLATES_CACHE:-$REPO_DIR/results/templates/templates_r1_min5.json.gz}"
BUYABLES_JSON="${BUYABLES_JSON:-$REPO_DIR/minicasp/data/buyables/buyables.jsonl.gz}"
BUYABLES_CACHE="${BUYABLES_CACHE:-$REPO_DIR/minicasp/data/buyables/buyables_smiles.txt.gz}"
STEP0_CMD="${STEP0_CMD:-}"
TRAIN_CMD="${TRAIN_CMD:-}"
AUDIT_CMD="${AUDIT_CMD:-}"
SWEEP_CMD="${SWEEP_CMD:-}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
# Ensure conda-provided C/C++ runtimes are used (fixes GLIBCXX mismatches on older nodes)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD || true

cd "$REPO_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p "$REPO_DIR/slurm" "$REPO_DIR/results/runs"

echo "== MiniCASP job start =="
echo "HOST: $(hostname)"
echo "STEP: $STEP"
echo "ENV : $CONDA_ENV"
echo "RUN : $RUN_DIR"

# Fast-fail checks before heavy work
python - <<'PY'
import importlib
required = [
    "minicasp",
    "minicasp.stock.config",
    "minicasp.search.search",
    "minicasp.analysis.tree_analysis",
]
for mod in required:
    importlib.import_module(mod)
print("Core imports OK")
PY

if [[ -z "$STEP0_CMD" && -f "$REPO_DIR/scripts/step0_templates.py" ]]; then
  STEP0_CMD="python \"$REPO_DIR/scripts/step0_templates.py\" --out_pairs \"$PAIRS_CACHE\" --out_templates \"$TEMPLATES_CACHE\""
fi
if [[ -z "$TRAIN_CMD" && -f "$REPO_DIR/scripts/step1_train.py" ]]; then
  TRAIN_CMD="python \"$REPO_DIR/scripts/step1_train.py\" --pairs_cache \"$PAIRS_CACHE\" --templates_cache \"$TEMPLATES_CACHE\" --run_dir \"$RUN_DIR\""
fi
if [[ -z "$AUDIT_CMD" && -f "$REPO_DIR/scripts/step2_audit.py" ]]; then
  AUDIT_CMD="python \"$REPO_DIR/scripts/step2_audit.py\" --run_dir \"$RUN_DIR\" --buyables_jsonl_gz \"$BUYABLES_JSON\" --buyables_cache_txt_gz \"$BUYABLES_CACHE\""
fi
if [[ -z "$SWEEP_CMD" && -f "$REPO_DIR/scripts/train_audit_sweep.py" ]]; then
  SWEEP_CMD="python \"$REPO_DIR/scripts/train_audit_sweep.py\" --pairs_cache \"$PAIRS_CACHE\" --templates_cache \"$TEMPLATES_CACHE\" --run_dir \"$RUN_DIR\" --targets_n 50 --model_type sgd --max_iter 50 --topk_templates 10,25,50 --max_outcomes_per_template 10,25,50 --max_depth 4,6,8 --max_expansions 100,300,600 --time_limit_s 5,10,20 --buyables_jsonl_gz \"$BUYABLES_JSON\" --buyables_cache_txt_gz \"$BUYABLES_CACHE\""
fi

require_cmd() {
  local var_name="$1"
  local cmd_val="$2"
  if [[ -z "$cmd_val" ]]; then
    echo "Missing command: $var_name"
    echo "This checkout has no minicasp.scripts modules."
    echo "Submit with overrides, e.g.:"
    echo "  sbatch --export=ALL,${var_name}='python /path/to/script.py ...' slurm/run_minicasp.sbatch $STEP"
    exit 3
  fi
}

run_smoke() {
  echo "[STEP smoke] verifying env + imports"
  python -m compileall -q minicasp
  python - <<'PY'
from minicasp.stock.config import Configuration
cfg = Configuration()
print("Config OK; default max_transforms:", cfg.search.max_transforms)
PY
}

run_train() {
  echo "[STEP train] template extraction + model training"
  require_cmd "STEP0_CMD" "$STEP0_CMD"
  require_cmd "TRAIN_CMD" "$TRAIN_CMD"
  eval "$STEP0_CMD"
  eval "$TRAIN_CMD"
}

run_audit() {
  echo "[STEP audit] evaluate one trained run"
  require_cmd "AUDIT_CMD" "$AUDIT_CMD"
  eval "$AUDIT_CMD"
}

run_sweep() {
  echo "[STEP sweep] train + audit hyperparameter sweep"
  require_cmd "SWEEP_CMD" "$SWEEP_CMD"
  eval "$SWEEP_CMD"
}

case "$STEP" in
  smoke) run_smoke ;;
  train) run_train ;;
  audit) run_audit ;;
  sweep) run_sweep ;;
  all)
    run_smoke
    run_train
    run_audit
    ;;
  *)
    echo "Unknown step: $STEP"
    echo "Valid steps: smoke | train | audit | sweep | all"
    exit 2
    ;;
esac

echo "== MiniCASP job complete =="
