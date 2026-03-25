#!/bin/bash
#SBATCH -p pi_melkin
#SBATCH --nodelist=node3616
#SBATCH --job-name=minicasp_mcts
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gzbrown/minicasp/slurm/results/%x-%j.out

set -euo pipefail

# ------------------------------
# MiniCASP MCTS Search Runner
# ------------------------------
# Usage:
#   sbatch slurm/run_mcts.sbatch                          # smoke test only
#   sbatch slurm/run_mcts.sbatch search "CCO"             # search single target
#   sbatch slurm/run_mcts.sbatch search_file targets.txt  # search from file
#   sbatch slurm/run_mcts.sbatch train                    # retrain policy model
#
# Override model paths:
#   sbatch --export=ALL,MINICASP_MODEL=/path/to/model.onnx slurm/run_mcts.sbatch search "CCO"

STEP="${1:-smoke}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Conda setup
CONDA_SH="${CONDA_SH:-/home/gzbrown/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-minicasp}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD || true

cd "$REPO_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Default model paths (override via environment)
export MINICASP_MODEL="${MINICASP_MODEL:-$REPO_DIR/minicasp/data/models/uspto_model.onnx}"
export MINICASP_TEMPLATES="${MINICASP_TEMPLATES:-$REPO_DIR/minicasp/data/models/uspto_templates.csv.gz}"
export MINICASP_STOCK="${MINICASP_STOCK:-$REPO_DIR/minicasp/data/models/zinc_stock.hdf5}"
CONFIG="${CONFIG:-$REPO_DIR/configs/mcts_config.yml}"

mkdir -p "$REPO_DIR/slurm/results" "$REPO_DIR/results"

echo "== MiniCASP job start =="
echo "HOST: $(hostname)"
echo "STEP: $STEP"
echo "ENV : $CONDA_ENV"
echo "PYTHON: $(python --version)"

# ---- Smoke test (always runs first) ----
run_smoke() {
  echo "[smoke] Verifying imports..."
  python -c "
import importlib
for mod in ['minicasp', 'minicasp.stock.config', 'minicasp.search.search',
            'minicasp.analysis.tree_analysis', 'minicasp.chem', 'minicasp.policy']:
    importlib.import_module(mod)
print('Core imports OK')

from minicasp.stock.config import Configuration
cfg = Configuration()
print(f'Configuration OK: max_transforms={cfg.search.max_transforms}')
"
  python -m compileall -q minicasp
  echo "[smoke] All checks passed."
}

# ---- MCTS search ----
run_search() {
  local target="$1"
  echo "[search] Target: $target"
  echo "[search] Model: $MINICASP_MODEL"
  echo "[search] Templates: $MINICASP_TEMPLATES"
  echo "[search] Stock: $MINICASP_STOCK"

  local ts
  ts="$(date +%y%m%d-%H%M%S)"
  local outfile="$REPO_DIR/results/mcts_${ts}.json"

  python scripts/run_mcts.py \
    --config "$CONFIG" \
    --target "$target" \
    --output "$outfile"
}

run_search_file() {
  local targets_file="$1"
  echo "[search_file] Targets: $targets_file"

  local ts
  ts="$(date +%y%m%d-%H%M%S)"
  local outfile="$REPO_DIR/results/mcts_${ts}.json"

  python scripts/run_mcts.py \
    --config "$CONFIG" \
    --targets_file "$targets_file" \
    --output "$outfile"
}

# ---- Train / retrain policy model ----
run_train() {
  echo "[train] Retraining expansion policy model..."
  python scripts/train_policy.py \
    --config "$CONFIG" \
    --output_dir "$REPO_DIR/results/training/$(date +%y%m%d-%H%M%S)" \
    "${@:1}"
}

# ---- Dispatch ----
run_smoke

case "$STEP" in
  smoke)
    echo "Smoke test complete."
    ;;
  search)
    if [[ -z "${2:-}" ]]; then
      echo "Usage: sbatch slurm/run_mcts.sbatch search \"SMILES\""
      exit 2
    fi
    run_search "$2"
    ;;
  search_file)
    if [[ -z "${2:-}" ]]; then
      echo "Usage: sbatch slurm/run_mcts.sbatch search_file targets.txt"
      exit 2
    fi
    run_search_file "$2"
    ;;
  train)
    run_train "${@:2}"
    ;;
  *)
    echo "Unknown step: $STEP"
    echo "Valid steps: smoke | search <SMILES> | search_file <file> | train [args]"
    exit 2
    ;;
esac

echo "== MiniCASP job complete =="
