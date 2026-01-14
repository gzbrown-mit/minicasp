#!/bin/bash
#SBATCH -p pi_melkin
#SBATCH --nodelist=node3616
#SBATCH --job-name=minicasp_train_audit_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/gzbrown/minicasp/slurm/step2/%x-%j.out

set -e
source /home/gzbrown/miniconda3/etc/profile.d/conda.sh
conda activate higherlev_retro

cd /home/gzbrown/minicasp
export PYTHONPATH="$PWD:$PYTHONPATH"

RUN_DIR="/home/gzbrown/minicasp/results/runs/$(date +%y%m%d-%H%M%S)_sweep"
PAIRS_CACHE="/home/gzbrown/minicasp/results/templates/pairs_r1_min5.jsonl.gz"
TEMPLATES_CACHE="/home/gzbrown/minicasp/results/templates/templates_r1_min5.json.gz"

echo "START train_audit_sweep"
echo "RUN_DIR=$RUN_DIR"
echo "PAIRS_CACHE=$PAIRS_CACHE"
echo "TEMPLATES_CACHE=$TEMPLATES_CACHE"

python -m minicasp.scripts.train_audit_sweep \
  --pairs_cache "$PAIRS_CACHE" \
  --templates_cache "$TEMPLATES_CACHE" \
  --run_dir "$RUN_DIR" \
  --targets_n 200 \
  --model_type sgd \
  --max_iter 50 \
  --topk_templates 10,25,50 \
  --max_outcomes_per_template 10,25,50 \
  --max_depth 4,6,8 \
  --max_expansions 100,300,600 \
  --time_limit_s 5,10,20 \
  --buyables_jsonl_gz /home/gzbrown/minicasp/data/buyables/buyables.jsonl.gz \
  --buyables_cache_txt_gz /home/gzbrown/minicasp/data/buyables/buyables_smiles.txt.gz

echo "DONE train_audit_sweep RUN_DIR=$RUN_DIR"