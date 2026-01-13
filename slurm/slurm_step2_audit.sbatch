#!/bin/bash
#SBATCH -p pi_melkin
#SBATCH --nodelist=node3616
#SBATCH --job-name=minicasp_s2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gzbrown/minicasp/slurm/slurm-step2-%j.out

set -e
source /home/gzbrown/miniconda3/etc/profile.d/conda.sh
conda activate higherlev_retro

cd /home/gzbrown/minicasp
export PYTHONPATH="$PWD:$PYTHONPATH"

RUN_DIR="$(ls -td /home/gzbrown/minicasp/results/runs/* | head -n 1)"
echo "Using RUN_DIR=$RUN_DIR"

python -m minicasp.scripts.step2_audit \
  --run_dir "$RUN_DIR" \
  --targets_n 100 \
  --seed 0 \
  --max_depth 6 \
  --max_expansions 300 \
  --topk_templates 25 \
  --max_outcomes_per_template 25 \
  --time_limit_s 10.0 \
  --buyables_jsonl_gz /home/gzbrown/minicasp/data/buyables/buyables.jsonl.gz \
  --buyables_cache_txt_gz /home/gzbrown/minicasp/data/buyables/buyables_smiles.txt.gz

echo "STEP2_DONE RUN_DIR=$RUN_DIR"
