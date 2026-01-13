#!/bin/bash
#SBATCH -p pi_melkin
#SBATCH --nodelist=node3616
#SBATCH --job-name=minicasp_s2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gzbrown/minicasp/slurm/step2/%x-%j.out


set -e
source /home/gzbrown/miniconda3/etc/profile.d/conda.sh
conda activate higherlev_retro

cd /home/gzbrown/minicasp
export PYTHONPATH="$PWD:$PYTHONPATH"

RUN_DIR="$(ls -td /home/gzbrown/minicasp/results/runs/* | head -n 1)"
echo "Using RUN_DIR=$RUN_DIR"

# check to make sure model trained and run are the same
EXPECT_MODEL_TYPE=""

python -m minicasp.scripts.step2_audit \
  --run_dir "$RUN_DIR" \
  --targets_n 200 \
  --seed 0 \
  --max_depth 8 \
  --max_expansions 800 \
  --topk_templates 100 \
  --max_outcomes_per_template 25 \
  --time_limit_s 30.0 \
  --buyables_jsonl_gz /home/gzbrown/minicasp/data/buyables/buyables.jsonl.gz \
  --buyables_cache_txt_gz /home/gzbrown/minicasp/data/buyables/buyables_smiles.txt.gz \
  --expect_model_type "$EXPECT_MODEL_TYPE"

echo "STEP2_DONE RUN_DIR=$RUN_DIR EXPECT_MODEL_TYPE=$EXPECT_MODEL_TYPE"
