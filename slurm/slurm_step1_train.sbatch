#!/bin/bash
#SBATCH -p pi_melkin
#SBATCH --nodelist=node3616
#SBATCH --job-name=minicasp_s1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gzbrown/minicasp/slurm/slurm-step1-%j.out

set -e
source /home/gzbrown/miniconda3/etc/profile.d/conda.sh
conda activate higherlev_retro

cd /home/gzbrown/minicasp
export PYTHONPATH="$PWD:$PYTHONPATH"

RUN_ID="$(date +%y%m%d-%H%M%S)"
RUN_DIR="/home/gzbrown/minicasp/results/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

TEMPLATES="/home/gzbrown/minicasp/results/templates/templates_r1_min5.json.gz"
PAIRS="/home/gzbrown/minicasp/results/templates/pairs_r1_min5.jsonl.gz"

python -m minicasp.scripts.step1_train \
  --pairs_cache "$PAIRS" \
  --templates_cache "$TEMPLATES" \
  --run_dir "$RUN_DIR" \
  --test_size 0.2 \
  --split_mode ordered_group \
  --split_seed 0 \
  --model_seed 0 \
  --fp_radius 2 \
  --n_bits 2048 \
  --max_iter 25

echo "STEP1_DONE RUN_DIR=$RUN_DIR"
