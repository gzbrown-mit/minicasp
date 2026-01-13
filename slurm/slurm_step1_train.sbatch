#!/bin/bash
#SBATCH -p pi_melkin
#SBATCH --nodelist=node3616
#SBATCH --job-name=minicasp_s1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gzbrown/minicasp/slurm/step1/%x-%j.out


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

# INPUT MODEL TYPE GISELLE:
MODEL_TYPE="mlp_sklearn"          # sgd | mlp | mlp_torch | mlp_sklearn
SPLIT_TYPE="random_group" # ordered_group | random_group

echo "Using MODEL_TYPE=$MODEL_TYPE"
echo "Using SPLIT_TYPE=$SPLIT_TYPE"

python -m minicasp.scripts.step1_train \
  --pairs_cache "$PAIRS" \
  --templates_cache "$TEMPLATES" \
  --run_dir "$RUN_DIR" \
  --test_size 0.2 \
  --split_mode "$SPLIT_TYPE" \
  --split_seed 0 \
  --model_seed 0 \
  --fp_radius 2 \
  --n_bits 2048 \
  --max_iter 25 \
  --model_type "$MODEL_TYPE" \
  --mlp_hidden "1024,1024" \
  --mlp_epochs 8 \
  --mlp_batch_size 256 \
  --mlp_lr 1e-3 \
  --mlp_dropout 0.1

echo "STEP1_DONE RUN_DIR=$RUN_DIR MODEL_TYPE=$MODEL_TYPE"
