#!/bin/bash
#SBATCH -p pi_melkin
#SBATCH --nodelist=node3616
#SBATCH --job-name=minicasp_step1
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

CSV="/home/gzbrown/minicasp/data/reactions/uspto_original.csv"
RUN_ID="$(date +%y%m%d-%H%M%S)"

python -m minicasp.scripts.step1_train \
  --csv "$CSV" \
  --rxn_col rxn_smiles \
  --train_limit 50000 \
  --template_radius 1 \
  --template_min_count 5 \
  --seed 0 \
  --run_id "$RUN_ID"

echo "STEP1_DONE RUN_ID=$RUN_ID"
echo "Artifacts: /home/gzbrown/minicasp/results/artifacts/$RUN_ID"