#!/bin/bash
#SBATCH -p pi_melkin
#SBATCH --nodelist=node3616
#SBATCH --job-name=minicasp_step2
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

CSV="/home/gzbrown/minicasp/data/reactions/uspto_original.csv"

# Set this to the RUN_ID printed by step1:
RUN_ID="${RUN_ID:?Set RUN_ID when submitting step2, e.g. sbatch --export=RUN_ID=... slurm/slurm_step2_audit.sbatch}"

python -m minicasp.scripts.step2_audit \
  --csv "$CSV" \
  --rxn_col rxn_smiles \
  --train_limit 50000 \
  --targets_n 100 \
  --seed 0 \
  --run_id "$RUN_ID" \
  --max_depth 6 \
  --max_expansions 300 \
  --topk_templates 25 \
  --max_outcomes_per_template 25 \
  --time_limit_s 10.0

echo "Audit: /home/gzbrown/minicasp/results/audits/$RUN_ID/audit.json"
