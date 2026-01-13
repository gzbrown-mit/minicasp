#!/bin/bash
#SBATCH -p pi_melkin
#SBATCH --nodelist=node3616
#SBATCH --job-name=minicasp_s0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gzbrown/minicasp/slurm/step0/%x-%j.out


set -e
source /home/gzbrown/miniconda3/etc/profile.d/conda.sh
conda activate higherlev_retro

cd /home/gzbrown/minicasp
export PYTHONPATH="$PWD:$PYTHONPATH"

CSV="/home/gzbrown/minicasp/data/reactions/uspto_original.csv"

OUTDIR="/home/gzbrown/minicasp/results/templates"
mkdir -p "$OUTDIR"

python -m minicasp.scripts.step0_templates \
  --csv "$CSV" \
  --rxn_col rxn_smiles \
  --limit 50000 \
  --shuffle \
  --seed 0 \
  --template_radius 1 \
  --template_min_count 5 \
  --templates_out "$OUTDIR/templates_r1_min5.json.gz" \
  --pairs_out "$OUTDIR/pairs_r1_min5.jsonl.gz"

echo "STEP0_DONE templates=$OUTDIR/templates_r1_min5.json.gz pairs=$OUTDIR/pairs_r1_min5.jsonl.gz"
