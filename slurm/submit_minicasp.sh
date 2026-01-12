#!/bin/bash
#SBATCH -p pi_melkin
#SBATCH --nodelist=node3616
#SBATCH --job-name=minicasp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/gzbrown/minicasp/slurm-%j.out

set -e
source /home/gzbrown/miniconda3/etc/profile.d/conda.sh
conda activate higherlev_retro
set -eo pipefail

cd /home/gzbrown/minicasp
export PYTHONPATH=/home/gzbrown/minicasp:$PYTHONPATH

python - <<'PY'
import os
from datetime import datetime

from minicasp.utils import (
    build_and_train_from_csv,
    load_reaction_records_csv,
    load_askcos_buyables_jsonl_gz,
    sample_targets_from_products,
    audit_targets,
    save_json,
    SearchConfig,
)

CSV = "/home/gzbrown/minicasp/data/reactions/uspto_original.csv"

TRAIN_LIMIT = 50000
TARGETS_N = 100

art = build_and_train_from_csv(
    csv_path=CSV,
    rxn_col="rxn_smiles",
    limit=TRAIN_LIMIT,
    shuffle=True,
    seed=0,
    template_radius=1,
    template_min_count=1,
)

recs = load_reaction_records_csv(
    csv_path=CSV,
    rxn_col="rxn_smiles",
    limit=TRAIN_LIMIT,
    shuffle=True,
    seed=0,
)

buyables = load_askcos_buyables_jsonl_gz(
    "/home/gzbrown/higherlev_retro/ASKCOSv2/askcos2_core/buyables/buyables.jsonl.gz"
)

targets = sample_targets_from_products(recs, n=TARGETS_N, seed=0)

cfg = SearchConfig(
    max_depth=6,
    max_expansions=300,
    topk_templates=25,
    max_outcomes_per_template=25,
    time_limit_s=10.0,
)

report = audit_targets(targets, art.model, art.templates, buyables, config=cfg)

ts = datetime.now().strftime("%y%m%d-%H%M%S")
outdir = "/home/gzbrown/minicasp/results"
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, f"audit_{ts}.json")
save_json(outpath, report)

print("Saved:", outpath)
print("Success:", report["n_solved"], "/", report["n_targets"], "=", report["success_rate"])
PY
