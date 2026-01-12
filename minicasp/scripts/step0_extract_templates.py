import argparse, os
from minicasp.utils import load_reaction_records_csv, build_template_library, save_templates_cache

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rxn_col", default="rxn_smiles")
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--template_radius", type=int, default=1)
    ap.add_argument("--template_min_count", type=int, default=5)
    ap.add_argument("--out", required=True)  # e.g. results/templates/templates_r1_min5.json.gz
    args = ap.parse_args()

    recs = load_reaction_records_csv(args.csv, rxn_col=args.rxn_col, limit=args.limit, shuffle=True, seed=args.seed)
    templates, _ = build_template_library(recs, radius=args.template_radius, min_count=args.template_min_count)
    save_templates_cache(args.out, templates)
    print("Saved templates:", args.out, "n_templates=", len(templates))

if __name__ == "__main__":
    main()
