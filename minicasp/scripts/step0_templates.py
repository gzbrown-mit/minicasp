import argparse
import os

from minicasp.utils import (
    load_reaction_records_csv,
    build_template_library,
    make_training_pairs,
    save_templates_cache,
    save_pairs_jsonl_gz,
)

def main():
    ap = argparse.ArgumentParser()

    # inputs
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rxn_col", default="rxn_smiles")
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true", help="Shuffle reactions before limiting")

    ap.add_argument("--template_radius", type=int, default=1)
    ap.add_argument("--template_min_count", type=int, default=5)

    # outputs
    ap.add_argument("--templates_out", required=True,
                    help="e.g. results/templates/templates_r1_min5.json.gz")
    ap.add_argument("--pairs_out", required=True,
                    help="e.g. results/templates/pairs_r1_min5.jsonl.gz")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.templates_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.pairs_out), exist_ok=True)

    recs = load_reaction_records_csv(
        args.csv,
        rxn_col=args.rxn_col,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    templates, smarts_to_id = build_template_library(
        recs,
        radius=args.template_radius,
        min_count=args.template_min_count,
    )
    save_templates_cache(args.templates_out, templates)
    print("Saved templates:", args.templates_out, "n_templates=", len(templates))

    prod_smiles, y_ids = make_training_pairs(
        recs,
        smarts_to_id,
        radius=args.template_radius,
    )
    pairs = [{"product": p, "template_id": int(t)} for p, t in zip(prod_smiles, y_ids)]
    save_pairs_jsonl_gz(args.pairs_out, pairs)
    print("Saved pairs:", args.pairs_out, "n_pairs=", len(pairs))

if __name__ == "__main__":
    main()
