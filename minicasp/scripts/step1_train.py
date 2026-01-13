import argparse
import os
import json
import shutil

from minicasp.utils import (
    load_pairs_jsonl_gz,
    save_pairs_jsonl_gz,
    ordered_group_split,
    random_group_split,
    train_template_model,
    save_model_joblib,
)

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pairs_cache", required=True)
    ap.add_argument("--templates_cache", required=True)
    ap.add_argument("--run_dir", required=True)

    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--split_mode", type=str, default="ordered_group",
                    choices=["ordered_group", "random_group"])
    ap.add_argument("--split_seed", type=int, default=0)

    # model seed (SGD randomness)
    ap.add_argument("--model_seed", type=int, default=0)

    # features / baseline hyperparams
    ap.add_argument("--fp_radius", type=int, default=2)
    ap.add_argument("--n_bits", type=int, default=2048)
    ap.add_argument("--max_iter", type=int, default=25)

    # pluggable model
    ap.add_argument("--model_type", default="sgd",
                    choices=["sgd", "mlp", "mlp_torch", "mlp_sklearn"])
    ap.add_argument("--mlp_hidden", default="1024,1024",
                    help="Comma-separated hidden sizes, e.g. 1024,1024")
    ap.add_argument("--mlp_epochs", type=int, default=8)
    ap.add_argument("--mlp_batch_size", type=int, default=256)
    ap.add_argument("--mlp_lr", type=float, default=1e-3)
    ap.add_argument("--mlp_dropout", type=float, default=0.1)

    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    # Parse hidden layer sizes EARLY so it's always defined
    hidden = tuple(int(x) for x in args.mlp_hidden.split(",") if x.strip())
    if args.model_type != "sgd" and len(hidden) == 0:
        raise ValueError("--mlp_hidden must have at least one layer size for MLP models.")

    # Copy templates cache into the run directory so step2 only needs run_dir
    templates_out = os.path.join(args.run_dir, "templates.json.gz")
    shutil.copyfile(args.templates_cache, templates_out)

    pairs = load_pairs_jsonl_gz(args.pairs_cache)
    X = [p["product"] for p in pairs]
    y = [int(p["template_id"]) for p in pairs]
    groups = X  # group-by-product to prevent leakage

    if args.split_mode == "ordered_group":
        (Xtr, ytr), (Xte, yte), meta = ordered_group_split(
            X, y, groups, test_size=args.test_size
        )
    else:
        (Xtr, ytr), (Xte, yte), meta = random_group_split(
            X, y, groups, test_size=args.test_size, seed=args.split_seed
        )

    meta.update({
        "split_mode": args.split_mode,
        "test_size": args.test_size,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "fp_radius": args.fp_radius,
        "n_bits": args.n_bits,
        "max_iter": args.max_iter,
        "model_type": args.model_type,
        "mlp_hidden": list(hidden),
        "mlp_epochs": args.mlp_epochs,
        "mlp_batch_size": args.mlp_batch_size,
        "mlp_lr": args.mlp_lr,
        "mlp_dropout": args.mlp_dropout,
        "n_pairs_total": len(X),
        "n_pairs_train": len(Xtr),
        "n_pairs_test": len(Xte),
    })

    model = train_template_model(
        Xtr, ytr,
        fp_radius=args.fp_radius,
        n_bits=args.n_bits,
        random_state=args.model_seed,
        max_iter=args.max_iter,          # used by SGD
        model_type=args.model_type,
        mlp_hidden=hidden,
        mlp_epochs=args.mlp_epochs,
        mlp_batch_size=args.mlp_batch_size,
        mlp_lr=args.mlp_lr,
        mlp_dropout=args.mlp_dropout,
    )

    model_path = os.path.join(args.run_dir, "model.joblib")
    save_model_joblib(model_path, model)

    # Save train/test pairs for step2
    train_pairs = [{"product": p, "template_id": int(t)} for p, t in zip(Xtr, ytr)]
    test_pairs  = [{"product": p, "template_id": int(t)} for p, t in zip(Xte, yte)]
    save_pairs_jsonl_gz(os.path.join(args.run_dir, "train_pairs.jsonl.gz"), train_pairs)
    save_pairs_jsonl_gz(os.path.join(args.run_dir, "test_pairs.jsonl.gz"), test_pairs)

    # Save test targets (unique products)
    with open(os.path.join(args.run_dir, "test_targets.txt"), "w") as f:
        for smi in sorted(set(Xte)):
            f.write(smi + "\n")

    with open(os.path.join(args.run_dir, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved run_dir:", args.run_dir)
    print("Model:", model_path)
    print("Templates:", templates_out)
    print("Split mode:", args.split_mode)
    print("Model type:", args.model_type)
    print("Train pairs:", len(Xtr), "Test pairs:", len(Xte))

if __name__ == "__main__":
    main()
