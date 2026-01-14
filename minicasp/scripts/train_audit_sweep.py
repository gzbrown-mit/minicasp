import argparse
import json
import os
import random
from datetime import datetime

from minicasp.utils import (
    SearchConfig,
    audit_targets,
    load_buyables_cached,
    load_pairs_jsonl_gz,
    load_templates_cache,
    ordered_group_split,
    random_group_split,
    save_model_joblib,
    save_pairs_jsonl_gz,
    save_templates_cache,
    setup_logger,
    strip_atom_maps,
    train_template_model,
)


def _parse_hidden(hidden_arg: str) -> tuple[int, ...]:
    return tuple(int(x) for x in hidden_arg.split(",") if x.strip())


def _select_targets(test_pairs: list[dict], n_targets: int, seed: int | None) -> list[str]:
    products = sorted({strip_atom_maps(p["product"]) for p in test_pairs if p.get("product")})
    if not products:
        raise RuntimeError("No test products found in test_pairs.")
    seed_value = seed
    if seed_value is None:
        seed_value = random.SystemRandom().randint(0, 2**32 - 1)
    rng = random.Random(seed_value)
    rng.shuffle(products)
    return products[: min(n_targets, len(products))]


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--pairs_cache", required=True)
    ap.add_argument("--templates_cache", required=True)
    ap.add_argument("--run_dir", required=True,
                    help="Directory to store model and audit sweep outputs.")

    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--split_mode", type=str, default="ordered_group",
                    choices=["ordered_group", "random_group"])
    ap.add_argument("--split_seed", type=int, default=0)

    ap.add_argument("--model_seed", type=int, default=0)
    ap.add_argument("--fp_radius", type=int, default=2)
    ap.add_argument("--n_bits", type=int, default=2048)
    ap.add_argument("--max_iter", type=int, default=25)

    ap.add_argument("--model_type", default="sgd",
                    choices=["sgd", "mlp", "mlp_torch", "mlp_sklearn"])
    ap.add_argument("--mlp_hidden", default="1024,1024")
    ap.add_argument("--mlp_epochs", type=int, default=8)
    ap.add_argument("--mlp_batch_size", type=int, default=256)
    ap.add_argument("--mlp_lr", type=float, default=1e-3)
    ap.add_argument("--mlp_dropout", type=float, default=0.1)

    ap.add_argument("--targets_n", type=int, default=100)
    ap.add_argument("--targets_seed", type=int, default=None,
                    help="Seed for target selection. Omit for non-deterministic runs.")

    ap.add_argument("--buyables_jsonl_gz", default="/home/gzbrown/minicasp/data/buyables/buyables.jsonl.gz")
    ap.add_argument("--buyables_cache_txt_gz", default="/home/gzbrown/minicasp/data/buyables/buyables_smiles.txt.gz")

    ap.add_argument("--topk_templates", default="10,25,50")
    ap.add_argument("--max_outcomes_per_template", default="10,25,50")
    ap.add_argument("--max_depth", default="4,6,8")
    ap.add_argument("--max_expansions", default="100,300,600")
    ap.add_argument("--time_limit_s", default="5,10,20")

    args = ap.parse_args()
    setup_logger()

    os.makedirs(args.run_dir, exist_ok=True)
    hidden = _parse_hidden(args.mlp_hidden)
    if args.model_type != "sgd" and len(hidden) == 0:
        raise ValueError("--mlp_hidden must have at least one layer size for MLP models.")

    templates = load_templates_cache(args.templates_cache)
    templates_out = os.path.join(args.run_dir, "templates.json.gz")
    save_templates_cache(templates_out, templates)

    pairs = load_pairs_jsonl_gz(args.pairs_cache)
    X = [p["product"] for p in pairs]
    y = [int(p["template_id"]) for p in pairs]
    groups = X

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
        Xtr,
        ytr,
        fp_radius=args.fp_radius,
        n_bits=args.n_bits,
        random_state=args.model_seed,
        max_iter=args.max_iter,
        model_type=args.model_type,
        mlp_hidden=hidden,
        mlp_epochs=args.mlp_epochs,
        mlp_batch_size=args.mlp_batch_size,
        mlp_lr=args.mlp_lr,
        mlp_dropout=args.mlp_dropout,
    )

    model_path = os.path.join(args.run_dir, "model.joblib")
    save_model_joblib(model_path, model)

    train_pairs = [{"product": p, "template_id": int(t)} for p, t in zip(Xtr, ytr)]
    test_pairs = [{"product": p, "template_id": int(t)} for p, t in zip(Xte, yte)]
    save_pairs_jsonl_gz(os.path.join(args.run_dir, "train_pairs.jsonl.gz"), train_pairs)
    save_pairs_jsonl_gz(os.path.join(args.run_dir, "test_pairs.jsonl.gz"), test_pairs)

    with open(os.path.join(args.run_dir, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    targets = _select_targets(test_pairs, args.targets_n, args.targets_seed)
    buyables = load_buyables_cached(args.buyables_jsonl_gz, args.buyables_cache_txt_gz)

    sweep = {
        "topk_templates": [int(x) for x in args.topk_templates.split(",") if x.strip()],
        "max_outcomes_per_template": [int(x) for x in args.max_outcomes_per_template.split(",") if x.strip()],
        "max_depth": [int(x) for x in args.max_depth.split(",") if x.strip()],
        "max_expansions": [int(x) for x in args.max_expansions.split(",") if x.strip()],
        "time_limit_s": [float(x) for x in args.time_limit_s.split(",") if x.strip()],
    }

    if any(len(v) != 3 for v in sweep.values()):
        raise ValueError("Each sweep variable must provide exactly three values: min, mid, max.")

    baseline = SearchConfig(
        max_depth=sweep["max_depth"][1],
        max_expansions=sweep["max_expansions"][1],
        topk_templates=sweep["topk_templates"][1],
        max_outcomes_per_template=sweep["max_outcomes_per_template"][1],
        time_limit_s=sweep["time_limit_s"][1],
    )

    results = []
    for var_name, values in sweep.items():
        for value in values:
            config = SearchConfig(
                max_depth=baseline.max_depth,
                max_expansions=baseline.max_expansions,
                topk_templates=baseline.topk_templates,
                max_outcomes_per_template=baseline.max_outcomes_per_template,
                time_limit_s=baseline.time_limit_s,
            )

            setattr(config, var_name, value)
            report = audit_targets(
                targets=targets,
                model=model,
                templates=templates,
                buyables=buyables,
                config=config,
            )

            results.append({
                "variable": var_name,
                "value": value,
                "success_rate": report["success_rate"],
                "n_solved": report["n_solved"],
                "n_targets": report["n_targets"],
                "config": {
                    "max_depth": config.max_depth,
                    "max_expansions": config.max_expansions,
                    "topk_templates": config.topk_templates,
                    "max_outcomes_per_template": config.max_outcomes_per_template,
                    "time_limit_s": config.time_limit_s,
                },
            })

    out_dir = os.path.join(args.run_dir, "audit")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%y%m%d-%H%M%S")
    out_path = os.path.join(out_dir, f"audit_sweep_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({
            "meta": {
                "run_dir": args.run_dir,
                "model_path": model_path,
                "templates_path": templates_out,
                "targets_n": len(targets),
                "targets_seed": args.targets_seed,
                "buyables_jsonl_gz": args.buyables_jsonl_gz,
                "buyables_cache_txt_gz": args.buyables_cache_txt_gz,
                "sweep": sweep,
            },
            "baseline": {
                "max_depth": baseline.max_depth,
                "max_expansions": baseline.max_expansions,
                "topk_templates": baseline.topk_templates,
                "max_outcomes_per_template": baseline.max_outcomes_per_template,
                "time_limit_s": baseline.time_limit_s,
            },
            "results": results,
        }, f, indent=2)

    print("Saved audit sweep:", out_path)


if __name__ == "__main__":
    main()