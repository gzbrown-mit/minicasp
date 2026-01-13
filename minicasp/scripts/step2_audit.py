import argparse
import os
from datetime import datetime
import random

from minicasp.utils import (
    load_templates_cache,
    load_pairs_jsonl_gz,
    load_model_joblib,
    load_buyables_cached,
    audit_targets,
    save_json,
    SearchConfig,
)

def _infer_model_type(model):
    # best-effort: support either an attribute or dict payloads
    if hasattr(model, "model_type"):
        return getattr(model, "model_type")
    if isinstance(model, dict) and "model_type" in model:
        return model["model_type"]
    return "unknown"

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--run_dir", required=True,
                    help="e.g. /home/gzbrown/minicasp/results/runs/YYMMDD-HHMMSS")
    ap.add_argument("--targets_n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)

    # Optional: sanity-check model type
    ap.add_argument("--expect_model_type", default="",
                    choices=["", "sgd", "mlp", "mlp_torch", "mlp_sklearn"])

    # Search params
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--max_expansions", type=int, default=300)
    ap.add_argument("--topk_templates", type=int, default=25)
    ap.add_argument("--max_outcomes_per_template", type=int, default=25)
    ap.add_argument("--time_limit_s", type=float, default=10.0)

    # Buyables (use your copied files in minicasp)
    ap.add_argument("--buyables_jsonl_gz", default="/home/gzbrown/minicasp/data/buyables/buyables.jsonl.gz")
    ap.add_argument("--buyables_cache_txt_gz", default="/home/gzbrown/minicasp/data/buyables/buyables_smiles.txt.gz")

    args = ap.parse_args()
    run_dir = args.run_dir

    model_path = os.path.join(run_dir, "model.joblib")
    templates_path = os.path.join(run_dir, "templates.json.gz")
    test_pairs_path = os.path.join(run_dir, "test_pairs.jsonl.gz")

    for p in (model_path, templates_path, test_pairs_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    print("Loading model:", model_path)
    model = load_model_joblib(model_path)

    loaded_type = _infer_model_type(model)
    print("Loaded model_type:", loaded_type)

    if args.expect_model_type and loaded_type != "unknown" and loaded_type != args.expect_model_type:
        raise RuntimeError(f"Model type mismatch: expected {args.expect_model_type}, got {loaded_type}")

    print("Loading templates:", templates_path)
    templates = load_templates_cache(templates_path)

    print("Loading test pairs:", test_pairs_path)
    test_pairs = load_pairs_jsonl_gz(test_pairs_path)

    products = sorted({p["product"] for p in test_pairs if p.get("product")})
    if not products:
        raise RuntimeError("No test products found in test_pairs.")

    rng = random.Random(args.seed)
    rng.shuffle(products)
    targets = products[: min(args.targets_n, len(products))]

    print("Loading buyables (cached if present)...")
    buyables = load_buyables_cached(args.buyables_jsonl_gz, args.buyables_cache_txt_gz)
    print("Buyables loaded:", len(buyables))

    cfg = SearchConfig(
        max_depth=args.max_depth,
        max_expansions=args.max_expansions,
        topk_templates=args.topk_templates,
        max_outcomes_per_template=args.max_outcomes_per_template,
        time_limit_s=args.time_limit_s,
    )

    report = audit_targets(
        targets=targets,
        model=model,
        templates=templates,
        buyables=buyables,
        config=cfg,
    )

    # attach meta
    report = {
        "meta": {
            "run_dir": run_dir,
            "model_path": model_path,
            "model_type": loaded_type,
            "expect_model_type": args.expect_model_type,
            "templates_path": templates_path,
            "test_pairs_path": test_pairs_path,
            "targets_n": args.targets_n,
            "seed": args.seed,
            "search": {
                "max_depth": args.max_depth,
                "max_expansions": args.max_expansions,
                "topk_templates": args.topk_templates,
                "max_outcomes_per_template": args.max_outcomes_per_template,
                "time_limit_s": args.time_limit_s,
            },
            "buyables": {
                "jsonl_gz": args.buyables_jsonl_gz,
                "cache_txt_gz": args.buyables_cache_txt_gz,
                "n_buyables": len(buyables),
            },
        },
        **report,
    }

    out_dir = os.path.join(run_dir, "audit")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%y%m%d-%H%M%S")
    out_path = os.path.join(out_dir, f"audit_{ts}.json")
    save_json(out_path, report)

    print("Saved audit:", out_path)
    print("Success:", report["n_solved"], "/", report["n_targets"], "=", report["success_rate"])

if __name__ == "__main__":
    main()
