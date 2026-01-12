
# =========================
# FILE: minicasp/scripts/step2_audit.py
# =========================
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Set

from minicasp.utils import (
    SearchConfig,
    TemplateModel,
    TemplateRecord,
    audit_targets,
    load_askcos_buyables_jsonl_gz,
    load_reaction_records_csv,
    sample_targets_from_products,
    save_json,
    setup_logger,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_artifacts(artifacts_dir: Path) -> tuple[TemplateModel, list[TemplateRecord], Dict[str, Any]]:
    # Templates
    templates_json = artifacts_dir / "templates.json"
    data = json.loads(templates_json.read_text())
    templates = [TemplateRecord(**t) for t in data["templates"]]

    # Model
    import joblib  # type: ignore

    model_blob = joblib.load(artifacts_dir / "template_model.joblib")
    model = TemplateModel(
        clf=model_blob["clf"],
        label_encoder=model_blob["label_encoder"],
        n_bits=int(model_blob["n_bits"]),
        fp_radius=int(model_blob["fp_radius"]),
    )

    meta = json.loads((artifacts_dir / "meta.json").read_text())
    return model, templates, meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with USPTO reactions")
    p.add_argument("--rxn_col", default="rxn_smiles")
    p.add_argument("--train_limit", type=int, default=50_000, help="Must match (or be <=) step1 limit")
    p.add_argument("--targets_n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_id", required=True, help="Run id produced by step1")
    p.add_argument(
        "--buyables",
        default="/home/gzbrown/higherlev_retro/ASKCOSv2/askcos2_core/buyables/buyables.jsonl.gz",
        help="ASKCOS buyables jsonl.gz",
    )

    # Search knobs
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--max_expansions", type=int, default=300)
    p.add_argument("--topk_templates", type=int, default=25)
    p.add_argument("--max_outcomes_per_template", type=int, default=25)
    p.add_argument("--time_limit_s", type=float, default=10.0)

    args = p.parse_args()
    setup_logger()

    repo_root = _repo_root()
    artifacts_dir = repo_root / "results" / "artifacts" / args.run_id
    if not artifacts_dir.exists():
        raise SystemExit(f"Artifacts dir not found: {artifacts_dir} (did step1 run?)")

    model, templates, meta = _load_artifacts(artifacts_dir)

    recs = load_reaction_records_csv(
        csv_path=args.csv,
        rxn_col=args.rxn_col,
        limit=args.train_limit,
        shuffle=True,
        seed=args.seed,
    )

    buyables: Set[str] = load_askcos_buyables_jsonl_gz(args.buyables)

    targets = sample_targets_from_products(recs, n=args.targets_n, seed=args.seed)

    cfg = SearchConfig(
        max_depth=args.max_depth,
        max_expansions=args.max_expansions,
        topk_templates=args.topk_templates,
        max_outcomes_per_template=args.max_outcomes_per_template,
        time_limit_s=args.time_limit_s,
    )

    report = audit_targets(targets, model, templates, buyables, config=cfg)

    outdir = repo_root / "results" / "audits" / args.run_id
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "audit.json"
    save_json(str(outpath), report)

    print("Saved:", outpath)
    print("Success:", report["n_solved"], "/", report["n_targets"], "=", report["success_rate"])


if __name__ == "__main__":
    main()
