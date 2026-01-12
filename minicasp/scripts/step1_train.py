# =========================
# FILE: minicasp/scripts/step1_train.py
# =========================
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from minicasp.utils import (
    MiniCaspArtifacts,
    build_and_train_from_csv,
    save_json,
    setup_logger,
)


def _repo_root() -> Path:
    # .../minicasp/scripts/step1_train.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _artifacts_dir(repo_root: Path, run_id: str) -> Path:
    return repo_root / "results" / "artifacts" / run_id


def _save_artifacts(outdir: Path, artifacts: MiniCaspArtifacts) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Templates
    templates_path = outdir / "templates.json"
    save_json(
        str(templates_path),
        {
            "templates": [asdict(t) for t in artifacts.templates],
            "reactions_used": artifacts.reactions_used,
        },
    )

    # Model (pickle via joblib)
    import joblib  # type: ignore

    model_path = outdir / "template_model.joblib"
    joblib.dump(
        {
            "clf": artifacts.model.clf,
            "label_encoder": artifacts.model.label_encoder,
            "n_bits": artifacts.model.n_bits,
            "fp_radius": artifacts.model.fp_radius,
        },
        model_path,
    )

    # Metadata for step2 convenience
    meta = {
        "templates_path": str(templates_path),
        "model_path": str(model_path),
        "reactions_used": artifacts.reactions_used,
    }
    save_json(str(outdir / "meta.json"), meta)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with USPTO reactions")
    p.add_argument("--rxn_col", default="rxn_smiles")
    p.add_argument("--train_limit", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--template_radius", type=int, default=1)
    p.add_argument("--template_min_count", type=int, default=5)
    p.add_argument(
        "--run_id",
        default="",
        help="Optional run id; default is timestamp-based",
    )
    args = p.parse_args()

    setup_logger()

    repo_root = _repo_root()
    run_id = args.run_id or datetime.now().strftime("%y%m%d-%H%M%S")
    outdir = _artifacts_dir(repo_root, run_id)

    print("REPO_ROOT:", repo_root)
    print("RUN_ID:", run_id)
    print("ARTIFACTS_DIR:", outdir)

    artifacts = build_and_train_from_csv(
        csv_path=args.csv,
        rxn_col=args.rxn_col,
        limit=args.train_limit,
        shuffle=True,
        seed=args.seed,
        template_radius=args.template_radius,
        template_min_count=args.template_min_count,
    )

    _save_artifacts(outdir, artifacts)
    print("Saved artifacts to:", outdir)


if __name__ == "__main__":
    main()


