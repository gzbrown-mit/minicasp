#!/usr/bin/env python
"""
Train or retrain an expansion policy model for MCTS search.

The model predicts which reaction template to apply given a product molecule's
Morgan fingerprint. Output is an ONNX model compatible with minicasp's
LocalOnnxModel loader.

Usage:
    # Train from a reaction SMILES CSV (extract templates + train)
    python scripts/train_policy.py \
        --reaction_csv minicasp/data/reactions/uspto_original.csv \
        --output_dir results/training/my_run

    # Train from existing templates + labeled pairs
    python scripts/train_policy.py \
        --templates_csv results/templates.csv.gz \
        --pairs_jsonl results/pairs.jsonl.gz \
        --output_dir results/training/my_run

    # Retrain with more data or different hyperparameters
    python scripts/train_policy.py \
        --reaction_csv minicasp/data/reactions/uspto_original.csv \
        --output_dir results/training/my_run \
        --fp_radius 2 --fp_nbits 2048 --epochs 10 --hidden 512,512
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def smiles_to_fingerprint(
    smiles: str, radius: int = 2, nbits: int = 2048
) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def extract_templates_from_csv(
    csv_path: str,
    rxn_col: str = "rxn_smiles",
    limit: int = 0,
    min_count: int = 5,
    radius: int = 1,
) -> tuple[pd.DataFrame, list[dict]]:
    """Extract templates from a reaction CSV and return (templates_df, pairs)."""
    from rdchiral.template_extractor import extract_from_reaction

    print(f"Loading reactions from {csv_path}...")
    df = pd.read_csv(csv_path)
    if rxn_col not in df.columns:
        for alt in ("reaction_smiles", "reaction", "mapped_rxn"):
            if alt in df.columns:
                rxn_col = alt
                break
        else:
            raise ValueError(f"Column '{rxn_col}' not found. Available: {list(df.columns)[:20]}")

    rxns = df[rxn_col].dropna().astype(str).tolist()
    if limit > 0:
        rxns = rxns[:limit]
    print(f"  {len(rxns)} reactions loaded")

    # Extract templates
    from collections import Counter

    template_counter: Counter[str] = Counter()
    rxn_templates: list[tuple[str, str]] = []  # (product_smiles, template_smarts)

    print("Extracting templates...")
    for i, rxn in enumerate(rxns):
        if ">>" in rxn:
            parts = rxn.split(">>")
            reactants, products = parts[0].strip(), parts[-1].strip()
        else:
            seg = rxn.split(">")
            if len(seg) == 3:
                reactants, products = seg[0].strip(), seg[2].strip()
            else:
                continue

        rxn_dict = {
            "_id": str(i),
            "reactants": reactants,
            "products": products,
        }
        try:
            result = extract_from_reaction(rxn_dict)
        except Exception:
            continue

        smarts = None
        if isinstance(result, dict):
            smarts = result.get("reaction_smarts") or result.get("retro_smarts")
        elif isinstance(result, str):
            smarts = result

        if not smarts or not smarts.strip():
            continue

        smarts = smarts.strip()
        template_counter[smarts] += 1

        # Canonicalize product
        prod_mol = Chem.MolFromSmiles(products)
        if prod_mol:
            canon_prod = Chem.MolToSmiles(prod_mol, canonical=True)
            rxn_templates.append((canon_prod, smarts))

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(rxns)} reactions...")

    # Filter by min count
    valid_smarts = {s for s, c in template_counter.items() if c >= min_count}
    smarts_list = sorted(valid_smarts, key=lambda s: template_counter[s], reverse=True)
    smarts_to_idx = {s: i for i, s in enumerate(smarts_list)}
    print(f"  {len(smarts_list)} templates with count >= {min_count}")

    # Build templates DataFrame (same format as AiZynthFinder)
    templates_df = pd.DataFrame(
        {"retro_template": smarts_list, "count": [template_counter[s] for s in smarts_list]}
    )

    # Build pairs
    pairs = []
    for prod, smarts in rxn_templates:
        idx = smarts_to_idx.get(smarts)
        if idx is not None:
            pairs.append({"product": prod, "template_idx": idx})

    print(f"  {len(pairs)} training pairs")
    return templates_df, pairs


def load_existing_data(
    templates_path: str, pairs_path: str
) -> tuple[pd.DataFrame, list[dict]]:
    """Load pre-extracted templates and pairs."""
    print(f"Loading templates from {templates_path}...")
    if templates_path.endswith(".csv.gz") or templates_path.endswith(".csv"):
        templates_df = pd.read_csv(templates_path, index_col=0, sep="\t")
    else:
        templates_df = pd.read_hdf(templates_path, "table")
    print(f"  {len(templates_df)} templates")

    print(f"Loading pairs from {pairs_path}...")
    pairs = []
    opener = gzip.open if pairs_path.endswith(".gz") else open
    with opener(pairs_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"  {len(pairs)} pairs")
    return templates_df, pairs


def train_model(
    pairs: list[dict],
    n_templates: int,
    fp_radius: int = 2,
    fp_nbits: int = 2048,
    hidden_sizes: tuple[int, ...] = (512,),
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    dropout: float = 0.2,
    test_fraction: float = 0.1,
) -> tuple:
    """Train a policy model and return (model, metrics)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print(f"\nFeaturizing {len(pairs)} products (radius={fp_radius}, nbits={fp_nbits})...")
    X_list = []
    y_list = []
    skipped = 0
    for p in pairs:
        fp = smiles_to_fingerprint(p["product"], radius=fp_radius, nbits=fp_nbits)
        if fp is None:
            skipped += 1
            continue
        X_list.append(fp)
        y_list.append(p["template_idx"])

    if skipped:
        print(f"  Skipped {skipped} invalid SMILES")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Train/test split
    n_test = max(1, int(len(X) * test_fraction))
    indices = np.random.permutation(len(X))
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Build model
    layers: list[nn.Module] = []
    in_dim = fp_nbits
    for h in hidden_sizes:
        layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
        in_dim = h
    layers.append(nn.Linear(in_dim, n_templates))
    model = nn.Sequential(*layers)

    # Training
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nTraining ({epochs} epochs, hidden={hidden_sizes}, lr={lr})...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test))
        preds = logits.argmax(dim=1).numpy()
        top1_acc = (preds == y_test).mean()

        # Top-k accuracy
        topk = min(10, n_templates)
        topk_preds = logits.topk(topk, dim=1).indices.numpy()
        topk_acc = np.mean([y_test[i] in topk_preds[i] for i in range(len(y_test))])

    metrics = {
        "top1_accuracy": float(top1_acc),
        f"top{topk}_accuracy": float(topk_acc),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_templates": n_templates,
        "epochs": epochs,
        "hidden_sizes": list(hidden_sizes),
        "fp_radius": fp_radius,
        "fp_nbits": fp_nbits,
    }
    print(f"\n  Top-1 accuracy: {top1_acc:.3f}")
    print(f"  Top-{topk} accuracy: {topk_acc:.3f}")

    return model, metrics


def export_onnx(model, fp_nbits: int, output_path: str) -> None:
    """Export PyTorch model to ONNX format."""
    import torch

    model.eval()
    dummy_input = torch.randn(1, fp_nbits)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["fingerprint"],
        output_names=["output"],
        dynamic_axes={"fingerprint": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11,
    )
    print(f"  ONNX model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train expansion policy model")

    # Data source (choose one)
    parser.add_argument("--reaction_csv", default=None, help="Path to reaction SMILES CSV")
    parser.add_argument("--rxn_col", default="rxn_smiles", help="Column name for reaction SMILES")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of reactions (0=all)")
    parser.add_argument("--min_count", type=int, default=5, help="Min template occurrences")
    parser.add_argument("--templates_csv", default=None, help="Pre-extracted templates file")
    parser.add_argument("--pairs_jsonl", default=None, help="Pre-extracted pairs file")

    # Model hyperparameters
    parser.add_argument("--fp_radius", type=int, default=2)
    parser.add_argument("--fp_nbits", type=int, default=2048)
    parser.add_argument("--hidden", default="512", help="Comma-separated hidden layer sizes")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_dir", required=True, help="Directory for outputs")
    parser.add_argument("--config", default=None, help="Ignored (for slurm compat)")

    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.reaction_csv:
        templates_df, pairs = extract_templates_from_csv(
            args.reaction_csv,
            rxn_col=args.rxn_col,
            limit=args.limit,
            min_count=args.min_count,
        )
    elif args.templates_csv and args.pairs_jsonl:
        templates_df, pairs = load_existing_data(args.templates_csv, args.pairs_jsonl)
    else:
        parser.error("Provide either --reaction_csv or both --templates_csv and --pairs_jsonl")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save templates in AiZynthFinder format (tab-separated CSV.gz)
    templates_path = os.path.join(args.output_dir, "templates.csv.gz")
    templates_df.to_csv(templates_path, sep="\t", compression="gzip")
    print(f"Templates saved to {templates_path}")

    # Save pairs
    pairs_path = os.path.join(args.output_dir, "pairs.jsonl.gz")
    with gzip.open(pairs_path, "wt", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Pairs saved to {pairs_path}")

    hidden_sizes = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    model, metrics = train_model(
        pairs=pairs,
        n_templates=len(templates_df),
        fp_radius=args.fp_radius,
        fp_nbits=args.fp_nbits,
        hidden_sizes=hidden_sizes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
    )

    # Export ONNX
    model_path = os.path.join(args.output_dir, "policy_model.onnx")
    export_onnx(model, args.fp_nbits, model_path)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print(f"\nTo use this model:")
    print(f"  export MINICASP_MODEL={os.path.abspath(model_path)}")
    print(f"  export MINICASP_TEMPLATES={os.path.abspath(templates_path)}")


if __name__ == "__main__":
    main()
