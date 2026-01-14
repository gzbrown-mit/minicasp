from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import logging
import numpy as np
import os, gzip, json
from .chem import normalize_mol_set
from .data import ReactionRecord
from .templates import extract_retro_template_smarts

def make_training_pairs(
    reactions: Sequence[ReactionRecord],
    smarts_to_id: Dict[str, int],
    radius: int = 1,
) -> Tuple[List[str], List[int]]:
    x_smiles: List[str] = []
    y_ids: List[int] = []
    logger = logging.getLogger(__name__)
    missing_template = 0
    missing_smarts = 0
    empty_product = 0

    for rec in reactions:
        smarts = extract_retro_template_smarts(rec.rxn_smiles, radius=radius)
        if not smarts:
            missing_smarts += 1
            continue
        tid = smarts_to_id.get(smarts)
        if tid is None:
            missing_template += 1
            continue

        prod_parts = [p for p in rec.products.split(".") if p.strip()]
        prod_norm = ".".join(normalize_mol_set(prod_parts))
        if not prod_norm:
            empty_product += 1
            continue

        x_smiles.append(prod_norm)
        y_ids.append(int(tid))

    logger.info(
        "Training pairs summary: total=%d kept=%d missing_smarts=%d missing_template=%d empty_product=%d",
        len(reactions),
        len(x_smiles),
        missing_smarts,
        missing_template,
        empty_product,
    )
    return x_smiles, y_ids

def save_training_pairs_npz(path: str, product_smiles: Sequence[str], template_ids: Sequence[int]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        product_smiles=np.array(list(product_smiles), dtype=object),
        template_ids=np.array(list(template_ids), dtype=np.int64),
    )

def load_training_pairs_npz(path: str) -> Tuple[List[str], List[int]]:
    z = np.load(path, allow_pickle=True)
    return z["product_smiles"].tolist(), z["template_ids"].astype(int).tolist()

def save_pairs_jsonl_gz(path: str, pairs: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for obj in pairs:
            f.write(json.dumps(obj) + "\n")

def load_pairs_jsonl_gz(path: str) -> List[dict]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out