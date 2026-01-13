from __future__ import annotations
from typing import Any, Dict, Sequence, Set
import gzip, json, os
from .chem import canonicalize_smiles
from .data import ReactionRecord

def load_buyables_txt(path: str) -> Set[str]:
    buyables: Set[str] = set()
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                buyables.add(canonicalize_smiles(s.split(",")[0].strip()))
    return buyables

def default_buyables_from_reactants(reactions: Sequence[ReactionRecord], max_unique: int = 200_000) -> Set[str]:
    buyables: Set[str] = set()
    for rec in reactions:
        parts = [p for p in rec.reactants.split(".") if p.strip()]
        for p in parts:
            buyables.add(canonicalize_smiles(p))
            if len(buyables) >= max_unique:
                return buyables
    return buyables

def load_askcos_buyables_jsonl_gz(
    path: str,
    smiles_key_candidates: Sequence[str] = ("smiles", "smile", "canonical_smiles", "isomeric_smiles"),
    limit: int = 0,
) -> Set[str]:
    buyables: Set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            smi = None
            for k in smiles_key_candidates:
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    smi = v.strip()
                    break
            if not smi:
                continue

            buyables.add(canonicalize_smiles(smi))
            if limit and len(buyables) >= limit:
                break
    return buyables

def save_smiles_txt_gz(path: str, smiles_set: Set[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for s in sorted(smiles_set):
            f.write(s + "\n")

def load_buyables_cached(jsonl_gz_path: str, cache_txt_gz_path: str) -> Set[str]:
    if os.path.exists(cache_txt_gz_path):
        buyables = set()
        with gzip.open(cache_txt_gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    buyables.add(s)
        return buyables

    buyables = load_askcos_buyables_jsonl_gz(jsonl_gz_path)
    save_smiles_txt_gz(cache_txt_gz_path, buyables)
    return buyables
