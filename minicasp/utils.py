# minicasp/utils.py
"""
MiniCASP utilities (no Docker, no ASKCOS server).

Core pipeline:
- Load USPTO atom-mapped reaction SMILES from a CSV
- Extract retrosynthesis templates via RDChiral template extractor
- Train a lightweight template relevance model (product -> template_id)
- Run a local multi-step retrosynthesis search using the trained model + templates

Expected CSV:
- At least one column containing reaction SMILES in the form:
  reactants>agents>products
- Atom mapping is strongly recommended for template extraction.

Dependencies:
- rdkit
- rdchiral
- numpy, pandas
- scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import collections
import heapq
import json
import logging
import math
import os
import random
import time
import gzip
import inspect

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

# RDChiral is optional at import-time, but required for templates/applications.
try:
    from rdchiral.template_extractor import extract_from_reaction  # type: ignore
    from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun  # type: ignore
except Exception:  # pragma: no cover
    extract_from_reaction = None
    rdchiralReaction = None
    rdchiralReactants = None
    rdchiralRun = None


# ----------------------------
# Logging
# ----------------------------

def setup_logger(log_level: int = logging.INFO) -> None:
    """Set up a basic console logger."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class ReactionRecord:
    rxn_smiles: str
    reactants: str
    agents: str
    products: str


@dataclass(frozen=True)
class TemplateRecord:
    template_id: int
    rxn_smarts: str
    count: int


@dataclass(frozen=True)
class Step:
    """A single retrosynthesis step: product -> precursors using a template."""
    product: str
    precursors: Tuple[str, ...]
    template_id: int
    template_score: float  # model probability for the chosen template


@dataclass
class Route:
    solved: bool
    steps: List[Step]
    open_mols: Set[str]
    score: float  # lower is better (we use -log probs + depth penalty)


# ----------------------------
# Chemistry helpers
# ----------------------------


def split_rxn_smiles(rxn_smiles: str) -> Tuple[str, str, str]:
    """
    Split reaction SMILES into (reactants, agents, products).

    Supports:
      - reactants>>products
      - reactants>agents>products
    """
    s = rxn_smiles.strip()

    # Handle USPTO-style first: reactants>>products
    if ">>" in s:
        left, right = s.split(">>", 1)
        return left.strip(), "", right.strip()

    # Handle full form: reactants>agents>products
    parts = s.split(">")
    if len(parts) == 3:
        return parts[0].strip(), parts[1].strip(), parts[2].strip()

    raise ValueError(
        "Reaction SMILES not recognized. Expected 'reactants>>products' or 'reactants>agents>products'. "
        f"Got: {s[:200]}"
    )


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize SMILES with RDKit; returns original on failure."""
    smiles = smiles.strip()
    if not smiles:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


def normalize_mol_set(smiles_list: Iterable[str]) -> Tuple[str, ...]:
    """
    Canonicalize and sort a set/list of SMILES to use as a stable key.
    """
    cleaned = []
    for s in smiles_list:
        s = s.strip()
        if not s:
            continue
        cleaned.append(canonicalize_smiles(s))
    return tuple(sorted(set(cleaned)))


def smiles_to_morgan_fp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> Optional[np.ndarray]:
    """Morgan fingerprint as numpy float32 vector (0/1). Returns None on parse failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=use_chirality)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ----------------------------
# IO: load reactions
# ----------------------------

def load_reaction_records_csv(
    csv_path: str,
    rxn_col: str = "rxn_smiles",
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 0,
) -> List[ReactionRecord]:
    """
    Load ReactionRecords from a CSV file.

    If rxn_col doesn't exist, we'll try a few common names.
    """
    df = pd.read_csv(csv_path)
    if rxn_col not in df.columns:
        for alt in ("reaction_smiles", "reaction", "mapped_rxn", "mapped_reaction_smiles", "rxn"):
            if alt in df.columns:
                rxn_col = alt
                break
        else:
            raise ValueError(f"Could not find a reaction SMILES column. Tried rxn_smiles + common fallbacks. "
                             f"Columns: {list(df.columns)[:50]}")

    rxns = df[rxn_col].astype(str).tolist()
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rxns)
    if limit is not None:
        rxns = rxns[:limit]

    out: List[ReactionRecord] = []
    for r in rxns:
        reactants, agents, products = split_rxn_smiles(r)
        out.append(ReactionRecord(rxn_smiles=r, reactants=reactants, agents=agents, products=products))
    return out


# ----------------------------
# Template extraction (RDChiral)
# ----------------------------

def require_rdchiral() -> None:
    """Fail fast with a helpful error if RDChiral isn't importable."""
    if extract_from_reaction is None or rdchiralReaction is None or rdchiralReactants is None or rdchiralRun is None:
        raise RuntimeError(
            "RDChiral is required for template extraction/application but is not importable.\n"
            "Install (typical): pip install rdchiral\n"
            "Or ensure it's on PYTHONPATH."
        )


import hashlib
import inspect
import logging
from typing import Optional

def extract_retro_template_smarts(
    rxn_smiles: str,
    radius: int = 1,
    include_agents: bool = False,
    debug: bool = False,
) -> Optional[str]:
    """
    Extract a retrosynthesis SMARTS template from an atom-mapped reaction SMILES using RDChiral.

    Works across RDChiral variants by:
      - always passing a dict that includes '_id'
      - attempting to pass radius when supported, otherwise falling back cleanly
    """
    require_rdchiral()

    reactants, agents, products = split_rxn_smiles(rxn_smiles)

    # Make a stable-ish unique id (RDChiral requires reaction['_id'] in some versions)
    rid = "minicasp_" + hashlib.md5(rxn_smiles.encode("utf-8")).hexdigest()[:12]

    rxn_dict = {
        "_id": rid,
        "reactants": reactants,
        "products": products,
        "agents": agents if include_agents else "",
        "reaction_smiles": rxn_smiles,
    }

    def _extract_call(obj, use_radius: bool):
        # Some RDChiral versions accept radius=..., others reject it.
        if use_radius:
            return extract_from_reaction(obj, radius=radius)
        return extract_from_reaction(obj)

    # Optional: detect support via signature (helps avoid raising TypeError every time)
    supports_radius = False
    try:
        sig = inspect.signature(extract_from_reaction)
        supports_radius = ("radius" in sig.parameters)
    except Exception:
        # If signature introspection fails, we just try/except below.
        supports_radius = False

    try:
        # Try with radius if it *looks* supported; otherwise try without.
        if supports_radius:
            res = _extract_call(rxn_dict, use_radius=True)
        else:
            res = _extract_call(rxn_dict, use_radius=False)
    except TypeError as e:
        # Fallback if we guessed wrong (e.g., wrapper doesn't expose signature properly)
        if "radius" in str(e):
            if debug:
                logging.warning("RDChiral does not accept radius=...; retrying without radius. (%s)", e)
            try:
                res = _extract_call(rxn_dict, use_radius=False)
            except Exception as e2:
                if debug:
                    logging.exception("RDChiral extraction failed after fallback: %s", e2)
                return None
        else:
            if debug:
                logging.exception("RDChiral extraction TypeError: %s", e)
            return None
    except Exception as e:
        if debug:
            logging.exception("RDChiral extraction failed: %s", e)
        return None

    # Normalize outputs across RDChiral versions
    if isinstance(res, dict):
        smarts = res.get("reaction_smarts") or res.get("retro_smarts") or res.get("smarts")
        if isinstance(smarts, str) and smarts.strip():
            return smarts.strip()
        return None
    if isinstance(res, str) and res.strip():
        return res.strip()
    return None


def build_template_library(
    reactions: Sequence[ReactionRecord],
    radius: int = 1,
    min_count: int = 1,
    debug_failures: int = 5,
) -> Tuple[List[TemplateRecord], Dict[str, int]]:
    """
    Extract templates for all reactions, deduplicate, count, and assign integer IDs.

    debug_failures: print stack traces for first N extraction failures to diagnose issues.
    """
    require_rdchiral()

    counter: Dict[str, int] = collections.Counter()
    fail_printed = 0

    for rec in reactions:
        try:
            smarts = extract_retro_template_smarts(
                rec.rxn_smiles,
                radius=radius,
                debug=(fail_printed < debug_failures),
            )
            if smarts:
                counter[smarts] += 1
            else:
                if fail_printed < debug_failures:
                    logging.warning("Template extraction returned None for rxn: %s", rec.rxn_smiles[:200])
                    fail_printed += 1
        except Exception:
            if fail_printed < debug_failures:
                logging.exception("Template extraction crashed for rxn: %s", rec.rxn_smiles[:200])
                fail_printed += 1
            continue

    items = [(s, c) for s, c in counter.items() if c >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)

    templates: List[TemplateRecord] = []
    smarts_to_id: Dict[str, int] = {}
    for i, (smarts, count) in enumerate(items):
        templates.append(TemplateRecord(template_id=i, rxn_smarts=smarts, count=count))
        smarts_to_id[smarts] = i

    return templates, smarts_to_id

def save_templates_json(path: str, templates: Sequence[TemplateRecord]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            [{"template_id": t.template_id, "rxn_smarts": t.rxn_smarts, "count": t.count} for t in templates],
            f,
            indent=2,
        )

def load_templates_json(path: str) -> List[TemplateRecord]:
    with open(path, "r") as f:
        data = json.load(f)
    return [TemplateRecord(int(d["template_id"]), d["rxn_smarts"], int(d["count"])) for d in data]


# ----------------------------
# Training data construction
# ----------------------------

def make_training_pairs(
    reactions: Sequence[ReactionRecord],
    smarts_to_id: Dict[str, int],
    radius: int = 1,
) -> Tuple[List[str], List[int]]:
    """
    Build (product_smiles, template_id) training pairs from reaction records.

    For multi-product reactions, we pick the whole products string, canonicalized as a dot-joined set.
    """
    require_rdchiral()

    x_smiles: List[str] = []
    y_ids: List[int] = []

    for rec in reactions:
        smarts = extract_retro_template_smarts(rec.rxn_smiles, radius=radius)
        if not smarts:
            continue
        if smarts not in smarts_to_id:
            continue

        prod_parts = [p for p in rec.products.split(".") if p.strip()]
        prod_norm = ".".join(normalize_mol_set(prod_parts))
        if not prod_norm:
            continue

        x_smiles.append(prod_norm)
        y_ids.append(smarts_to_id[smarts])

    return x_smiles, y_ids


def featurize_smiles_list(
    smiles_list: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """
    Convert SMILES -> Morgan fingerprint feature matrix.
    Invalid SMILES become all-zeros (keeps alignment).
    """
    X = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_morgan_fp(smi, radius=radius, n_bits=n_bits)
        if fp is not None:
            X[i] = fp
    return X


# ----------------------------
# Model training / inference
# ----------------------------

@dataclass
class TemplateModel:
    """
    Lightweight template relevance model.
    - label_encoder maps template_id integers to contiguous class indices (and back).
    """
    clf: SGDClassifier
    label_encoder: LabelEncoder
    n_bits: int
    fp_radius: int


def train_template_model(
    product_smiles: Sequence[str],
    template_ids: Sequence[int],
    fp_radius: int = 2,
    n_bits: int = 2048,
    random_state: int = 0,
    max_iter: int = 25,
) -> TemplateModel:
    """
    Train a multiclass classifier: product fingerprint -> template_id.

    Uses SGDClassifier with log_loss (fast, works for many classes).
    """
    if len(product_smiles) != len(template_ids):
        raise ValueError("product_smiles and template_ids must have same length")
    if len(product_smiles) == 0:
        raise ValueError("No training pairs provided.")

    X = featurize_smiles_list(product_smiles, radius=fp_radius, n_bits=n_bits)

    le = LabelEncoder()
    y = le.fit_transform(np.asarray(template_ids, dtype=np.int64))

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=max_iter,
        tol=1e-3,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X, y)

    return TemplateModel(clf=clf, label_encoder=le, n_bits=n_bits, fp_radius=fp_radius)


def predict_topk_templates(
    model: TemplateModel,
    product_smiles: str,
    k: int = 25,
) -> List[Tuple[int, float]]:
    """
    Predict top-k template_ids for a product SMILES.

    Returns:
        list of (template_id, prob) sorted by prob desc.
    """
    x = featurize_smiles_list([product_smiles], radius=model.fp_radius, n_bits=model.n_bits)
    probs = model.clf.predict_proba(x)[0]
    k = min(k, probs.shape[0])
    top_idx = np.argpartition(-probs, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-probs[top_idx])]

    class_ids = model.label_encoder.inverse_transform(top_idx)
    return [(int(tid), float(probs[i])) for tid, i in zip(class_ids, top_idx)]

def save_model(path: str, model: TemplateModel) -> None:
    import joblib
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str) -> TemplateModel:
    import joblib
    return joblib.load(path)



# ----------------------------
# Apply templates (one-step expansion)
# ----------------------------

def apply_retro_template(
    product_smiles: str,
    rxn_smarts: str,
    max_outcomes: int = 50,
) -> List[Tuple[str, ...]]:
    """
    Apply a retrosynthesis template SMARTS to a product SMILES.

    Returns:
        list of precursor tuples (canonicalized, sorted, unique)
    """
    require_rdchiral()

    try:
        rxn = rdchiralReaction(rxn_smarts)
        reactants = rdchiralReactants(product_smiles)
        outcomes = rdchiralRun(rxn, reactants, combine_enantiomers=False)
    except Exception:
        return []

    precursors: List[Tuple[str, ...]] = []
    for out in outcomes[:max_outcomes]:
        parts = [p for p in out.split(".") if p.strip()]
        precursors.append(normalize_mol_set(parts))

    # De-dup while preserving order
    seen: Set[Tuple[str, ...]] = set()
    uniq: List[Tuple[str, ...]] = []
    for pset in precursors:
        if pset and pset not in seen:
            seen.add(pset)
            uniq.append(pset)
    return uniq


# ----------------------------
# Multi-step search (local)
# ----------------------------

@dataclass(frozen=True)
class SearchConfig:
    max_depth: int = 8
    max_expansions: int = 500
    topk_templates: int = 25
    max_outcomes_per_template: int = 25
    time_limit_s: float = 30.0
    depth_penalty: float = 0.2  # encourages shorter routes


def _is_solved(mols: Set[str], buyables: Set[str]) -> bool:
    return all(m in buyables for m in mols)


def plan_route_best_first(
    target_smiles: str,
    model: TemplateModel,
    templates_by_id: Dict[int, str],
    buyables: Set[str],
    config: SearchConfig = SearchConfig(),
) -> Route:
    """
    Best-first (priority queue) search over retrosynthesis states.

    State = set of molecules that still need to be made (open set).
    Expand = pick one open molecule, propose templates, apply them to generate precursors,
             replace expanded molecule with precursors.

    Score (lower is better):
      sum(-log(prob(template))) + depth_penalty*depth
    """
    require_rdchiral()

    start_time = time.time()
    target = canonicalize_smiles(target_smiles)

    if target in buyables:
        return Route(solved=True, steps=[], open_mols=set(), score=0.0)

    # Priority queue item: (score, depth, tie, open_set_frozenset, steps_jsonable)
    tie = 0
    start_open = frozenset([target])
    pq: List[Tuple[float, int, int, frozenset, Tuple[Step, ...]]] = []
    heapq.heappush(pq, (0.0, 0, tie, start_open, tuple()))
    visited: Set[frozenset] = set([start_open])

    expansions = 0

    while pq:
        if time.time() - start_time > config.time_limit_s:
            break
        score, depth, _, open_fs, steps = heapq.heappop(pq)
        open_set = set(open_fs)

        if _is_solved(open_set, buyables):
            return Route(solved=True, steps=list(steps), open_mols=set(), score=score)

        if depth >= config.max_depth:
            continue
        if expansions >= config.max_expansions:
            break

        # Pick an unsolved molecule to expand (largest first is a decent cheap heuristic)
        to_expand = max((m for m in open_set if m not in buyables), key=len, default=None)
        if to_expand is None:
            continue

        top_templates = predict_topk_templates(model, to_expand, k=config.topk_templates)
        expansions += 1

        for template_id, prob in top_templates:
            rxn_smarts = templates_by_id.get(template_id)
            if not rxn_smarts:
                continue

            prec_sets = apply_retro_template(
                product_smiles=to_expand,
                rxn_smarts=rxn_smarts,
                max_outcomes=config.max_outcomes_per_template,
            )
            if not prec_sets:
                continue

            # Use best few precursor sets (already limited)
            for precursors in prec_sets:
                new_open = set(open_set)
                new_open.remove(to_expand)
                for p in precursors:
                    if p not in buyables:
                        new_open.add(p)

                new_open_fs = frozenset(new_open)
                if new_open_fs in visited:
                    continue
                visited.add(new_open_fs)

                # update score: add -log(prob) and depth penalty
                prob_clamped = max(prob, 1e-9)
                new_score = score + (-math.log(prob_clamped)) + config.depth_penalty

                new_steps = steps + (Step(
                    product=to_expand,
                    precursors=precursors,
                    template_id=template_id,
                    template_score=prob,
                ),)

                tie += 1
                heapq.heappush(pq, (new_score, depth + 1, tie, new_open_fs, new_steps))

    # Failed
    best = min(pq, default=(float("inf"), 0, 0, start_open, tuple()), key=lambda x: x[0])
    return Route(solved=False, steps=list(best[4]), open_mols=set(best[3]), score=float(best[0]))


# ----------------------------
# Buyables / building blocks helpers
# ----------------------------

def load_buyables_txt(path: str) -> Set[str]:
    """
    Load buyable SMILES from a .txt file (one SMILES per line).
    """
    buyables: Set[str] = set()
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            buyables.add(canonicalize_smiles(s.split(",")[0].strip()))
    return buyables


def default_buyables_from_reactants(
    reactions: Sequence[ReactionRecord],
    max_unique: int = 200_000,
) -> Set[str]:
    """
    Simple fallback: treat all reactants observed in the dataset as 'buyable'.
    This is not chemically correct, but useful for debugging end-to-end search.
    """
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
    """
    Load ASKCOS buyables.jsonl.gz into a set of canonical SMILES.

    Args:
        path: e.g. /home/gzbrown/higherlev_retro/ASKCOSv2/askcos2_core/buyables/buyables.jsonl.gz
        smiles_key_candidates: keys to look for in each JSON line.
        limit: if >0, stop after loading this many entries (useful for quick tests).

    Returns:
        Set of canonical SMILES.
    """
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

# ----------------------------
# Audit / evaluation helpers
# ----------------------------

def sample_targets_from_products(
    reactions: Sequence[ReactionRecord],
    n: int,
    seed: int = 0,
) -> List[str]:
    """
    Sample target molecules from the product side of reactions.
    """
    rng = random.Random(seed)
    prods = []
    for r in reactions:
        parts = [p for p in r.products.split(".") if p.strip()]
        prod = ".".join(normalize_mol_set(parts))
        if prod:
            prods.append(prod)
    rng.shuffle(prods)
    return prods[:n]


def audit_targets(
    targets: Sequence[str],
    model: TemplateModel,
    templates: Sequence[TemplateRecord],
    buyables: Set[str],
    config: SearchConfig = SearchConfig(),
) -> Dict[str, Any]:
    """
    Run planning on each target and summarize success/failure.
    """
    templates_by_id = {t.template_id: t.rxn_smarts for t in templates}

    results = []
    for i, t in enumerate(targets):
        route = plan_route_best_first(t, model, templates_by_id, buyables, config=config)
        results.append({
            "idx": i,
            "target": t,
            "solved": route.solved,
            "score": route.score,
            "depth": len(route.steps),
            "open_mols": sorted(route.open_mols),
            "steps": [
                {
                    "product": s.product,
                    "precursors": list(s.precursors),
                    "template_id": s.template_id,
                    "template_score": s.template_score,
                }
                for s in route.steps
            ],
        })

    n_solved = sum(1 for r in results if r["solved"])
    return {
        "n_targets": len(targets),
        "n_solved": n_solved,
        "success_rate": (n_solved / max(1, len(targets))),
        "results": results,
    }


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ----------------------------
# Convenience: end-to-end build/train
# ----------------------------

@dataclass
class MiniCaspArtifacts:
    templates: List[TemplateRecord]
    smarts_to_id: Dict[str, int]
    model: TemplateModel
    reactions_used: int


def build_and_train_from_csv(
    csv_path: str,
    rxn_col: str = "rxn_smiles",
    limit: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 0,
    template_radius: int = 1,          # NOTE: kept for API compatibility; ignored by your RDChiral version
    template_min_count: int = 1,
    fp_radius: int = 2,
    n_bits: int = 2048,
    cache_dir: Optional[str] = None,
    cache_tag: str = "default",
    reuse_cache: bool = True,
) -> MiniCaspArtifacts:
    """
    End-to-end:
      - load reactions
      - extract templates (dedup + counts)
      - make training pairs (product -> template_id)
      - train template classifier

    Caching:
      If cache_dir is provided, writes:
        templates.json
        pairs.npz
        model.joblib
        meta.json

      If reuse_cache=True and all exist, loads them and skips extraction/training.
    """
    setup_logger()

    # Build cache paths
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = f"{cache_tag}_lim{limit}_min{template_min_count}_fp{fp_radius}_{n_bits}_seed{seed}"
        tmpl_path = os.path.join(cache_dir, f"templates_{key}.json")
        pairs_path = os.path.join(cache_dir, f"pairs_{key}.npz")
        model_path = os.path.join(cache_dir, f"model_{key}.joblib")
        meta_path = os.path.join(cache_dir, f"meta_{key}.json")
    else:
        tmpl_path = pairs_path = model_path = meta_path = None

    # Reuse cache if present
    if (
        cache_dir and reuse_cache
        and tmpl_path and pairs_path and model_path and meta_path
        and os.path.exists(tmpl_path)
        and os.path.exists(pairs_path)
        and os.path.exists(model_path)
        and os.path.exists(meta_path)
    ):
        logging.info("Loading cached artifacts from %s (tag=%s)", cache_dir, cache_tag)
        templates = load_templates_json(tmpl_path)
        product_smiles, y_ids = load_training_pairs_npz(pairs_path)
        model = load_model_joblib(model_path)
        smarts_to_id = {t.rxn_smarts: t.template_id for t in templates}

        return MiniCaspArtifacts(
            templates=templates,
            smarts_to_id=smarts_to_id,
            model=model,
            reactions_used=0,
        )

    # Load reactions
    reactions = load_reaction_records_csv(
        csv_path=csv_path,
        rxn_col=rxn_col,
        limit=limit,
        shuffle=shuffle,
        seed=seed,
    )
    logging.info("Loaded %d reactions from %s", len(reactions), csv_path)

    # Extract templates (RDChiral)
    # NOTE: your rdchiral expects reaction dict with '_id'; also does NOT accept radius kwarg.
    templates, smarts_to_id = build_template_library(
        reactions=reactions,
        radius=template_radius,        # passed through your wrapper; your extractor should ignore radius internally
        min_count=template_min_count,
    )
    logging.info("Extracted %d unique templates (min_count=%d)", len(templates), template_min_count)

    # Training pairs
    product_smiles, y_ids = make_training_pairs(
        reactions=reactions,
        smarts_to_id=smarts_to_id,
        radius=template_radius,
    )
    logging.info("Built %d training pairs", len(product_smiles))

    if len(product_smiles) == 0:
        raise ValueError(
            "No training pairs produced. This usually means template extraction is failing "
            "or templates got filtered out by min_count."
        )

    # Train
    model = train_template_model(
        product_smiles=product_smiles,
        template_ids=y_ids,
        fp_radius=fp_radius,
        n_bits=n_bits,
        random_state=seed,
    )
    logging.info("Trained model with %d classes", len(model.label_encoder.classes_))

    # Save cache
    if cache_dir and tmpl_path and pairs_path and model_path and meta_path:
        save_templates_json(tmpl_path, templates)
        save_training_pairs_npz(pairs_path, product_smiles, y_ids)
        save_model_joblib(model_path, model)
        save_meta_json(meta_path, {
            "csv_path": csv_path,
            "rxn_col": rxn_col,
            "limit": limit,
            "shuffle": shuffle,
            "seed": seed,
            "template_radius": template_radius,
            "template_min_count": template_min_count,
            "fp_radius": fp_radius,
            "n_bits": n_bits,
            "n_reactions_loaded": len(reactions),
            "n_templates": len(templates),
            "n_pairs": len(product_smiles),
        })
        logging.info("Saved cache to %s", cache_dir)

    return MiniCaspArtifacts(
        templates=templates,
        smarts_to_id=smarts_to_id,
        model=model,
        reactions_used=len(reactions),
    )

# -- helpers --

import joblib

def save_templates_json(path: str, templates: Sequence[TemplateRecord]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            [{"template_id": t.template_id, "rxn_smarts": t.rxn_smarts, "count": t.count} for t in templates],
            f,
            indent=2,
        )

def load_templates_json(path: str) -> List[TemplateRecord]:
    with open(path, "r") as f:
        data = json.load(f)
    return [TemplateRecord(int(d["template_id"]), d["rxn_smarts"], int(d["count"])) for d in data]

def save_training_pairs_npz(path: str, product_smiles: Sequence[str], template_ids: Sequence[int]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, product_smiles=np.array(product_smiles, dtype=object),
                        template_ids=np.array(template_ids, dtype=np.int64))

def load_training_pairs_npz(path: str) -> Tuple[List[str], List[int]]:
    z = np.load(path, allow_pickle=True)
    return z["product_smiles"].tolist(), z["template_ids"].astype(int).tolist()

def save_model_joblib(path: str, model: TemplateModel) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)

def load_model_joblib(path: str) -> TemplateModel:
    return joblib.load(path)

def save_meta_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_templates_cache(path: str, templates: Sequence[TemplateRecord]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = [{"template_id": t.template_id, "rxn_smarts": t.rxn_smarts, "count": t.count} for t in templates]
    if path.endswith(".gz"):
        import gzip
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f)
    else:
        with open(path, "w") as f:
            json.dump(payload, f)

def load_templates_cache(path: str) -> List[TemplateRecord]:
    if path.endswith(".gz"):
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        with open(path, "r") as f:
            payload = json.load(f)
    return [TemplateRecord(int(x["template_id"]), x["rxn_smarts"], int(x.get("count", 1))) for x in payload]
