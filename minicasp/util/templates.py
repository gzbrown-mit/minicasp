from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import collections
import gzip
import hashlib
import inspect
import json
import logging

from .chem import split_rxn_smiles
from .data import ReactionRecord

try:
    from rdchiral.template_extractor import extract_from_reaction  # type: ignore
    from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun  # type: ignore
except Exception:  # pragma: no cover
    extract_from_reaction = None
    rdchiralReaction = None
    rdchiralReactants = None
    rdchiralRun = None

@dataclass(frozen=True)
class TemplateRecord:
    template_id: int
    rxn_smarts: str
    count: int

def require_rdchiral() -> None:
    if extract_from_reaction is None or rdchiralReaction is None or rdchiralReactants is None or rdchiralRun is None:
        raise RuntimeError(
            "RDChiral is required for template extraction/application but is not importable.\n"
            "Install: pip install rdchiral"
        )

def extract_retro_template_smarts(
    rxn_smiles: str,
    radius: int = 1,
    include_agents: bool = False,
    debug: bool = False,
) -> Optional[str]:
    require_rdchiral()

    reactants, agents, products = split_rxn_smiles(rxn_smiles)
    rid = "minicasp_" + hashlib.md5(rxn_smiles.encode("utf-8")).hexdigest()[:12]

    rxn_dict = {
        "_id": rid,
        "reactants": reactants,
        "products": products,
        "agents": agents if include_agents else "",
        "reaction_smiles": rxn_smiles,
    }

    supports_radius = False
    try:
        sig = inspect.signature(extract_from_reaction)
        supports_radius = ("radius" in sig.parameters)
    except Exception:
        supports_radius = False

    try:
        if supports_radius:
            res = extract_from_reaction(rxn_dict, radius=radius)
        else:
            res = extract_from_reaction(rxn_dict)
    except TypeError as e:
        if "radius" in str(e):
            if debug:
                logging.warning("RDChiral doesn't accept radius=...; retrying without radius: %s", e)
            try:
                res = extract_from_reaction(rxn_dict)
            except Exception as e2:
                if debug:
                    logging.exception("RDChiral extraction failed after fallback: %s", e2)
                return None
        else:
            if debug:
                logging.exception("RDChiral TypeError: %s", e)
            return None
    except Exception as e:
        if debug:
            logging.exception("RDChiral extraction failed: %s", e)
        return None

    if isinstance(res, dict):
        smarts = res.get("reaction_smarts") or res.get("retro_smarts") or res.get("smarts")
        return smarts.strip() if isinstance(smarts, str) and smarts.strip() else None
    if isinstance(res, str) and res.strip():
        return res.strip()
    return None

def build_template_library(
    reactions: Sequence[ReactionRecord],
    radius: int = 1,
    min_count: int = 1,
    debug_failures: int = 5,
) -> Tuple[List[TemplateRecord], Dict[str, int]]:
    require_rdchiral()

    counter: Dict[str, int] = collections.Counter()
    fail_printed = 0

    for rec in reactions:
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

    items = [(s, c) for s, c in counter.items() if c >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)

    templates: List[TemplateRecord] = []
    smarts_to_id: Dict[str, int] = {}
    for i, (smarts, count) in enumerate(items):
        templates.append(TemplateRecord(template_id=i, rxn_smarts=smarts, count=count))
        smarts_to_id[smarts] = i

    return templates, smarts_to_id

def save_templates_cache(path: str, templates: Sequence[TemplateRecord]) -> None:
    payload = [{"template_id": t.template_id, "rxn_smarts": t.rxn_smarts, "count": t.count} for t in templates]
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f)
    else:
        with open(path, "w") as f:
            json.dump(payload, f)

def load_templates_cache(path: str) -> List[TemplateRecord]:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        with open(path, "r") as f:
            payload = json.load(f)
    return [TemplateRecord(int(x["template_id"]), x["rxn_smarts"], int(x.get("count", 1))) for x in payload]
