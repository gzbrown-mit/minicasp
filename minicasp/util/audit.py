from __future__ import annotations
from typing import Any, Dict, List, Sequence, Set
import random
from .chem import normalize_mol_set
from .data import ReactionRecord
from .templates import TemplateRecord
from .model import TemplateModel
from .search import SearchConfig, plan_route_best_first

def sample_targets_from_products(reactions: Sequence[ReactionRecord], n: int, seed: int = 0) -> List[str]:
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
