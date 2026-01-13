from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import heapq, math, time

from .chem import canonicalize_smiles, normalize_mol_set
from .model import TemplateModel, predict_topk_templates
from .templates import require_rdchiral, rdchiralReaction, rdchiralReactants, rdchiralRun


@dataclass(frozen=True)
class Step:
    product: str
    precursors: Tuple[str, ...]
    template_id: int
    template_score: float


@dataclass
class Route:
    solved: bool
    steps: List[Step]
    open_mols: Set[str]
    score: Optional[float]
    fail_reason: str = ""     # <-- add this
    solved_via: str = ""      # <-- "buyable" | "search"


@dataclass(frozen=True)
class SearchConfig:
    max_depth: int = 8
    max_expansions: int = 500
    topk_templates: int = 25
    max_outcomes_per_template: int = 25
    time_limit_s: float = 30.0
    depth_penalty: float = 0.2


def apply_retro_template(product_smiles: str, rxn_smarts: str, max_outcomes: int = 50) -> List[Tuple[str, ...]]:
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

    seen = set()
    uniq = []
    for pset in precursors:
        if pset and pset not in seen:
            seen.add(pset)
            uniq.append(pset)
    return uniq


def _is_solved(mols: Set[str], buyables: Set[str]) -> bool:
    return all(m in buyables for m in mols)


def plan_route_best_first(
    target_smiles: str,
    model: TemplateModel,
    templates_by_id: Dict[int, str],
    buyables: Set[str],
    config: SearchConfig = SearchConfig(),
) -> Route:
    require_rdchiral()

    start_time = time.time()
    target = canonicalize_smiles(target_smiles)

    # Trivial solution: already buyable
    if target in buyables:
        return Route(
            solved=True,
            steps=[],
            open_mols=set(),
            score=0.0,
            fail_reason="",
            solved_via="buyable",
        )

    tie = 0
    start_open = frozenset([target])
    pq: List[Tuple[float, int, int, frozenset, Tuple[Step, ...]]] = []
    heapq.heappush(pq, (0.0, 0, tie, start_open, tuple()))
    visited = {start_open}
    expansions = 0

    # best-so-far for failure return
    best_score = float("inf")
    best_open_fs = start_open
    best_steps: Tuple[Step, ...] = tuple()
    fail_reason = "no_expansion"

    while pq:
        if time.time() - start_time > config.time_limit_s:
            fail_reason = "time_limit"
            break
        if expansions >= config.max_expansions:
            fail_reason = "max_expansions"
            break

        score, depth, _, open_fs, steps = heapq.heappop(pq)

        # update best-so-far
        if score < best_score:
            best_score = score
            best_open_fs = open_fs
            best_steps = steps

        open_set = set(open_fs)

        if _is_solved(open_set, buyables):
            return Route(
                solved=True,
                steps=list(steps),
                open_mols=set(),
                score=score,
                fail_reason="",
                solved_via="search",
            )

        if depth >= config.max_depth:
            fail_reason = "max_depth"
            continue

        to_expand = max((m for m in open_set if m not in buyables), key=len, default=None)
        if to_expand is None:
            fail_reason = "no_unsolved"
            continue

        top_templates = predict_topk_templates(model, to_expand, k=config.topk_templates)
        expansions += 1

        any_child = False

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

            any_child = True

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

                prob_clamped = max(prob, 1e-9)
                new_score = score + (-math.log(prob_clamped)) + config.depth_penalty
                new_steps = steps + (Step(to_expand, precursors, template_id, float(prob)),)

                tie += 1
                heapq.heappush(pq, (new_score, depth + 1, tie, new_open_fs, new_steps))

        if not any_child:
            fail_reason = "no_applicable_templates"

    # Failure return (no Infinity, still returns best explored)
    return Route(
        solved=False,
        steps=list(best_steps),
        open_mols=set(best_open_fs),
        score=None if best_score == float("inf") else float(best_score),
        fail_reason=fail_reason,
        solved_via="",
    )
