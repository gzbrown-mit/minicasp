"""Structural annotation helpers for reaction SMARTS template curation."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Sequence, Tuple

from rdkit.Chem import rdChemReactions

HETERO_ATOMIC_NUMS = {7, 8, 16}


@dataclass(frozen=True)
class AtomFeatures:
    """Structural features tracked for mapped atoms in a template side."""

    is_in_ring: bool
    is_aromatic: bool
    in_heterocycle: bool


@dataclass(frozen=True)
class BondFeatures:
    """Structural features tracked for mapped bonds in a template side."""

    bond_type: str
    is_aromatic: bool
    is_in_ring: bool
    in_heterocycle: bool


def normalize_smarts(smarts: str) -> str:
    """Trim whitespace without attempting semantic SMARTS canonicalization."""

    return str(smarts).strip()


def annotate_templates(smarts_list: Sequence[str]) -> Dict[str, Dict[str, object]]:
    """Annotate a sequence of exact template SMARTS strings."""

    annotations: Dict[str, Dict[str, object]] = {}
    for smarts in smarts_list:
        normalized = normalize_smarts(smarts)
        if normalized not in annotations:
            annotations[normalized] = analyze_template_smarts(normalized)
    return annotations


@lru_cache(maxsize=250000)
def analyze_template_smarts(retro_template: str) -> Dict[str, object]:
    """Return structural flags used by the curation pipeline."""

    smarts = normalize_smarts(retro_template)
    try:
        reaction = rdChemReactions.ReactionFromSmarts(smarts)
    except Exception as err:  # pragma: no cover - defensive against malformed input
        return _failed_annotation(f"parse_error:{type(err).__name__}")

    if reaction is None:
        return _failed_annotation("parse_error:none")
    if reaction.GetNumProductTemplates() == 0:
        return _failed_annotation("parse_error:no_products")
    if reaction.GetNumReactantTemplates() == 0:
        return _failed_annotation("parse_error:no_reactants")

    product_atoms, product_bonds = _collect_side_features(
        reaction.GetProductTemplate(idx) for idx in range(reaction.GetNumProductTemplates())
    )
    reactant_atoms, reactant_bonds = _collect_side_features(
        reaction.GetReactantTemplate(idx) for idx in range(reaction.GetNumReactantTemplates())
    )

    changed_bond_keys = {
        bond_key
        for bond_key in set(product_bonds) | set(reactant_bonds)
        if product_bonds.get(bond_key) != reactant_bonds.get(bond_key)
    }
    changed_atom_maps = {map_num for bond_key in changed_bond_keys for map_num in bond_key}

    forms_heterocycle = any(
        bond_key in product_bonds
        and product_bonds[bond_key].is_in_ring
        and product_bonds[bond_key].in_heterocycle
        and product_bonds.get(bond_key) != reactant_bonds.get(bond_key)
        for bond_key in changed_bond_keys
    )

    acts_on_heterocycle = any(
        product_atoms.get(map_num, reactant_atoms.get(map_num, AtomFeatures(False, False, False))).in_heterocycle
        or reactant_atoms.get(map_num, product_atoms.get(map_num, AtomFeatures(False, False, False))).in_heterocycle
        for map_num in changed_atom_maps
    )

    return {
        "forms_heterocycle": forms_heterocycle,
        "acts_on_heterocycle": acts_on_heterocycle,
        "tier_candidate": _tier_candidate(forms_heterocycle, acts_on_heterocycle),
        "annotation_status": "ok",
    }


def _failed_annotation(status: str) -> Dict[str, object]:
    return {
        "forms_heterocycle": False,
        "acts_on_heterocycle": False,
        "tier_candidate": "other",
        "annotation_status": status,
    }


def _tier_candidate(forms_heterocycle: bool, acts_on_heterocycle: bool) -> str:
    if forms_heterocycle:
        return "heterocycle_forming_candidate"
    if acts_on_heterocycle:
        return "heterocycle_editing_candidate"
    return "other"


def _collect_side_features(mols: Iterable[object]) -> Tuple[Dict[int, AtomFeatures], Dict[Tuple[int, int], BondFeatures]]:
    atom_features: Dict[int, AtomFeatures] = {}
    bond_features: Dict[Tuple[int, int], BondFeatures] = {}

    for mol in mols:
        component_by_atom_idx, heterocycle_components = _heterocycle_components(mol)

        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num <= 0:
                continue

            component_id = component_by_atom_idx.get(atom.GetIdx())
            features = AtomFeatures(
                is_in_ring=atom.IsInRing(),
                is_aromatic=atom.GetIsAromatic(),
                in_heterocycle=component_id in heterocycle_components if component_id is not None else False,
            )
            atom_features[map_num] = _merge_atom_features(atom_features.get(map_num), features)

        for bond in mol.GetBonds():
            begin_map = bond.GetBeginAtom().GetAtomMapNum()
            end_map = bond.GetEndAtom().GetAtomMapNum()
            if begin_map <= 0 or end_map <= 0:
                continue

            begin_component = component_by_atom_idx.get(bond.GetBeginAtomIdx())
            end_component = component_by_atom_idx.get(bond.GetEndAtomIdx())
            in_heterocycle = (
                begin_component is not None
                and begin_component == end_component
                and begin_component in heterocycle_components
            )

            features = BondFeatures(
                bond_type=str(bond.GetBondType()),
                is_aromatic=bond.GetIsAromatic(),
                is_in_ring=bond.IsInRing(),
                in_heterocycle=in_heterocycle,
            )
            bond_key = tuple(sorted((begin_map, end_map)))
            bond_features[bond_key] = features

    return atom_features, bond_features


def _heterocycle_components(mol: object) -> Tuple[Dict[int, int], set[int]]:
    ring_graph: Dict[int, set[int]] = defaultdict(set)
    hetero_atoms: set[int] = set()

    for atom in mol.GetAtoms():
        if atom.IsInRing() and atom.GetAtomicNum() in HETERO_ATOMIC_NUMS:
            hetero_atoms.add(atom.GetIdx())

    for bond in mol.GetBonds():
        if not bond.IsInRing():
            continue
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        ring_graph[begin_idx].add(end_idx)
        ring_graph[end_idx].add(begin_idx)

    component_by_atom_idx: Dict[int, int] = {}
    heterocycle_components: set[int] = set()
    next_component_id = 0

    for atom_idx in ring_graph:
        if atom_idx in component_by_atom_idx:
            continue

        queue = deque([atom_idx])
        component_atom_idxs = []
        component_contains_hetero = False

        while queue:
            current_idx = queue.popleft()
            if current_idx in component_by_atom_idx:
                continue
            component_by_atom_idx[current_idx] = next_component_id
            component_atom_idxs.append(current_idx)
            component_contains_hetero = component_contains_hetero or current_idx in hetero_atoms
            queue.extend(ring_graph[current_idx])

        if component_contains_hetero:
            heterocycle_components.add(next_component_id)
        next_component_id += 1

    return component_by_atom_idx, heterocycle_components


def _merge_atom_features(existing: AtomFeatures | None, new: AtomFeatures) -> AtomFeatures:
    if existing is None:
        return new
    return AtomFeatures(
        is_in_ring=existing.is_in_ring or new.is_in_ring,
        is_aromatic=existing.is_aromatic or new.is_aromatic,
        in_heterocycle=existing.in_heterocycle or new.in_heterocycle,
    )

