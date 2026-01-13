from __future__ import annotations
from typing import Iterable, Optional, Tuple
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def split_rxn_smiles(rxn_smiles: str) -> Tuple[str, str, str]:
    s = rxn_smiles.strip()

    if ">>" in s:
        left, right = s.split(">>", 1)
        return left.strip(), "", right.strip()

    parts = s.split(">")
    if len(parts) == 3:
        return parts[0].strip(), parts[1].strip(), parts[2].strip()

    raise ValueError(
        "Reaction SMILES not recognized. Expected 'reactants>>products' or 'reactants>agents>products'. "
        f"Got: {s[:200]}"
    )

def canonicalize_smiles(smiles: str) -> str:
    smiles = smiles.strip()
    if not smiles:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)

def normalize_mol_set(smiles_list: Iterable[str]) -> Tuple[str, ...]:
    cleaned = []
    for s in smiles_list:
        s = s.strip()
        if not s:
            continue
        cleaned.append(canonicalize_smiles(s))
    return tuple(sorted(set(cleaned)))

def canonical_product_key(products_smiles: str) -> str:
    parts = [p for p in str(products_smiles).split(".") if p.strip()]
    return ".".join(normalize_mol_set(parts))

def smiles_to_morgan_fp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=use_chirality)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
def strip_atom_maps(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    if not m:
        return smiles
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m, canonical=True)
