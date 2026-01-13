from __future__ import annotations
from typing import Sequence
import numpy as np
from .chem import smiles_to_morgan_fp

def featurize_smiles_list(
    smiles_list: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    X = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_morgan_fp(smi, radius=radius, n_bits=n_bits)
        if fp is not None:
            X[i] = fp
    return X
