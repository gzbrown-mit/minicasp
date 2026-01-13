from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import random
import pandas as pd
from .chem import split_rxn_smiles

@dataclass(frozen=True)
class ReactionRecord:
    rxn_smiles: str
    reactants: str
    agents: str
    products: str

def load_reaction_records_csv(
    csv_path: str,
    rxn_col: str = "rxn_smiles",
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 0,
) -> List[ReactionRecord]:
    df = pd.read_csv(csv_path)
    if rxn_col not in df.columns:
        for alt in ("reaction_smiles", "reaction", "mapped_rxn", "mapped_reaction_smiles", "rxn"):
            if alt in df.columns:
                rxn_col = alt
                break
        else:
            raise ValueError(
                f"Could not find a reaction SMILES column. Tried {rxn_col} + fallbacks. "
                f"Columns: {list(df.columns)[:50]}"
            )

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
