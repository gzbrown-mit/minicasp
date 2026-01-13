from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging, os

from .log import setup_logger
from .data import load_reaction_records_csv
from .templates import TemplateRecord, build_template_library, save_templates_cache, load_templates_cache
from .pairs import make_training_pairs
from .model import TemplateModel, train_template_model

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
    template_radius: int = 1,
    template_min_count: int = 1,
    fp_radius: int = 2,
    n_bits: int = 2048,
) -> MiniCaspArtifacts:
    setup_logger()
    reactions = load_reaction_records_csv(csv_path, rxn_col=rxn_col, limit=limit, shuffle=shuffle, seed=seed)
    logging.info("Loaded %d reactions", len(reactions))

    templates, smarts_to_id = build_template_library(reactions, radius=template_radius, min_count=template_min_count)
    logging.info("Extracted %d templates", len(templates))

    X, y = make_training_pairs(reactions, smarts_to_id, radius=template_radius)
    logging.info("Built %d pairs", len(X))
    if not X:
        raise ValueError("No training pairs produced.")

    model = train_template_model(X, y, fp_radius=fp_radius, n_bits=n_bits, random_state=seed)
    logging.info("Trained model classes=%d", len(model.label_encoder.classes_))

    return MiniCaspArtifacts(templates=templates, smarts_to_id=smarts_to_id, model=model, reactions_used=len(reactions))
