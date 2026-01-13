# minicasp/utils.py
# Thin re-export facade so existing imports keep working.

from .util.log import setup_logger

from .util.chem import (
    split_rxn_smiles,
    canonicalize_smiles,
    normalize_mol_set,
    smiles_to_morgan_fp,
    canonical_product_key,
    strip_atom_maps,
)

from .util.data import ReactionRecord, load_reaction_records_csv

from .util.templates import (
    TemplateRecord,
    require_rdchiral,
    extract_retro_template_smarts,
    build_template_library,
    save_templates_cache,
    load_templates_cache,
)

from .util.pairs import (
    make_training_pairs,
    save_training_pairs_npz,
    load_training_pairs_npz,
    save_pairs_jsonl_gz,
    load_pairs_jsonl_gz,
)

from .util.features import featurize_smiles_list

from .util.model import (
    TemplateModel,
    train_template_model,
    predict_topk_templates,
    save_model_joblib,
    load_model_joblib,
)

from .util.search import (
    Step,
    Route,
    SearchConfig,
    apply_retro_template,
    plan_route_best_first,
)

from .util.audit import (
    sample_targets_from_products,
    audit_targets,
)

from .util.buyables import (
    load_buyables_txt,
    default_buyables_from_reactants,
    load_askcos_buyables_jsonl_gz,
    save_smiles_txt_gz,
    load_buyables_cached,
)

from .util.split import (
    ordered_group_split,
    random_group_split,
)

from .util.io import (
    save_json,
    save_meta_json,
)

from .util.pipeline import (
    MiniCaspArtifacts,
    build_and_train_from_csv,
)
