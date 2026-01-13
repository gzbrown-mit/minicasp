from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np
import os
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from .features import featurize_smiles_list

@dataclass
class TemplateModel:
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

def predict_topk_templates(model: TemplateModel, product_smiles: str, k: int = 25) -> List[Tuple[int, float]]:
    x = featurize_smiles_list([product_smiles], radius=model.fp_radius, n_bits=model.n_bits)
    probs = model.clf.predict_proba(x)[0]
    k = min(k, probs.shape[0])

    top_idx = np.argpartition(-probs, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-probs[top_idx])]
    class_ids = model.label_encoder.inverse_transform(top_idx)
    return [(int(tid), float(probs[i])) for tid, i in zip(class_ids, top_idx)]

def save_model_joblib(path: str, model: TemplateModel) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)

def load_model_joblib(path: str) -> TemplateModel:
    return joblib.load(path)
