from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import os
import joblib

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

from .features import featurize_smiles_list


@dataclass
class TemplateModel:
    """
    Pluggable template relevance model.

    kind:
      - "sgd"
      - "mlp_sklearn"
      - "mlp_torch"
    """
    kind: str
    label_encoder: LabelEncoder
    n_bits: int
    fp_radius: int

    # For sklearn models (sgd, mlp_sklearn)
    clf: Optional[Any] = None

    # For torch MLP (we store weights/config only)
    torch_state: Optional[Dict[str, np.ndarray]] = None
    torch_config: Optional[Dict[str, Any]] = None


def train_template_model(
    product_smiles: Sequence[str],
    template_ids: Sequence[int],
    fp_radius: int = 2,
    n_bits: int = 2048,
    random_state: int = 0,
    max_iter: int = 25,
    # NEW:
    model_type: str = "sgd",  # "sgd", "mlp", "mlp_torch", "mlp_sklearn"
    mlp_hidden: Tuple[int, ...] = (1024, 1024),
    mlp_dropout: float = 0.1,
    mlp_epochs: int = 8,
    mlp_batch_size: int = 256,
    mlp_lr: float = 1e-3,
    mlp_weight_decay: float = 1e-5,
) -> TemplateModel:
    if len(product_smiles) != len(template_ids):
        raise ValueError("product_smiles and template_ids must have same length")
    if len(product_smiles) == 0:
        raise ValueError("No training pairs provided.")

    X = featurize_smiles_list(product_smiles, radius=fp_radius, n_bits=n_bits)
    le = LabelEncoder()
    y = le.fit_transform(np.asarray(template_ids, dtype=np.int64))
    n_classes = int(len(le.classes_))

    # Alias: "mlp" prefers torch if available, otherwise sklearn
    if model_type == "mlp":
        try:
            import torch  # noqa: F401
            model_type = "mlp_torch"
        except Exception:
            model_type = "mlp_sklearn"

    if model_type == "sgd":
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
        return TemplateModel(kind="sgd", clf=clf, label_encoder=le, n_bits=n_bits, fp_radius=fp_radius)

    if model_type == "mlp_sklearn":
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(
            hidden_layer_sizes=mlp_hidden,
            activation="relu",
            solver="adam",
            alpha=mlp_weight_decay,
            batch_size=min(int(mlp_batch_size), len(X)),
            learning_rate_init=mlp_lr,
            max_iter=max(50, int(mlp_epochs) * 20),  # rough mapping (sklearn uses iters)
            random_state=random_state,
            verbose=False,
        )
        clf.fit(X, y)
        return TemplateModel(kind="mlp_sklearn", clf=clf, label_encoder=le, n_bits=n_bits, fp_radius=fp_radius)

    if model_type == "mlp_torch":
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except Exception as e:
            raise RuntimeError("Requested mlp_torch but PyTorch is not available in this environment.") from e

        # simple MLP
        layers: List[nn.Module] = []
        in_dim = n_bits
        for h in mlp_hidden:
            layers += [nn.Linear(in_dim, int(h)), nn.ReLU(), nn.Dropout(float(mlp_dropout))]
            in_dim = int(h)
        layers += [nn.Linear(in_dim, n_classes)]
        net = nn.Sequential(*layers)

        # data loader
        Xt = torch.from_numpy(X).float()
        yt = torch.from_numpy(y.astype(np.int64))
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=min(int(mlp_batch_size), len(ds)), shuffle=True)

        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(net.parameters(), lr=float(mlp_lr), weight_decay=float(mlp_weight_decay))

        # deterministic-ish
        torch.manual_seed(int(random_state))

        net.train()
        for _epoch in range(int(mlp_epochs)):
            for xb, yb in dl:
                opt.zero_grad(set_to_none=True)
                logits = net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

        # store state as numpy arrays (joblib-friendly)
        state_np: Dict[str, np.ndarray] = {}
        for k, v in net.state_dict().items():
            state_np[k] = v.detach().cpu().numpy()

        cfg = {
            "hidden": tuple(int(h) for h in mlp_hidden),
            "dropout": float(mlp_dropout),
            "n_classes": int(n_classes),
        }

        return TemplateModel(
            kind="mlp_torch",
            clf=None,
            label_encoder=le,
            n_bits=n_bits,
            fp_radius=fp_radius,
            torch_state=state_np,
            torch_config=cfg,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def _torch_predict_proba(model: TemplateModel, x: np.ndarray) -> np.ndarray:
    # model.kind must be "mlp_torch"
    import torch
    import torch.nn as nn

    cfg = model.torch_config or {}
    hidden = cfg.get("hidden", (1024, 1024))
    dropout = float(cfg.get("dropout", 0.1))
    n_classes = int(cfg["n_classes"])

    layers: List[nn.Module] = []
    in_dim = model.n_bits
    for h in hidden:
        layers += [nn.Linear(in_dim, int(h)), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = int(h)
    layers += [nn.Linear(in_dim, n_classes)]
    net = nn.Sequential(*layers)

    # rebuild state_dict from numpy
    state = {}
    for k, v in (model.torch_state or {}).items():
        state[k] = torch.from_numpy(v)
    net.load_state_dict(state, strict=True)
    net.eval()

    with torch.no_grad():
        xt = torch.from_numpy(x).float()
        logits = net(xt)[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float32)
    return probs


def predict_topk_templates(model: TemplateModel, product_smiles: str, k: int = 25) -> List[Tuple[int, float]]:
    x = featurize_smiles_list([product_smiles], radius=model.fp_radius, n_bits=model.n_bits)

    if model.kind in ("sgd", "mlp_sklearn"):
        probs = model.clf.predict_proba(x)[0].astype(np.float32)
    elif model.kind == "mlp_torch":
        probs = _torch_predict_proba(model, x)
    else:
        raise ValueError(f"Unknown model kind: {model.kind}")

    k = min(int(k), int(probs.shape[0]))
    top_idx = np.argpartition(-probs, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-probs[top_idx])]

    # inverse_transform maps class-index -> ORIGINAL template_id
    class_ids = model.label_encoder.inverse_transform(top_idx)
    return [(int(tid), float(probs[i])) for tid, i in zip(class_ids, top_idx)]


def save_model_joblib(path: str, model: TemplateModel) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Save as a dict for forward-compat (still loads older TemplateModel objects too)
    payload = {
        "kind": model.kind,
        "label_encoder": model.label_encoder,
        "n_bits": model.n_bits,
        "fp_radius": model.fp_radius,
        "clf": model.clf,
        "torch_state": model.torch_state,
        "torch_config": model.torch_config,
    }
    joblib.dump(payload, path)


def load_model_joblib(path: str) -> TemplateModel:
    obj = joblib.load(path)

    # Backward compat: if you previously saved a TemplateModel directly
    if isinstance(obj, TemplateModel):
        return obj

    # New format: dict payload
    return TemplateModel(
        kind=obj["kind"],
        label_encoder=obj["label_encoder"],
        n_bits=obj["n_bits"],
        fp_radius=obj["fp_radius"],
        clf=obj.get("clf"),
        torch_state=obj.get("torch_state"),
        torch_config=obj.get("torch_config"),
    )
