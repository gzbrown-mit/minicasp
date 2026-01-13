from __future__ import annotations
from typing import Any
import json, os

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_meta_json(path: str, obj: Any) -> None:
    save_json(path, obj)
