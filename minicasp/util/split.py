from __future__ import annotations
from typing import Any, Dict, List, Tuple
import random

def ordered_group_split(
    X: List[str],
    y: List[int],
    groups: List[str],
    test_size: float = 0.2,
) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Dict[str, Any]]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1")

    n = len(X)
    if not (len(y) == n and len(groups) == n):
        raise ValueError("X, y, groups must have same length")

    first_idx: Dict[str, int] = {}
    for i, g in enumerate(groups):
        if g not in first_idx:
            first_idx[g] = i

    groups_in_order = sorted(first_idx.keys(), key=lambda g: first_idx[g])
    n_groups = len(groups_in_order)
    n_test_groups = max(1, int(round(test_size * n_groups)))
    split_point = n_groups - n_test_groups

    train_groups = set(groups_in_order[:split_point])
    test_groups = set(groups_in_order[split_point:])

    X_train, y_train, X_test, y_test = [], [], [], []
    for xi, yi, gi in zip(X, y, groups):
        if gi in train_groups:
            X_train.append(xi); y_train.append(yi)
        else:
            X_test.append(xi); y_test.append(yi)

    meta = {
        "mode": "ordered_group",
        "n_pairs": n,
        "n_groups": n_groups,
        "n_train_groups": len(train_groups),
        "n_test_groups": len(test_groups),
        "n_train_pairs": len(X_train),
        "n_test_pairs": len(X_test),
        "test_size": test_size,
        "split_point_groups": split_point,
    }
    return (X_train, y_train), (X_test, y_test), meta

def random_group_split(
    X: List[str],
    y: List[int],
    groups: List[str],
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Dict[str, Any]]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1")

    n = len(X)
    if not (len(y) == n and len(groups) == n):
        raise ValueError("X, y, groups must have same length")

    uniq_groups = list(sorted(set(groups)))
    rng = random.Random(seed)
    rng.shuffle(uniq_groups)

    n_groups = len(uniq_groups)
    n_test_groups = max(1, int(round(test_size * n_groups)))
    test_groups = set(uniq_groups[:n_test_groups])

    X_train, y_train, X_test, y_test = [], [], [], []
    for xi, yi, gi in zip(X, y, groups):
        if gi in test_groups:
            X_test.append(xi); y_test.append(yi)
        else:
            X_train.append(xi); y_train.append(yi)

    meta = {
        "mode": "random_group",
        "seed": seed,
        "n_pairs": n,
        "n_groups": n_groups,
        "n_test_groups": len(test_groups),
        "n_train_pairs": len(X_train),
        "n_test_pairs": len(X_test),
        "test_size": test_size,
    }
    return (X_train, y_train), (X_test, y_test), meta
