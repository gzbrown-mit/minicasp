""" Module containing routines for returning package paths
"""
import os
from typing import Optional


def package_path() -> str:
    """Return the path to the package"""
    return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def data_path() -> str:
    """Return the path to the ``data`` directory of the package"""
    return os.path.join(package_path(), "data")


def zinc_stock_path() -> str:
    """Return the canonical path to the bundled ZINC stock file."""
    return os.path.join(data_path(), "buyables", "zinc_stock_17_04_20.hdf5")


def uspto_original_path() -> str:
    """Return the canonical path to the bundled USPTO original reactions file."""
    return os.path.join(data_path(), "reactions", "uspto_original.csv")


def uspto_higher_level_path() -> str:
    """Return the canonical path to the bundled USPTO higher-level reactions file."""
    return os.path.join(data_path(), "reactions", "uspto_higher-level.csv")


def resolve_data_file(path: str, subdir_hint: Optional[str] = None) -> str:
    """
    Resolve a possibly relative/moved data file path against known package data directories.

    If ``path`` exists as given, it is returned unchanged.
    Otherwise, this tries the basename under common ``minicasp/data`` subdirectories.
    """
    if os.path.exists(path):
        return path

    filename = os.path.basename(path)
    search_dirs = ["buyables", "reactions", "route_testset", "routes"]
    if subdir_hint in search_dirs:
        search_dirs = [subdir_hint] + [d for d in search_dirs if d != subdir_hint]

    for subdir in search_dirs:
        candidate = os.path.join(data_path(), subdir, filename)
        if os.path.exists(candidate):
            return candidate

    return path
