#!/usr/bin/env python
"""
Download AiZynthFinder's pre-trained USPTO models and data.

Usage:
    python scripts/download_model.py [--output_dir minicasp/data/models]

Downloads:
    - USPTO expansion policy model (ONNX)
    - USPTO templates (CSV.gz)
    - ZINC stock (HDF5) — skipped if already present
    - USPTO filter model (ONNX) — optional
"""
from __future__ import annotations

import argparse
import os
import sys

import requests

FILES = {
    "uspto_model.onnx": {
        "url": "https://zenodo.org/record/7797465/files/uspto_model.onnx",
        "description": "USPTO expansion policy model",
        "required": True,
    },
    "uspto_templates.csv.gz": {
        "url": "https://zenodo.org/record/7341155/files/uspto_unique_templates.csv.gz",
        "description": "USPTO reaction templates",
        "required": True,
    },
    "zinc_stock.hdf5": {
        "url": "https://ndownloader.figshare.com/files/23086469",
        "description": "ZINC stock compounds",
        "required": True,
    },
    "uspto_filter_model.onnx": {
        "url": "https://zenodo.org/record/7797465/files/uspto_filter_model.onnx",
        "description": "USPTO filter policy model (optional)",
        "required": False,
    },
}


def download_file(url: str, dest: str, description: str) -> None:
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return

    print(f"  Downloading {description}...")
    print(f"    {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(dest + ".tmp", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                mb = downloaded / 1024 / 1024
                print(f"\r    {mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    os.rename(dest + ".tmp", dest)
    size_mb = os.path.getsize(dest) / 1024 / 1024
    print(f"\r    Done: {size_mb:.1f} MB -> {dest}")


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained AiZynthFinder models")
    parser.add_argument(
        "--output_dir",
        default="minicasp/data/models",
        help="Directory to save downloaded files (default: minicasp/data/models)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download optional files too (filter model)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Download directory: {os.path.abspath(args.output_dir)}\n")

    for filename, info in FILES.items():
        if not info["required"] and not args.all:
            continue
        dest = os.path.join(args.output_dir, filename)
        try:
            download_file(info["url"], dest, info["description"])
        except Exception as e:
            print(f"  ERROR downloading {filename}: {e}", file=sys.stderr)
            if info["required"]:
                sys.exit(1)

    print("\nDownload complete.")
    print(f"\nTo use these models, set environment variables:")
    print(f"  export MINICASP_MODEL={os.path.abspath(os.path.join(args.output_dir, 'uspto_model.onnx'))}")
    print(f"  export MINICASP_TEMPLATES={os.path.abspath(os.path.join(args.output_dir, 'uspto_templates.csv.gz'))}")
    print(f"  export MINICASP_STOCK={os.path.abspath(os.path.join(args.output_dir, 'zinc_stock.hdf5'))}")
    print(f"\nOr update configs/mcts_config.yml with the paths directly.")


if __name__ == "__main__":
    main()
