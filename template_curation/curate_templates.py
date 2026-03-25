#!/usr/bin/env python
"""Curate heterocycle and general med-chem template libraries from local inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from template_curation.pipeline import curate_template_libraries


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate medicinal chemistry template libraries")
    parser.add_argument(
        "--mtemplates",
        default="MTemplates/summary_FC_specific_raw_DH.parquet",
        help="Path to the MTemplates summary parquet file",
    )
    parser.add_argument(
        "--aizynth",
        default="data/templates/full_uspto_03_05_19_unique_templates.hdf5",
        help="Path to the AiZynth template HDF5 file",
    )
    parser.add_argument(
        "--output-dir",
        default="template_curation/output",
        help="Directory for audit artifacts and curated template exports",
    )
    args = parser.parse_args()

    results = curate_template_libraries(
        mtemplates_path=args.mtemplates,
        aizynth_path=args.aizynth,
        output_dir=args.output_dir,
    )

    output_dir = Path(args.output_dir).resolve()
    print(f"Curated template outputs written to {output_dir}")
    print(f"  Source rows: {len(results['source_templates'])}")
    print(f"  Unique rows: {len(results['merged_templates'])}")
    print(f"  Heterocycle core: {len(results['heterocycle_core'])}")
    print(f"  Medchem core: {len(results['medchem_core'])}")
    print(f"  Union: {len(results['medchem_union'])}")


if __name__ == "__main__":
    main()
