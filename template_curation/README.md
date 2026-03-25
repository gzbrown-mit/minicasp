# Template Curation

This folder contains the medicinal-chemistry template curation workflow requested for:

- `heterocycle_core`
- `medchem_core`
- the merged union export

## Files

- `annotations.py`: reaction SMARTS structural annotation helpers
- `pipeline.py`: source normalization, filtering, deduplication, and export logic
- `curate_templates.py`: CLI entrypoint
- `tests/test_pipeline.py`: focused structural and export tests
- `output/`: generated audit artifacts and curated template tables

## Usage

```bash
python template_curation/curate_templates.py \
  --mtemplates MTemplates/summary_FC_specific_raw_DH.parquet \
  --aizynth data/templates/full_uspto_03_05_19_unique_templates.hdf5 \
  --output-dir template_curation/output
```

## Primary outputs

- `heterocycle_core_templates.csv.gz`
- `medchem_core_templates.csv.gz`
- `medchem_union_templates.csv.gz`

## Audit outputs

- `full_template_annotations.parquet`
- `full_template_annotations.csv.gz`
- `deduplicated_template_annotations.parquet`
- `duplicate_resolution.csv`
- `summary_report.json`
- `summary_report.md`

