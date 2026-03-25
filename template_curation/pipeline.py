"""Pipeline for curating medicinal chemistry focused template libraries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from template_curation.annotations import annotate_templates, normalize_smarts

HETEROCYCLE_MTPL_PATENT_FLOOR = 5
HETEROCYCLE_MTPL_COMPLETE_FLOOR = 10
HETEROCYCLE_AIZYNTH_FLOOR = 5
MEDCHEM_MTPL_PATENT_FLOOR = 3
MEDCHEM_AIZYNTH_FLOOR = 3
MEDCHEM_SUPPORT_COVERAGE = 0.90

NORMALIZED_COLUMNS = [
    "retro_template",
    "source",
    "source_template_id",
    "template_hash",
    "classification",
    "library_occurence",
    "support_primary",
    "support_secondary",
    "support_tertiary",
    "number_of_complete_templates",
    "num_patents",
    "total_pathways",
]

MERGED_COLUMNS = [
    "retro_template",
    "source",
    "source_template_id",
    "template_hash",
    "classification",
    "library_occurence",
    "source_record_count",
    "support_primary",
    "support_secondary",
    "support_tertiary",
    "support_primary_sum",
    "number_of_complete_templates",
    "num_patents",
    "total_pathways",
    "forms_heterocycle",
    "acts_on_heterocycle",
    "tier_candidate",
    "annotation_status",
    "keep_heterocycle_core",
    "keep_medchem_core",
    "keep_reason",
    "source_support_details",
]


def curate_template_libraries(
    mtemplates_path: str | Path,
    aizynth_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, pd.DataFrame]:
    """Load, annotate, curate, deduplicate, and export template libraries."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading source templates...")
    source_templates = load_source_templates(mtemplates_path, aizynth_path)
    print(f"Loaded {len(source_templates)} source template rows")
    print("Annotating structural template flags...")
    source_templates = annotate_source_templates(source_templates)
    print("Applying selection flags...")
    source_templates = apply_selection_flags(source_templates)

    print("Building deduplicated template tables...")
    merged_templates = build_merged_template_table(source_templates)
    duplicate_resolution = build_duplicate_resolution_table(source_templates)
    summary = build_summary(source_templates, merged_templates)

    heterocycle_core = merged_templates[merged_templates["keep_heterocycle_core"]].copy()
    medchem_core = merged_templates[merged_templates["keep_medchem_core"]].copy()
    medchem_union = merged_templates[
        merged_templates["keep_heterocycle_core"] | merged_templates["keep_medchem_core"]
    ].copy()

    print("Writing audit artifacts and curated exports...")
    write_outputs(
        output_path,
        source_templates,
        merged_templates,
        duplicate_resolution,
        heterocycle_core,
        medchem_core,
        medchem_union,
        summary,
    )

    return {
        "source_templates": source_templates,
        "merged_templates": merged_templates,
        "duplicate_resolution": duplicate_resolution,
        "heterocycle_core": heterocycle_core,
        "medchem_core": medchem_core,
        "medchem_union": medchem_union,
    }


def load_source_templates(
    mtemplates_path: str | Path,
    aizynth_path: str | Path,
) -> pd.DataFrame:
    """Load and normalize MTemplates and AiZynth inputs into a common schema."""

    mtemplates = _load_mtemplates(mtemplates_path)
    aizynth = _load_aizynth(aizynth_path)
    return pd.concat([mtemplates, aizynth], ignore_index=True, sort=False)


def annotate_source_templates(source_templates: pd.DataFrame) -> pd.DataFrame:
    """Annotate exact SMARTS strings once, then fan annotations back to source rows."""

    annotated = source_templates.copy()
    structural_candidate_mask = (
        (
            (annotated["source"] == "mtemplates")
            & (annotated["num_patents"].fillna(0) >= MEDCHEM_MTPL_PATENT_FLOOR)
        )
        | (
            (annotated["source"] == "aizynth")
            & (annotated["library_occurence"].fillna(0) >= MEDCHEM_AIZYNTH_FLOOR)
        )
    )
    annotated["forms_heterocycle"] = False
    annotated["acts_on_heterocycle"] = False
    annotated["tier_candidate"] = "other"
    annotated["annotation_status"] = "skipped_low_support"

    candidate_templates = (
        annotated.loc[structural_candidate_mask, "retro_template"].drop_duplicates().tolist()
    )
    annotations = annotate_templates(candidate_templates)
    annotation_df = pd.DataFrame.from_dict(annotations, orient="index").reset_index()
    annotation_df = annotation_df.rename(columns={"index": "retro_template"})

    if annotation_df.empty:
        return annotated

    annotated = annotated.merge(
        annotation_df,
        on="retro_template",
        how="left",
        suffixes=("", "_annotated"),
    )
    for column in ["forms_heterocycle", "acts_on_heterocycle", "tier_candidate", "annotation_status"]:
        annotated[column] = annotated[f"{column}_annotated"].combine_first(annotated[column])
        del annotated[f"{column}_annotated"]
    return annotated


def apply_selection_flags(source_templates: pd.DataFrame) -> pd.DataFrame:
    """Apply heterocycle-core and popularity-driven medchem-core selection flags."""

    df = source_templates.copy()
    df["forms_heterocycle"] = df["forms_heterocycle"].fillna(False).astype(bool)
    df["acts_on_heterocycle"] = df["acts_on_heterocycle"].fillna(False).astype(bool)
    df["annotation_status"] = df["annotation_status"].fillna("missing")
    df["keep_heterocycle_core"] = False
    df["keep_medchem_core"] = False
    df["keep_reason"] = ""

    heterocycle_candidate = df["forms_heterocycle"] | df["acts_on_heterocycle"]
    mtemplate_heterocycle_mask = (
        (df["source"] == "mtemplates")
        & heterocycle_candidate
        & (df["num_patents"].fillna(0) >= HETEROCYCLE_MTPL_PATENT_FLOOR)
        & (df["number_of_complete_templates"].fillna(0) >= HETEROCYCLE_MTPL_COMPLETE_FLOOR)
    )
    aizynth_heterocycle_mask = (
        (df["source"] == "aizynth")
        & heterocycle_candidate
        & (df["library_occurence"].fillna(0) >= HETEROCYCLE_AIZYNTH_FLOOR)
    )
    df.loc[mtemplate_heterocycle_mask | aizynth_heterocycle_mask, "keep_heterocycle_core"] = True

    medchem_mask = _select_medchem_core_mask(df)
    df.loc[medchem_mask | df["keep_heterocycle_core"], "keep_medchem_core"] = True
    df["tier_candidate"] = np.where(
        df["tier_candidate"].eq("other") & df["keep_medchem_core"],
        "popular_general_medchem_candidate",
        df["tier_candidate"],
    )

    keep_reasons = []
    for row in df.itertuples(index=False):
        reasons = []
        if row.keep_heterocycle_core:
            if row.forms_heterocycle:
                reasons.append("heterocycle_forming")
            elif row.acts_on_heterocycle:
                reasons.append("heterocycle_editing")
        if row.keep_medchem_core and not reasons:
            reasons.append("popular_general_medchem")
        elif row.keep_medchem_core and "popular_general_medchem" not in reasons:
            reasons.append("popular_general_medchem")
        keep_reasons.append(";".join(reasons))
    df["keep_reason"] = keep_reasons

    return df


def build_merged_template_table(source_templates: pd.DataFrame) -> pd.DataFrame:
    """Collapse exact duplicate SMARTS across sources while preserving provenance."""

    duplicate_mask = source_templates["retro_template"].duplicated(keep=False)

    unique_rows = source_templates.loc[~duplicate_mask].copy()
    unique_rows["source_record_count"] = 1
    unique_rows["support_primary_sum"] = unique_rows["support_primary"].fillna(0.0)
    unique_rows["source_support_details"] = [
        json.dumps(
            {
                row.source: {
                    "support_primary": float(row.support_primary or 0.0),
                    "support_secondary": float(row.support_secondary or 0.0),
                    "support_tertiary": float(row.support_tertiary or 0.0),
                }
            },
            sort_keys=True,
        )
        for row in unique_rows.itertuples(index=False)
    ]
    unique_rows = unique_rows.reindex(columns=MERGED_COLUMNS)

    duplicate_rows = source_templates.loc[duplicate_mask]
    if duplicate_rows.empty:
        merged = unique_rows
    else:
        merged_duplicates = (
            duplicate_rows.groupby("retro_template", sort=False, dropna=False)
            .apply(_merge_template_group)
            .reset_index(drop=True)
        )
        merged_duplicates = merged_duplicates.reindex(columns=MERGED_COLUMNS)
        merged = pd.concat([unique_rows, merged_duplicates], ignore_index=True, sort=False)

    merged = merged.sort_values(
        ["keep_medchem_core", "keep_heterocycle_core", "support_primary_sum"],
        ascending=[False, False, False],
    )
    merged = merged.reset_index(drop=True)
    return merged


def build_duplicate_resolution_table(source_templates: pd.DataFrame) -> pd.DataFrame:
    """Create an audit table for exact SMARTS duplicates across inputs."""

    duplicate_mask = source_templates["retro_template"].duplicated(keep=False)
    if not duplicate_mask.any():
        return pd.DataFrame(
            columns=[
                "retro_template",
                "record_count",
                "sources",
                "source_template_ids",
                "keep_heterocycle_core",
                "keep_medchem_core",
                "keep_reason",
            ]
        )

    duplicate_rows = []
    for retro_template, group in source_templates.loc[duplicate_mask].groupby("retro_template", sort=False):
        duplicate_rows.append(
            {
                "retro_template": retro_template,
                "record_count": len(group),
                "sources": "|".join(sorted(group["source"].astype(str).unique())),
                "source_template_ids": "|".join(group["source_template_id"].astype(str).tolist()),
                "keep_heterocycle_core": bool(group["keep_heterocycle_core"].any()),
                "keep_medchem_core": bool(group["keep_medchem_core"].any()),
                "keep_reason": _merge_pipe_values(group["keep_reason"]),
            }
        )
    return pd.DataFrame(duplicate_rows)


def build_summary(source_templates: pd.DataFrame, merged_templates: pd.DataFrame) -> Dict[str, object]:
    """Summarize counts for the generated outputs."""

    summary = {
        "source_templates": {
            source: {
                "rows": int(len(group)),
                "heterocycle_core_rows": int(group["keep_heterocycle_core"].sum()),
                "medchem_core_rows": int(group["keep_medchem_core"].sum()),
                "annotation_errors": int(group["annotation_status"].astype(str).str.startswith("parse_error").sum()),
                "annotation_skipped_low_support": int(group["annotation_status"].eq("skipped_low_support").sum()),
                "keep_reason_counts": group["keep_reason"].replace("", "dropped").value_counts().to_dict(),
            }
            for source, group in source_templates.groupby("source", sort=False)
        },
        "merged_templates": {
            "rows": int(len(merged_templates)),
            "heterocycle_core_rows": int(merged_templates["keep_heterocycle_core"].sum()),
            "medchem_core_rows": int(merged_templates["keep_medchem_core"].sum()),
            "union_rows": int(
                (merged_templates["keep_heterocycle_core"] | merged_templates["keep_medchem_core"]).sum()
            ),
        },
    }
    return summary


def write_outputs(
    output_dir: Path,
    source_templates: pd.DataFrame,
    merged_templates: pd.DataFrame,
    duplicate_resolution: pd.DataFrame,
    heterocycle_core: pd.DataFrame,
    medchem_core: pd.DataFrame,
    medchem_union: pd.DataFrame,
    summary: Dict[str, object],
) -> None:
    """Write audit artifacts and primary template exports."""

    output_dir.mkdir(parents=True, exist_ok=True)

    source_templates.to_parquet(output_dir / "full_template_annotations.parquet", index=False)
    source_templates.to_csv(
        output_dir / "full_template_annotations.csv.gz",
        index=False,
        compression="gzip",
    )
    merged_templates.to_parquet(output_dir / "deduplicated_template_annotations.parquet", index=False)

    duplicate_resolution.to_csv(output_dir / "duplicate_resolution.csv", index=False)

    _write_template_table(source_templates[source_templates["source"] == "mtemplates"], output_dir / "mtemplates_normalized_templates.csv.gz")
    _write_template_table(source_templates[source_templates["source"] == "aizynth"], output_dir / "aizynth_normalized_templates.csv.gz")
    _write_template_table(heterocycle_core, output_dir / "heterocycle_core_templates.csv.gz")
    _write_template_table(medchem_core, output_dir / "medchem_core_templates.csv.gz")
    _write_template_table(medchem_union, output_dir / "medchem_union_templates.csv.gz")

    with open(output_dir / "summary_report.json", "w", encoding="utf-8") as fileobj:
        json.dump(summary, fileobj, indent=2, sort_keys=True)

    with open(output_dir / "summary_report.md", "w", encoding="utf-8") as fileobj:
        fileobj.write("# Template Curation Summary\n\n")
        fileobj.write("## Source-level counts\n")
        for source, values in summary["source_templates"].items():
            fileobj.write(
                f"- {source}: {values['rows']} rows, "
                f"{values['heterocycle_core_rows']} heterocycle-core, "
                f"{values['medchem_core_rows']} medchem-core, "
                f"{values['annotation_errors']} annotation errors, "
                f"{values.get('annotation_skipped_low_support', 0)} skipped at low support\n"
            )
        merged = summary.get("merged_templates", {})
        fileobj.write("\n## Deduplicated counts\n")
        fileobj.write(
            f"- {merged.get('rows', 0)} unique templates, "
            f"{merged.get('heterocycle_core_rows', 0)} heterocycle-core, "
            f"{merged.get('medchem_core_rows', 0)} medchem-core, "
            f"{merged.get('union_rows', 0)} in the union export\n"
        )


def _load_mtemplates(path: str | Path) -> pd.DataFrame:
    columns = [
        "FC_specific_raw_DH",
        "template_smarts",
        "number_of_complete_templates",
        "num_patents",
        "total_pathways",
        "num_rings_created_avg",
        "num_rings_destroyed_avg",
        "num_CX_XX_bonds_avg",
    ]
    df = pd.read_parquet(path, columns=columns)
    df = df.rename(
        columns={
            "template_smarts": "retro_template",
            "FC_specific_raw_DH": "source_template_id",
        }
    )
    df["retro_template"] = df["retro_template"].map(normalize_smarts)
    df["source"] = "mtemplates"
    df["template_hash"] = pd.NA
    df["classification"] = pd.NA
    df["library_occurence"] = pd.NA
    df["support_primary"] = df["num_patents"].fillna(0).astype(float)
    df["support_secondary"] = df["number_of_complete_templates"].fillna(0).astype(float)
    df["support_tertiary"] = df["total_pathways"].fillna(0).astype(float)
    return df[NORMALIZED_COLUMNS + ["num_rings_created_avg", "num_rings_destroyed_avg", "num_CX_XX_bonds_avg"]]


def _load_aizynth(path: str | Path) -> pd.DataFrame:
    df = pd.read_hdf(path, "table")
    df = df.reset_index().rename(columns={"template_code": "source_template_id"})
    df["retro_template"] = df["retro_template"].map(normalize_smarts)
    df["source"] = "aizynth"
    df["support_primary"] = df["library_occurence"].fillna(0).astype(float)
    df["support_secondary"] = df["library_occurence"].fillna(0).astype(float)
    df["support_tertiary"] = 0.0
    df["number_of_complete_templates"] = pd.NA
    df["num_patents"] = pd.NA
    df["total_pathways"] = pd.NA
    return df[NORMALIZED_COLUMNS]


def _select_medchem_core_mask(df: pd.DataFrame) -> pd.Series:
    medchem_mask = pd.Series(False, index=df.index)

    mtemplate_mask = df["source"] == "mtemplates"
    medchem_mask.loc[mtemplate_mask] = _coverage_keep_mask(
        df.loc[mtemplate_mask],
        floor_mask=df.loc[mtemplate_mask, "num_patents"].fillna(0) >= MEDCHEM_MTPL_PATENT_FLOOR,
        sort_columns=["support_primary", "support_secondary", "support_tertiary"],
    )

    aizynth_mask = df["source"] == "aizynth"
    medchem_mask.loc[aizynth_mask] = _coverage_keep_mask(
        df.loc[aizynth_mask],
        floor_mask=df.loc[aizynth_mask, "library_occurence"].fillna(0) >= MEDCHEM_AIZYNTH_FLOOR,
        sort_columns=["support_primary"],
    )

    return medchem_mask


def _coverage_keep_mask(df: pd.DataFrame, floor_mask: pd.Series, sort_columns: list[str]) -> pd.Series:
    keep_mask = pd.Series(False, index=df.index)
    eligible = df.loc[floor_mask].copy()
    if eligible.empty:
        return keep_mask

    eligible = eligible.sort_values(sort_columns, ascending=[False] * len(sort_columns), kind="mergesort")
    total_support = eligible["support_primary"].sum()
    if total_support <= 0:
        return keep_mask

    target_support = total_support * MEDCHEM_SUPPORT_COVERAGE
    cumulative_support = eligible["support_primary"].cumsum()
    cutoff_position = int(np.searchsorted(cumulative_support.to_numpy(), target_support, side="left"))
    keep_indices = eligible.index[: cutoff_position + 1]
    keep_mask.loc[keep_indices] = True
    return keep_mask


def _merge_template_group(group: pd.DataFrame) -> pd.Series:
    forms_heterocycle = bool(group["forms_heterocycle"].any())
    acts_on_heterocycle = bool(group["acts_on_heterocycle"].any())
    keep_heterocycle_core = bool(group["keep_heterocycle_core"].any())
    keep_medchem_core = bool(group["keep_medchem_core"].any())

    if forms_heterocycle:
        tier_candidate = "heterocycle_forming_candidate"
    elif acts_on_heterocycle:
        tier_candidate = "heterocycle_editing_candidate"
    elif keep_medchem_core:
        tier_candidate = "popular_general_medchem_candidate"
    else:
        tier_candidate = "other"

    source_support = {
        source: {
            "support_primary": float(source_group["support_primary"].max()),
            "support_secondary": float(source_group["support_secondary"].max()),
            "support_tertiary": float(source_group["support_tertiary"].max()),
        }
        for source, source_group in group.groupby("source", sort=False)
    }

    return pd.Series(
        {
            "retro_template": group["retro_template"].iloc[0],
            "source": _merge_pipe_values(group["source"]),
            "source_template_id": _merge_pipe_values(group["source_template_id"]),
            "template_hash": _merge_pipe_values(group["template_hash"]),
            "classification": _merge_pipe_values(group["classification"]),
            "library_occurence": group["library_occurence"].max(),
            "source_record_count": int(len(group)),
            "support_primary": float(group["support_primary"].max()),
            "support_secondary": float(group["support_secondary"].max()),
            "support_tertiary": float(group["support_tertiary"].max()),
            "support_primary_sum": float(group["support_primary"].sum()),
            "number_of_complete_templates": group["number_of_complete_templates"].max(),
            "num_patents": group["num_patents"].max(),
            "total_pathways": group["total_pathways"].max(),
            "forms_heterocycle": forms_heterocycle,
            "acts_on_heterocycle": acts_on_heterocycle,
            "tier_candidate": tier_candidate,
            "annotation_status": _merge_pipe_values(group["annotation_status"]),
            "keep_heterocycle_core": keep_heterocycle_core,
            "keep_medchem_core": keep_medchem_core,
            "keep_reason": _merge_pipe_values(group["keep_reason"]),
            "source_support_details": json.dumps(source_support, sort_keys=True),
        }
    )


def _merge_pipe_values(series: pd.Series) -> str:
    values = []
    for value in series.dropna():
        text = str(value).strip()
        if text and text not in values:
            values.append(text)
    return "|".join(values)


def _write_template_table(df: pd.DataFrame, path: Path) -> None:
    export_df = df.copy().reset_index(drop=True)
    export_df.index.name = "template_code"
    export_df.to_csv(path, sep="\t", compression="gzip")
