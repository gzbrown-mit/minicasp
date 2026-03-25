"""Tests for template curation structural flags and export logic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from template_curation.annotations import analyze_template_smarts
from template_curation.pipeline import apply_selection_flags, build_merged_template_table, write_outputs


THIAZOLE_FORMATION = (
    "([Br;D1;H0;+0]-[C;H2;D2;+0:1]-[C;D3;H0;+0:2]=[O;D1;H0;+0])."
    "([S;D1;H0;+0:3]=[C;D3;H0;+0:4]-[N;H2;D1;+0:5])>>"
    "([s;H0;D2;+0:3]1(:[c;D3;H0;+0:4]:[n;H0;D2;+0:5]:[c;D3;H0;+0:2]:[c;H1;D2;+0:1]:1))"
)
PYRIDINE_AMIDE = (
    "([n;H0;D2;+0:1]1:[c;H1;D2;+0:2]:[c;H1;D2;+0:3]:[c;H1;D2;+0:4]:[c;H1;D2;+0:5]:[c;D3;H0;+0:6]:1"
    "-[C;D3;H0;+0:7](=[O;D1;H0;+0:8])-[N;H1;D2;+0:9]-[C;H3;D1;+0:10])>>"
    "([n;H0;D2;+0:1]1:[c;H1;D2;+0:2]:[c;H1;D2;+0:3]:[c;H1;D2;+0:4]:[c;H1;D2;+0:5]:[c;D3;H0;+0:6]:1"
    "-[C;D3;H0;+0:7](=[O;D1;H0;+0:8])-[Cl;D1;H0;+0]).([N;H2;D1;+0:9]-[C;H3;D1;+0:10])"
)
HETEROARYL_SUBSTITUTION = (
    "([n;H0;D2;+0:1]1:[c;H1;D2;+0:2]:[c;H1;D2;+0:3]:[c;H1;D2;+0:4]:[c;H1;D2;+0:5]:[c;D3;H0;+0:6]:1-[C;H3;D1;+0:7])>>"
    "([n;H0;D2;+0:1]1:[c;H1;D2;+0:2]:[c;H1;D2;+0:3]:[c;H1;D2;+0:4]:[c;H1;D2;+0:5]:[c;D3;H0;+0:6]:1-[Br;D1;H0;+0])."
    "([C;H3;D1;+0:7]-[B;D3;+0])"
)


class TemplateAnnotationTests(unittest.TestCase):
    def test_heterocycle_formation_is_detected(self) -> None:
        annotation = analyze_template_smarts(THIAZOLE_FORMATION)
        self.assertTrue(annotation["forms_heterocycle"])
        self.assertTrue(annotation["acts_on_heterocycle"])

    def test_remote_heterocycle_is_not_flagged(self) -> None:
        annotation = analyze_template_smarts(PYRIDINE_AMIDE)
        self.assertFalse(annotation["forms_heterocycle"])
        self.assertFalse(annotation["acts_on_heterocycle"])

    def test_reaction_on_heteroaryl_ring_is_detected(self) -> None:
        annotation = analyze_template_smarts(HETEROARYL_SUBSTITUTION)
        self.assertFalse(annotation["forms_heterocycle"])
        self.assertTrue(annotation["acts_on_heterocycle"])


class PipelineTests(unittest.TestCase):
    def test_duplicate_templates_merge_across_sources(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "retro_template": THIAZOLE_FORMATION,
                    "source": "mtemplates",
                    "source_template_id": "m1",
                    "template_hash": pd.NA,
                    "classification": pd.NA,
                    "library_occurence": pd.NA,
                    "support_primary": 10.0,
                    "support_secondary": 20.0,
                    "support_tertiary": 30.0,
                    "number_of_complete_templates": 20,
                    "num_patents": 10,
                    "total_pathways": 30,
                    "forms_heterocycle": True,
                    "acts_on_heterocycle": True,
                    "tier_candidate": "heterocycle_forming_candidate",
                    "annotation_status": "ok",
                },
                {
                    "retro_template": THIAZOLE_FORMATION,
                    "source": "aizynth",
                    "source_template_id": "a1",
                    "template_hash": "hash1",
                    "classification": "N-containing heterocycle formation",
                    "library_occurence": 12,
                    "support_primary": 12.0,
                    "support_secondary": 12.0,
                    "support_tertiary": 0.0,
                    "number_of_complete_templates": pd.NA,
                    "num_patents": pd.NA,
                    "total_pathways": pd.NA,
                    "forms_heterocycle": True,
                    "acts_on_heterocycle": True,
                    "tier_candidate": "heterocycle_forming_candidate",
                    "annotation_status": "ok",
                },
            ]
        )
        df = apply_selection_flags(df)
        merged = build_merged_template_table(df)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.loc[0, "source"], "mtemplates|aizynth")
        self.assertTrue(merged.loc[0, "keep_heterocycle_core"])

    def test_export_contains_retro_template(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "retro_template": THIAZOLE_FORMATION,
                    "source": "mtemplates",
                    "source_template_id": "m1",
                    "template_hash": pd.NA,
                    "classification": pd.NA,
                    "library_occurence": pd.NA,
                    "support_primary": 10.0,
                    "support_secondary": 20.0,
                    "support_tertiary": 30.0,
                    "number_of_complete_templates": 20,
                    "num_patents": 10,
                    "total_pathways": 30,
                    "forms_heterocycle": True,
                    "acts_on_heterocycle": True,
                    "tier_candidate": "heterocycle_forming_candidate",
                    "annotation_status": "ok",
                }
            ]
        )
        df = apply_selection_flags(df)
        merged = build_merged_template_table(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_outputs(
                output_dir=output_dir,
                source_templates=df,
                merged_templates=merged,
                duplicate_resolution=pd.DataFrame(),
                heterocycle_core=merged,
                medchem_core=merged,
                medchem_union=merged,
                summary={"source_templates": {}, "merged_templates": {}},
            )
            exported = pd.read_csv(output_dir / "heterocycle_core_templates.csv.gz", sep="\t", index_col=0)
            self.assertIn("retro_template", exported.columns)


if __name__ == "__main__":
    unittest.main()
