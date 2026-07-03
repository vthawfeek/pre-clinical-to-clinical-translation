"""Day 22 tests: BRAF/vemurafenib case-study data assembly."""

import json

import numpy as np
import pandas as pd
import pytest

from pctrans.casestudy.braf_vemurafenib import (
    assemble_braf_table,
    classify_braf_status,
    coverage_summary,
    is_braf_v600,
    load_ccle_depmap_id_map,
    load_vemurafenib_sensitivity,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_braf_status_parsing():
    assert classify_braf_status([{"proteinChange": "V600E", "mutationType": "Missense_Mutation"}]) == "mutant"
    assert classify_braf_status([{"proteinChange": "V600K", "mutationType": "Missense_Mutation"}]) == "mutant"
    # A silent hit at the hotspot codon (reference residue recurring) is not a substitution.
    assert classify_braf_status([{"proteinChange": "V600V", "mutationType": "Silent"}]) == "WT"
    # A non-V600 BRAF mutation elsewhere in the gene is also WT for this case study's purposes.
    assert classify_braf_status([{"proteinChange": "L597R", "mutationType": "Missense_Mutation"}]) == "WT"
    # No mutation records at all (sequenced, clean) -> WT.
    assert classify_braf_status([]) == "WT"
    # A real V600E among other unrelated hits still resolves to mutant.
    assert (
        classify_braf_status(
            [
                {"proteinChange": "L597R", "mutationType": "Missense_Mutation"},
                {"proteinChange": "V600E", "mutationType": "Missense_Mutation"},
            ]
        )
        == "mutant"
    )


def test_is_braf_v600_rejects_silent_and_non_v600():
    assert is_braf_v600("V600E") is True
    assert is_braf_v600("V600E", mutation_type="Silent") is False
    assert is_braf_v600("V600V") is False
    assert is_braf_v600("D594G") is False
    assert is_braf_v600(None) is False


def test_load_vemurafenib_sensitivity_aggregates_replicates(tmp_path):
    path = tmp_path / "dose_response.csv"
    pd.DataFrame(
        {
            "depmap_id": ["ACH-1", "ACH-1", "ACH-2", "ACH-3"],
            "broad_id": [
                "BRD-K56343971-001-14-8",
                "BRD-K56343971-001-10-6",
                "BRD-K56343971-001-14-8",
                "BRD-OTHERDRUG-1",
            ],
            "auc": [0.4, 0.6, 0.9, 0.1],
        }
    ).to_csv(path, index=False)

    result = load_vemurafenib_sensitivity(path)
    assert "ACH-3" not in result.index  # not vemurafenib -> excluded
    assert result.loc["ACH-1", "vemurafenib_auc"] == pytest.approx(0.5)  # mean of 0.4/0.6
    assert result.loc["ACH-2", "vemurafenib_auc"] == pytest.approx(0.9)
    assert set(result.columns) == {"vemurafenib_auc", "vemurafenib_auc_z"}


def test_load_ccle_depmap_id_map(tmp_path):
    path = tmp_path / "depmapid.json"
    _write_json(path, [{"sampleId": "A375_SKIN", "value": "ACH-000219"}])
    assert load_ccle_depmap_id_map(path) == {"A375_SKIN": "ACH-000219"}


@pytest.fixture
def braf_table_inputs(tmp_path):
    """Small end-to-end fixture: 3 CCLE SKCM lines (+1 other-lineage), 2 TCGA SKCM patients."""
    ccle_embeddings = tmp_path / "ccle_embeddings.npz"
    np.savez(
        ccle_embeddings,
        z=np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [9.0, 9.0]]),
        y=np.array([2, 2, 2, 0]),  # 2 = SKCM, last row is a different lineage
        ids=np.array(["ACH-MUT", "ACH-WT", "ACH-NOSTATUS", "ACH-OTHERLINEAGE"]),
    )

    tcga_embeddings = tmp_path / "embeddings_test.npz"
    np.savez(
        tcga_embeddings,
        z_tcga=np.array([[1.0, 0.1], [0.1, 1.0], [8.0, 8.0]]),
        y_tcga=np.array([2, 2, 0]),
        ids_tcga=np.array(["TCGA-AA-0001-01", "TCGA-AA-0002-01", "TCGA-AA-0003-01"]),
    )

    ccle_mutations = tmp_path / "ccle_mutations.json"
    _write_json(
        ccle_mutations,
        [{"sampleId": "MUT_SKIN", "proteinChange": "V600E", "mutationType": "Missense_Mutation"}],
    )
    ccle_sequenced = tmp_path / "ccle_sequenced.json"
    _write_json(ccle_sequenced, ["MUT_SKIN", "WT_SKIN"])  # NOSTATUS line deliberately absent
    ccle_depmap_map = tmp_path / "ccle_depmapid.json"
    _write_json(
        ccle_depmap_map,
        [
            {"sampleId": "MUT_SKIN", "value": "ACH-MUT"},
            {"sampleId": "WT_SKIN", "value": "ACH-WT"},
        ],
    )

    tcga_mutations = tmp_path / "tcga_mutations.json"
    _write_json(
        tcga_mutations,
        [{"sampleId": "TCGA-AA-0001-01", "proteinChange": "V600K", "mutationType": "Missense_Mutation"}],
    )
    tcga_sequenced = tmp_path / "tcga_sequenced.json"
    _write_json(tcga_sequenced, ["TCGA-AA-0001-01", "TCGA-AA-0002-01"])

    vemurafenib = tmp_path / "dose_response.csv"
    pd.DataFrame(
        {
            "depmap_id": ["ACH-MUT", "ACH-WT"],
            "broad_id": ["BRD-K56343971-001-14-8", "BRD-K56343971-001-14-8"],
            "auc": [0.3, 0.95],
        }
    ).to_csv(vemurafenib, index=False)

    return {
        "ccle_embeddings_path": ccle_embeddings,
        "tcga_embeddings_path": tcga_embeddings,
        "ccle_mutations_path": ccle_mutations,
        "ccle_sequenced_ids_path": ccle_sequenced,
        "ccle_depmap_id_map_path": ccle_depmap_map,
        "tcga_mutations_path": tcga_mutations,
        "tcga_sequenced_ids_path": tcga_sequenced,
        "vemurafenib_path": vemurafenib,
    }


def test_assemble_table_has_required_cols(braf_table_inputs):
    table = assemble_braf_table(**braf_table_inputs)

    required = {"sample_id", "domain", "lineage", "BRAF_status", "vemurafenib_auc", "embedding"}
    assert required.issubset(table.columns)
    assert set(table["domain"]) == {"cell_line", "patient"}
    assert (table["lineage"] == "SKCM").all()

    # ACH-OTHERLINEAGE and ACH-NOSTATUS (no BRAF call) must both be dropped.
    assert set(table.loc[table["domain"] == "cell_line", "sample_id"]) == {"ACH-MUT", "ACH-WT"}
    assert set(table.loc[table["domain"] == "patient", "sample_id"]) == {"TCGA-AA-0001-01", "TCGA-AA-0002-01"}

    mut_row = table[table["sample_id"] == "ACH-MUT"].iloc[0]
    assert mut_row["BRAF_status"] == "mutant"
    assert mut_row["vemurafenib_auc"] == pytest.approx(0.3)
    assert isinstance(mut_row["embedding"], np.ndarray)
    assert mut_row["embedding"].shape == (2,)

    patient_row = table[table["sample_id"] == "TCGA-AA-0002-01"].iloc[0]
    assert patient_row["BRAF_status"] == "WT"
    assert np.isnan(patient_row["vemurafenib_auc"])


def test_coverage_summary(braf_table_inputs):
    table = assemble_braf_table(**braf_table_inputs)
    summary = coverage_summary(table)
    assert summary["n_cell_lines"] == 2
    assert summary["n_patients"] == 2
    assert summary["n_cell_lines_with_vemurafenib"] == 2
    assert summary["cell_line_braf_split"] == {"mutant": 1, "WT": 1}
    assert summary["patient_braf_split"] == {"mutant": 1, "WT": 1}
