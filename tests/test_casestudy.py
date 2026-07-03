"""Day 22 tests: BRAF/vemurafenib case-study data assembly."""

import json

import numpy as np
import pandas as pd
import pytest

from pctrans.casestudy.braf_vemurafenib import (
    assemble_braf_table,
    braf_mutant_patient_centroid,
    braf_placement_test,
    braf_response_link,
    classify_braf_status,
    coverage_summary,
    distance_to_centroid,
    is_braf_v600,
    load_ccle_depmap_id_map,
    load_vemurafenib_sensitivity,
    pairwise_greater_fraction,
    spearman_with_ci,
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


def test_centroid_distance_math():
    # BRAF-mutant patients at (0,0) and (2,0) -> centroid (1,0); a WT patient
    # far away must not pull the centroid.
    patient_embeddings = np.array([[0.0, 0.0], [2.0, 0.0], [100.0, 100.0]])
    patient_status = np.array(["mutant", "mutant", "WT"])
    centroid = braf_mutant_patient_centroid(patient_embeddings, patient_status)
    assert centroid == pytest.approx([1.0, 0.0])

    # 5 mutant cell lines placed near the centroid, 5 WT lines placed far away.
    cell_line_embeddings = np.array(
        [[1.0, 0.1], [0.9, -0.1], [1.1, 0.05], [1.05, -0.05], [0.95, 0.0]]
        + [[10.0, 9.0], [10.5, 9.5], [11.0, 10.0], [9.5, 10.5], [11.5, 9.0]]
    )
    cell_line_status = np.array(["mutant"] * 5 + ["WT"] * 5)
    distances = distance_to_centroid(cell_line_embeddings, centroid)
    assert distances[0] == pytest.approx(np.hypot(0.0, 0.1))
    # Known ordering: every mutant line strictly closer than every WT line.
    assert distances[:5].max() < distances[5:].min()

    # pairwise_greater_fraction: perfect separation -> 1.0; reversed -> 0.0; tie -> 0.5.
    assert pairwise_greater_fraction([5, 6, 7], [1, 2, 3]) == pytest.approx(1.0)
    assert pairwise_greater_fraction([1, 2, 3], [5, 6, 7]) == pytest.approx(0.0)
    assert pairwise_greater_fraction([1, 1], [1, 1]) == pytest.approx(0.5)

    result = braf_placement_test(
        cell_line_embeddings, cell_line_status, patient_embeddings, patient_status, n_boot=200, seed=0
    )
    assert result["n_mutant"] == 5
    assert result["n_wt"] == 5
    assert result["p_value"] < 0.05
    assert result["effect_size"] == pytest.approx(1.0)
    assert 0.0 <= result["effect_size_ci_low"] <= result["effect_size_ci_high"] <= 1.0


def test_placement_test_requires_both_groups():
    patient_embeddings = np.array([[0.0, 0.0], [2.0, 0.0]])
    patient_status = np.array(["mutant", "mutant"])
    cell_line_embeddings = np.array([[1.0, 0.0], [1.1, 0.0]])
    cell_line_status = np.array(["mutant", "mutant"])  # no WT lines at all
    with pytest.raises(ValueError):
        braf_placement_test(cell_line_embeddings, cell_line_status, patient_embeddings, patient_status)


def test_response_correlation_runs_and_bounds():
    rng = np.random.default_rng(0)
    n = 30
    proximity = rng.uniform(0, 1, size=n)
    # Perfect negative relationship + tiny noise: closer (higher proximity) -> lower AUC.
    auc = 1.0 - proximity + rng.normal(0, 1e-6, size=n)

    result = spearman_with_ci(proximity, auc, n_boot=200, seed=1)
    assert set(result) == {"rho", "p_value", "ci_low", "ci_high", "n_boot", "n"}
    assert -1.0 <= result["rho"] <= 1.0
    assert -1.0 <= result["ci_low"] <= result["ci_high"] <= 1.0
    assert result["rho"] == pytest.approx(-1.0, abs=1e-3)
    assert result["n"] == n

    # End-to-end through braf_response_link on synthetic embeddings.
    patient_embeddings = np.array([[0.0, 0.0], [2.0, 0.0]])
    patient_status = np.array(["mutant", "mutant"])  # centroid at (1, 0)
    cell_line_embeddings = np.array([[1.0, 0.1], [3.0, 0.1], [6.0, 0.1], [9.0, 0.1]])
    cell_line_status = np.array(["mutant", "mutant", "WT", "WT"])
    # Sensitivity (low AUC) increases with distance here -- an arbitrary synthetic
    # relationship, this test only checks the function runs and returns valid bounds.
    vemurafenib_auc = np.array([0.9, 0.6, 0.4, 0.1])

    link = braf_response_link(
        cell_line_embeddings, cell_line_status, vemurafenib_auc,
        patient_embeddings, patient_status, n_boot=100, seed=2,
    )
    assert -1.0 <= link["rho"] <= 1.0
    assert -1.0 <= link["ci_low"] <= link["ci_high"] <= 1.0
    assert link["n"] == 4
    assert link["n_mutant"] == 2
    assert link["n_wt"] == 2
