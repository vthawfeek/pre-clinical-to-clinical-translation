from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pctrans.data.ccle_client import filter_lineages
from pctrans.data.dataset import LINEAGE_TO_IDX, CCLEDataset, TCGADataset
from pctrans.data.preprocessor import DataSplitter, FeatureSynchroniser
from pctrans.data.sampler import StratifiedContrastiveBatchSampler
from pctrans.data.tcga_client import filter_tcga_lineages


def test_tiny_ccle_shape(tiny_ccle):
    assert tiny_ccle.shape == (10, 51)


def test_ccle_client_filter_lineages(tiny_ccle_meta):
    filtered = filter_lineages(tiny_ccle_meta, ["LUAD", "BRCA", "SKCM"])
    assert filtered.sum() > 0


def test_ccle_client_filter_lineages_excludes_other_diseases(tiny_ccle_meta):
    filtered = filter_lineages(tiny_ccle_meta, ["LUAD", "BRCA", "SKCM"])
    # Glioblastoma (row 3) and Lung Squamous Cell Carcinoma (row 5) are not LUAD/BRCA/SKCM
    assert filtered.tolist() == [True, True, True, False, True, False]


@pytest.mark.integration
def test_ccle_expression_no_nan():
    path = Path("data/raw/ccle/OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    if not path.exists():
        pytest.skip("CCLE expression matrix not downloaded")
    head = pd.read_csv(path, index_col=0, nrows=10)
    assert not head.isna().any().any()


def test_tcga_client_filter_lineages(tiny_tcga_meta):
    filtered = filter_tcga_lineages(tiny_tcga_meta, ["LUAD", "BRCA", "SKCM"])
    assert filtered.tolist() == [True, True, True, False, True, False]


def test_tcga_client_filter_lineages_matched_labels(tiny_tcga_meta):
    # PLAN.md's test asserts the *matched* labels are a subset of {LUAD, BRCA, SKCM};
    # filter_tcga_lineages returns the boolean mask (consistent with filter_lineages
    # above), so apply it before checking the label set.
    filtered = filter_tcga_lineages(tiny_tcga_meta, ["LUAD", "BRCA", "SKCM"])
    matched = tiny_tcga_meta.loc[filtered, "cancer type abbreviation"]
    assert set(matched.unique()).issubset({"LUAD", "BRCA", "SKCM"})


@pytest.mark.integration
def test_tcga_expression_gene_count():
    path = Path("data/raw/tcga/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena")
    if not path.exists():
        pytest.skip("TCGA expression matrix not downloaded")
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    assert len(header) > 1000


@pytest.mark.integration
def test_tcga_phenotype_has_target_lineages():
    path = Path("data/raw/tcga/Survival_SupplementalTable_S1_20171025_xena_sp.tsv")
    if not path.exists():
        pytest.skip("TCGA phenotype table not downloaded")
    pheno = pd.read_csv(path, sep="\t")
    filtered = filter_tcga_lineages(pheno, ["LUAD", "BRCA", "SKCM"])
    assert filtered.sum() > 0


def test_find_common_genes_sorted():
    fs = FeatureSynchroniser()
    common = fs.find_common_genes(["B", "A", "C"], ["C", "B", "D"])
    assert common == ["B", "C"]


def _common_genes(fs, tiny_ccle, tiny_tcga):
    ccle_genes = [c for c in tiny_ccle.columns if c != "lineage"]
    tcga_genes = [c for c in tiny_tcga.columns if c != "lineage"]
    return fs.find_common_genes(ccle_genes, tcga_genes)


def test_hvg_selection_count(tiny_ccle, tiny_tcga):
    fs = FeatureSynchroniser()
    common_genes = _common_genes(fs, tiny_ccle, tiny_tcga)
    hvgs = fs.select_hvgs(tiny_ccle, tiny_tcga, common_genes, n_hvgs=10)
    assert len(hvgs) == 10


def test_hvg_selection_deterministic(tiny_ccle, tiny_tcga):
    # HVG variance is computed on ALL samples (no split yet) — this is correct,
    # since the split only needs to happen before fitting scalers (Day 5).
    fs = FeatureSynchroniser()
    common_genes = _common_genes(fs, tiny_ccle, tiny_tcga)
    hvgs_1 = fs.select_hvgs(tiny_ccle, tiny_tcga, common_genes, n_hvgs=10)
    hvgs_2 = fs.select_hvgs(tiny_ccle, tiny_tcga, common_genes, n_hvgs=10)
    assert hvgs_1 == hvgs_2


def test_hvg_tie_break_is_alphabetical():
    # GENEA and GENEB have identical variance in both domains, so mean_rank ties;
    # the deterministic tie-break must prefer the alphabetically-first symbol.
    ccle = pd.DataFrame({"GENEB": [1.0, 2.0, 3.0], "GENEA": [1.0, 2.0, 3.0]})
    tcga = pd.DataFrame({"GENEB": [4.0, 5.0, 6.0], "GENEA": [4.0, 5.0, 6.0]})
    fs = FeatureSynchroniser()
    hvgs = fs.select_hvgs(ccle, tcga, ["GENEA", "GENEB"], n_hvgs=1)
    assert hvgs == ["GENEA"]


# --- Day 5: splits, scalers, datasets, sampler ---


def test_no_data_leakage(tiny_ccle, tiny_tcga):
    # train/val/test IDs must be pairwise disjoint within each domain.
    splitter = DataSplitter()
    splits = splitter.stratified_split(tiny_ccle, tiny_tcga)
    for domain in ("ccle", "tcga"):
        train = set(splits[domain]["train"])
        val = set(splits[domain]["val"])
        test = set(splits[domain]["test"])
        assert train & val == set()
        assert train & test == set()
        assert val & test == set()


def test_split_covers_all_samples(tiny_ccle, tiny_tcga):
    # The split must partition every sample exactly once (no dropped/duplicated IDs).
    splitter = DataSplitter()
    splits = splitter.stratified_split(tiny_ccle, tiny_tcga)
    for domain, df in (("ccle", tiny_ccle), ("tcga", tiny_tcga)):
        allocated = (
            splits[domain]["train"] + splits[domain]["val"] + splits[domain]["test"]
        )
        assert sorted(allocated) == sorted(df.index.tolist())


def test_stratified_split_preserves_lineages(tiny_ccle, tiny_tcga):
    # Every lineage present in a domain must survive into the train split.
    splitter = DataSplitter()
    splits = splitter.stratified_split(tiny_ccle, tiny_tcga)
    train_lineages = set(tiny_tcga.loc[splits["tcga"]["train"], "lineage"])
    assert train_lineages == set(tiny_tcga["lineage"].unique())


def test_scaler_fit_on_train_only(tiny_ccle, tiny_tcga):
    # Fit on train pool only; the fitted mean must equal the train-pool mean and
    # differ from the train+val mean — proving val/test were excluded.
    splitter = DataSplitter()
    splits = splitter.stratified_split(tiny_ccle, tiny_tcga)
    ccle_train = tiny_ccle.loc[splits["ccle"]["train"]]
    tcga_train = tiny_tcga.loc[splits["tcga"]["train"]]
    tcga_val = tiny_tcga.loc[splits["tcga"]["val"]]

    scalers = splitter.fit_scalers(ccle_train, tcga_train)
    feature_cols = scalers["feature_cols"]

    train_mean = pd.concat(
        [ccle_train[feature_cols], tcga_train[feature_cols]], axis=0
    ).mean().to_numpy()
    assert np.allclose(scalers["scaler"].mean_, train_mean)

    train_plus_val_mean = pd.concat(
        [ccle_train[feature_cols], tcga_train[feature_cols], tcga_val[feature_cols]],
        axis=0,
    ).mean().to_numpy()
    assert not np.allclose(scalers["scaler"].mean_, train_plus_val_mean)


def test_apply_scalers_zscores_train(tiny_ccle, tiny_tcga):
    # After transforming the pooled train frame, each gene column is ~zero-mean.
    splitter = DataSplitter()
    splits = splitter.stratified_split(tiny_ccle, tiny_tcga)
    ccle_train = tiny_ccle.loc[splits["ccle"]["train"]]
    tcga_train = tiny_tcga.loc[splits["tcga"]["train"]]
    scalers = splitter.fit_scalers(ccle_train, tcga_train)

    scaled = splitter.apply_scalers(pd.concat([ccle_train, tcga_train], axis=0), scalers)
    assert "lineage" in scaled.columns
    gene_means = scaled[scalers["feature_cols"]].mean().to_numpy()
    assert np.allclose(gene_means, 0.0, atol=1e-9)


def test_save_splits_roundtrip(tiny_ccle, tiny_tcga, tmp_path):
    import json
    import pickle

    splitter = DataSplitter()
    splits = splitter.stratified_split(tiny_ccle, tiny_tcga)
    ccle_train = tiny_ccle.loc[splits["ccle"]["train"]]
    tcga_train = tiny_tcga.loc[splits["tcga"]["train"]]
    scalers = splitter.fit_scalers(ccle_train, tcga_train)

    splitter.save_splits(splits, scalers, tmp_path)
    with open(tmp_path / "splits.json", encoding="utf-8") as f:
        loaded_splits = json.load(f)
    with open(tmp_path / "scalers.pkl", "rb") as f:
        loaded_scalers = pickle.load(f)

    assert loaded_splits["ccle"]["train"] == splits["ccle"]["train"]
    assert np.allclose(loaded_scalers["scaler"].mean_, scalers["scaler"].mean_)


def test_dataset_getitem_shape_and_label(tiny_ccle):
    ds = CCLEDataset(tiny_ccle, lineage_col="lineage")
    features, label = ds[0]
    assert features.shape == (50,)  # 51 columns minus the lineage column
    assert len(ds) == len(tiny_ccle)
    assert label == LINEAGE_TO_IDX[tiny_ccle["lineage"].iloc[0]]


def test_dataset_rejects_unknown_lineage(tiny_ccle):
    bad = tiny_ccle.copy()
    bad.loc[bad.index[0], "lineage"] = "GBM"
    with pytest.raises(ValueError, match="unknown lineage"):
        CCLEDataset(bad, lineage_col="lineage")


def test_stratified_sampler_lineage_balance(tiny_ccle, tiny_tcga):
    ccle_dataset = CCLEDataset(tiny_ccle, lineage_col="lineage")
    tcga_dataset = TCGADataset(tiny_tcga, lineage_col="lineage")
    # batch_size=6 → per_lineage=1 across 3 lineages x 2 domains.
    sampler = StratifiedContrastiveBatchSampler(ccle_dataset, tcga_dataset, batch_size=6)
    batch = next(iter(sampler))

    ccle_labels = [ccle_dataset[i][1] for i in batch["ccle_indices"]]
    tcga_labels = [tcga_dataset[i][1] for i in batch["tcga_indices"]]
    assert len(set(ccle_labels)) == 3  # all 3 lineages present (CCLE)
    assert len(set(tcga_labels)) == 3  # all 3 lineages present (TCGA)


def test_sampler_tcga_no_replacement_within_epoch(tiny_ccle, tiny_tcga):
    # TCGA is drawn without replacement within an epoch: no index repeats.
    ccle_dataset = CCLEDataset(tiny_ccle, lineage_col="lineage")
    tcga_dataset = TCGADataset(tiny_tcga, lineage_col="lineage")
    sampler = StratifiedContrastiveBatchSampler(ccle_dataset, tcga_dataset, batch_size=6)
    seen = [i for batch in sampler for i in batch["tcga_indices"]]
    assert len(seen) == len(set(seen))


def test_sampler_reshuffles_between_epochs(tiny_ccle, tiny_tcga):
    ccle_dataset = CCLEDataset(tiny_ccle, lineage_col="lineage")
    tcga_dataset = TCGADataset(tiny_tcga, lineage_col="lineage")
    sampler = StratifiedContrastiveBatchSampler(ccle_dataset, tcga_dataset, batch_size=6)
    epoch0 = [batch["tcga_indices"] for batch in sampler]
    epoch1 = [batch["tcga_indices"] for batch in sampler]
    assert epoch0 != epoch1  # different shuffle each epoch
