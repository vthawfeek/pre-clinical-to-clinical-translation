from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pctrans.data.ccle_client import (
    METADATA_FILENAME as CCLE_METADATA_FILENAME,
)
from pctrans.data.ccle_client import LINEAGE_ALIASES, CCLEClient, filter_lineages
from pctrans.data.dataset import LINEAGE_TO_IDX, CCLEDataset, TCGADataset, build_lineage_maps
from pctrans.data.preprocessor import DataSplitter, FeatureSynchroniser
from pctrans.data.sampler import StratifiedContrastiveBatchSampler
from pctrans.data.tcga_client import (
    EXPRESSION_FILENAME as TCGA_EXPRESSION_FILENAME,
)
from pctrans.data.tcga_client import (
    PHENOTYPE_FILENAME as TCGA_PHENOTYPE_FILENAME,
)
from pctrans.data.tcga_client import TCGAClient, filter_tcga_lineages


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


# --- Download-client idempotency (skip re-download when the file already exists) ---


def test_ccle_download_idempotent_when_present(tmp_path):
    # An existing file short-circuits the download: the path is returned untouched
    # and no network call is made (force=False).
    dest = tmp_path / CCLE_METADATA_FILENAME
    dest.write_text("ModelID,OncotreePrimaryDisease\nACH-1,Melanoma\n", encoding="utf-8")
    before = dest.read_bytes()
    returned = CCLEClient().download_metadata(tmp_path)
    assert Path(returned) == dest
    assert dest.read_bytes() == before


def test_tcga_download_expression_idempotent_when_present(tmp_path):
    dest = tmp_path / TCGA_EXPRESSION_FILENAME
    dest.write_text("gene\tTCGA-1\nEGFR\t3.14\n", encoding="utf-8")
    returned = TCGAClient().download_expression(tmp_path)
    assert Path(returned) == dest


def test_tcga_download_phenotype_idempotent_when_present(tmp_path):
    dest = tmp_path / TCGA_PHENOTYPE_FILENAME
    dest.write_text("sample\tcancer type abbreviation\nTCGA-1\tLUAD\n", encoding="utf-8")
    returned = TCGAClient().download_phenotype(tmp_path)
    assert Path(returned) == dest


# --- FeatureSynchroniser.save_filtered ---


def test_save_filtered_writes_parquet_and_gene_list(tiny_ccle, tiny_tcga, tmp_path):
    fs = FeatureSynchroniser()
    common = _common_genes(fs, tiny_ccle, tiny_tcga)
    hvgs = fs.select_hvgs(tiny_ccle, tiny_tcga, common, n_hvgs=5)

    fs.save_filtered(tiny_ccle, tiny_tcga, hvgs, tmp_path)

    assert (tmp_path / "ccle_2k.parquet").exists()
    assert (tmp_path / "tcga_2k.parquet").exists()
    gene_list = (tmp_path / "gene_list.txt").read_text(encoding="utf-8").split()
    assert gene_list == hvgs
    loaded = pd.read_parquet(tmp_path / "ccle_2k.parquet")
    assert list(loaded.columns) == [*hvgs, "lineage"]


# --- Day 16: train-only HVG selection (close the leakage path) ---


def test_hvg_train_only_ignores_test(tiny_ccle, tiny_tcga):
    # Selecting with train_ids must match selecting on the train slice alone, and
    # must not change when the held-out (val/test) rows are corrupted -- proving
    # the ranking never looks at them.
    splitter = DataSplitter()
    splits = splitter.stratified_split(tiny_ccle, tiny_tcga)
    train_ids = {"ccle": splits["ccle"]["train"], "tcga": splits["tcga"]["train"]}

    fs = FeatureSynchroniser()
    common = _common_genes(fs, tiny_ccle, tiny_tcga)

    hvgs_via_train_ids = fs.select_hvgs(
        tiny_ccle, tiny_tcga, common, n_hvgs=10, train_ids=train_ids
    )
    hvgs_via_train_slice = fs.select_hvgs(
        tiny_ccle.loc[train_ids["ccle"]], tiny_tcga.loc[train_ids["tcga"]], common, n_hvgs=10
    )
    assert hvgs_via_train_ids == hvgs_via_train_slice

    gene_cols = [c for c in tiny_ccle.columns if c != "lineage"]
    corrupted_ccle = tiny_ccle.copy()
    non_train_ccle = corrupted_ccle.index.difference(train_ids["ccle"])
    corrupted_ccle.loc[non_train_ccle, gene_cols] = 999.0
    corrupted_tcga = tiny_tcga.copy()
    non_train_tcga = corrupted_tcga.index.difference(train_ids["tcga"])
    corrupted_tcga.loc[non_train_tcga, gene_cols] = 999.0

    hvgs_after_corruption = fs.select_hvgs(
        corrupted_ccle, corrupted_tcga, common, n_hvgs=10, train_ids=train_ids
    )
    assert hvgs_after_corruption == hvgs_via_train_ids


@pytest.mark.integration
def test_hvg_flag_reproduces_phase1():
    # --hvg-on all (train_ids=None) must reproduce the committed Day-4 gene list
    # byte-for-byte -- the Day 16 refactor must not change the default path.
    raw_dir = Path("data/raw")
    gene_list_path = Path("data/processed/gene_list.txt")
    if not raw_dir.exists() or not gene_list_path.exists():
        pytest.skip("raw data or committed gene_list.txt not present")

    fs = FeatureSynchroniser()
    ccle_expr, _ = fs.load_ccle(raw_dir)
    tcga_expr, _ = fs.load_tcga(raw_dir)
    ccle_genes = [c for c in ccle_expr.columns if c != "lineage"]
    tcga_genes = [c for c in tcga_expr.columns if c != "lineage"]
    common_genes = fs.find_common_genes(ccle_genes, tcga_genes)
    hvgs = fs.select_hvgs(ccle_expr, tcga_expr, common_genes, n_hvgs=2000)

    expected = gene_list_path.read_text(encoding="utf-8").split()
    assert hvgs == expected


# --- Day 18: config-driven lineages (15-lineage expansion) ---


def test_lineage_map_is_config_driven():
    fifteen = [
        "BLCA", "BRCA", "COAD", "GBM", "HNSC", "KIRC", "LGG", "LIHC",
        "LUAD", "LUSC", "OV", "PAAD", "READ", "SKCM", "STAD",
    ]
    lineage_to_idx, idx_to_lineage = build_lineage_maps(fifteen)
    assert len(lineage_to_idx) == 15
    assert set(lineage_to_idx.values()) == set(range(15))
    assert idx_to_lineage == {idx: lineage for lineage, idx in lineage_to_idx.items()}


def test_build_lineage_maps_reproduces_phase1_default():
    # The module-level default must stay byte-identical to the original hardcoded
    # dict -- existing checkpoints and committed embeddings depend on this order.
    lineage_to_idx, idx_to_lineage = build_lineage_maps(["LUAD", "BRCA", "SKCM"])
    assert lineage_to_idx == {"LUAD": 0, "BRCA": 1, "SKCM": 2}
    assert idx_to_lineage == {0: "LUAD", 1: "BRCA", 2: "SKCM"}
    assert lineage_to_idx == LINEAGE_TO_IDX


def test_dataset_accepts_custom_lineage_map():
    fifteen = [
        "BLCA", "BRCA", "COAD", "GBM", "HNSC", "KIRC", "LGG", "LIHC",
        "LUAD", "LUSC", "OV", "PAAD", "READ", "SKCM", "STAD",
    ]
    lineage_to_idx, _ = build_lineage_maps(fifteen)
    df = pd.DataFrame({"G0": [1.0, 2.0], "lineage": ["LUSC", "READ"]})
    ds = CCLEDataset(df, lineage_to_idx=lineage_to_idx)
    assert ds.labels.tolist() == [lineage_to_idx["LUSC"], lineage_to_idx["READ"]]


def test_ccle_client_resolves_new_day18_lineages():
    meta = pd.DataFrame(
        {
            "ModelID": [f"ACH-{i:06d}" for i in range(6)],
            "OncotreePrimaryDisease": [
                "Non-Small Cell Lung Cancer",
                "Colorectal Adenocarcinoma",
                "Colorectal Adenocarcinoma",
                "Diffuse Glioma",
                "Diffuse Glioma",
                "Pancreatic Adenocarcinoma",
            ],
            "OncotreeSubtype": [
                "Lung Squamous Cell Carcinoma",
                "Colon Adenocarcinoma",
                "Rectal Adenocarcinoma",
                "Glioblastoma",
                "Astrocytoma",
                "Pancreatic Adenocarcinoma",
            ],
        }
    )
    resolved = meta["OncotreePrimaryDisease"].map(LINEAGE_ALIASES)
    resolved = resolved.fillna(meta["OncotreeSubtype"].map(LINEAGE_ALIASES))
    assert resolved.tolist() == ["LUSC", "COAD", "READ", "GBM", "LGG", "PAAD"]


def test_drop_incomplete_genes_excludes_any_nan():
    fs = FeatureSynchroniser()
    ccle = pd.DataFrame({"CLEAN": [1.0, 2.0], "CCLE_NAN": [1.0, np.nan]})
    tcga = pd.DataFrame({"CLEAN": [3.0, 4.0], "CCLE_NAN": [3.0, 4.0], "TCGA_NAN": [np.nan, 5.0]})
    kept = fs.drop_incomplete_genes(ccle, tcga, ["CLEAN", "CCLE_NAN"])
    assert kept == ["CLEAN"]


def test_drop_incomplete_genes_noop_when_clean():
    fs = FeatureSynchroniser()
    ccle = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    tcga = pd.DataFrame({"A": [5.0, 6.0], "B": [7.0, 8.0]})
    assert fs.drop_incomplete_genes(ccle, tcga, ["A", "B"]) == ["A", "B"]


def test_feature_synchroniser_custom_lineages():
    fs = FeatureSynchroniser(lineages=["LUSC", "COAD"])
    assert fs.lineages == ["LUSC", "COAD"]
    fs_default = FeatureSynchroniser()
    assert fs_default.lineages == ["LUAD", "BRCA", "SKCM"]


def test_sampler_covers_all_lineages_15way():
    # Every batch must draw each of the 15 lineages from both domains (Day 18
    # generalises the sampler from a 3-lineage assumption to any config-driven set).
    fifteen = [
        "BLCA", "BRCA", "COAD", "GBM", "HNSC", "KIRC", "LGG", "LIHC",
        "LUAD", "LUSC", "OV", "PAAD", "READ", "SKCM", "STAD",
    ]
    lineage_to_idx, _ = build_lineage_maps(fifteen)
    n_per_lineage_ccle, n_per_lineage_tcga = 6, 20
    rng = np.random.default_rng(0)

    def _frame(n_per_lineage):
        rows, labels = [], []
        for lineage in fifteen:
            for _ in range(n_per_lineage):
                rows.append(rng.normal(size=5))
                labels.append(lineage)
        df = pd.DataFrame(rows, columns=[f"G{i}" for i in range(5)])
        df["lineage"] = labels
        return df

    ccle_ds = CCLEDataset(_frame(n_per_lineage_ccle), lineage_to_idx=lineage_to_idx)
    tcga_ds = TCGADataset(_frame(n_per_lineage_tcga), lineage_to_idx=lineage_to_idx)
    sampler = StratifiedContrastiveBatchSampler(ccle_ds, tcga_ds, batch_size=120)
    assert sampler.per_lineage == 4

    for batch in sampler:
        ccle_labels = {ccle_ds[i][1] for i in batch["ccle_indices"]}
        tcga_labels = {tcga_ds[i][1] for i in batch["tcga_indices"]}
        assert ccle_labels == set(range(15))
        assert tcga_labels == set(range(15))
