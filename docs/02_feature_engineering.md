# 02 · Feature Engineering

Normalisation, the train/val/test split, and the data-leakage boundary.

Code: [`pctrans/data/preprocessor.py`](../pctrans/data/preprocessor.py) —
`DataSplitter`. CLI: `pctrans-preprocess --split`.

## 1. Why z-score *after* log1p?

The two matrices arrive on different but both **log-compressed** scales:

- **CCLE** is `log1p(TPM)` (the DepMap `...TPMLogp1` file).
- **TCGA** is Xena's log2-normalised RSEM expression.

Log compression tames the heavy right tail of RNA-seq counts (a few genes with
enormous TPM would otherwise dominate any Euclidean/cosine geometry). But log
values still differ **per gene** in both centre and spread, and — critically —
differ *between domains* for the same gene (platform, library prep, and in-vitro vs
in-vivo effects). A dot product over raw log values would be driven by whichever
genes happen to have the largest absolute expression.

**Per-gene z-scoring** (`StandardScaler`, one mean/std per gene column) fixes both:
every gene contributes on a comparable, zero-mean unit-variance scale, so the
encoder learns *relative* expression structure rather than absolute magnitude. We
z-score after log1p, not instead of it — log first to stabilise variance, then
z-score to standardise it. This is the standard order for RNA-seq fed into a neural
encoder.

One scaler is fit on the **pooled** CCLE+TCGA training expression (not one scaler
per domain): using a shared per-gene mean/variance keeps the two domains on a
single common frame, which is what we want the contrastive loss to then *align*.

## 2. Split strategy

`stratified_split` partitions **each domain independently** but with the same seed
(42) and the same fractions: **70 % train / 15 % val / 15 % test**.

- **Stratified by lineage, within each domain.** For each domain, samples are
  grouped by `lineage` (deterministic sorted group order), each group is shuffled,
  and the test/val/train slices are taken *per lineage*. This preserves the
  LUAD/BRCA/SKCM proportions in every split — a lineage can never vanish from
  train (`test_stratified_split_preserves_lineages`).
- **Disjoint and complete.** Train/val/test IDs are pairwise disjoint and together
  cover every sample exactly once (`test_no_data_leakage`,
  `test_split_covers_all_samples`).
- IDs are the DataFrame index values (`ModelID` for CCLE, `sample` for TCGA), stored
  in `splits.json`.

## 3. Data-leakage prevention protocol

The single most important rule in this project:

> **Scalers are fit on `CCLE_train ∪ TCGA_train` only, then applied unchanged to
> val and test.**

`fit_scalers(ccle_train_expr, tcga_train_expr)`:

1. Concatenate the two **training** frames' gene columns (val/test excluded).
2. Fit one `StandardScaler` on the pooled training matrix.
3. Capture `feature_cols` (the gene column order) so any later frame can be
   realigned before transforming.

`apply_scalers(expr_df, scalers)` then z-scores *any* split with those frozen
train-derived statistics. Because the mean/variance never see val or test, no
information leaks from the evaluation sets into the model's input normalisation.

This boundary is unit-tested directly: `test_scaler_fit_on_train_only` asserts the
fitted mean equals the **train-pool** mean and is **not** equal to the
train+val mean — proving val/test were excluded from fitting.

The exact same frozen scalers are reused at inference time
(`TranslationEmbedder`, `pctrans-query`, the Streamlit app), so a query cell line is
z-scored identically to how training saw it — never re-fit at serve time.

## 4. Artefacts

`save_splits` writes to `data/processed/`:

- `splits.json` — `{"ccle": {"train"/"val"/"test": [ids]}, "tcga": {...}}`
- `scalers.pkl` — `{"scaler": StandardScaler, "feature_cols": [...]}`

Round-tripping both is tested by `test_save_splits_roundtrip`.

## 5. Datasets & the contrastive sampler

`CCLEDataset` / `TCGADataset` wrap a scaled frame and expose
`(features_tensor, lineage_int)` with lineages encoded via
`LINEAGE_TO_IDX = {LUAD:0, BRCA:1, SKCM:2}`. Batch construction (the stratified
contrastive sampler) is documented in [04_training.md](04_training.md).
