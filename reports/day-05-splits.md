# Day 5: Preprocessing Pipeline & Train/Val/Test Split

**Date:** 2026-07-02
**Commit:** `day 5: normalisation pipeline, stratified split, Dataset classes, StratifiedContrastiveBatchSampler`

## What Was Built

- `pctrans/data/preprocessor.py` — implemented `DataSplitter` (previously a stub):
  - `stratified_split(ccle_df, tcga_df, val_frac=0.15, test_frac=0.15, seed=42)`: splits each
    domain **independently**, stratifying by lineage *within* the domain (groupby lineage →
    per-lineage shuffle → per-lineage partition), so the LUAD/BRCA/SKCM proportions are preserved
    in train/val/test. Returns `{"ccle": {"train"/"val"/"test": [ids]}, "tcga": {...}}` with IDs
    sorted for byte-stable JSON.
  - `fit_scalers(ccle_train_expr, tcga_train_expr)`: fits **one** `StandardScaler` (per-gene
    z-score) across the *pooled CCLE_train + TCGA_train* expression only. Returns
    `{"scaler": StandardScaler, "feature_cols": [...]}` — the gene order is captured so
    `apply_scalers` can realign any frame.
  - `apply_scalers(expr_df, scalers)`: transforms gene columns with the fitted scaler, carrying
    the `lineage` column through untouched.
  - `save_splits(splits, scalers, out_dir)`: writes `splits.json` (indent=2) and `scalers.pkl`.
- `pctrans/data/dataset.py` — implemented `CCLEDataset` / `TCGADataset` on a shared
  `_ExpressionDataset` base. `__getitem__` returns `(features_tensor[2000], lineage_label_int)`;
  lineage strings encoded via the `LINEAGE_TO_IDX = {"LUAD": 0, "BRCA": 1, "SKCM": 2}` module
  constant (plus `IDX_TO_LINEAGE` for decoding). Unknown lineage labels raise `ValueError`.
  Exposes `.labels`, `.ids`, `.feature_names` for the sampler to group by.
- `pctrans/data/sampler.py` — implemented `StratifiedContrastiveBatchSampler`. Each batch is
  lineage-balanced across both domains: `per_lineage = batch_size // (n_lineages * 2)` samples of
  each lineage from each domain (so `batch_size=48` → 8 CCLE + 8 TCGA per lineage = 24 + 24).
  CCLE is oversampled **with** replacement (N_CCLE << N_TCGA); TCGA is shuffled and drawn
  **without** replacement within an epoch. Yields `{"ccle_indices": [...], "tcga_indices": [...]}`.
  Re-seeds `default_rng(seed + epoch)` each epoch so shuffles differ across passes.
- `pctrans/scripts/preprocess.py` — wired `--split` (previously raised `NotImplementedError`):
  after HVG selection it builds the 2000-gene frames, splits, fits scalers on train, saves
  `splits.json` + `scalers.pkl`, and prints the split-size table. Added `--val-frac`,
  `--test-frac`, `--seed` flags.
- `tests/test_data.py` — added 11 Day 5 tests (leakage, coverage, lineage preservation,
  scaler-fit-on-train-only, z-score sanity, save/load round-trip, dataset shape/label,
  unknown-lineage rejection, sampler lineage balance, TCGA no-replacement, per-epoch reshuffle).
- `data/processed/splits.json`, `data/processed/scalers.pkl` — real output (gitignored).

## What Was Learned

- **The actual split sizes come out well below PLAN.md's estimates for CCLE and slightly above
  for TCGA — but for a known reason, not a bug.** PLAN.md guessed ~300 CCLE / ~2,077 TCGA; the
  real numbers (established Day 4) are 259 CCLE (only these have RNA-seq) and 2,264 TCGA. At
  70/15/15 that gives CCLE 183/38/38 and TCGA 1586/339/339 — proportionally correct against the
  *actual* totals.
- **`int(round(n * frac))` per lineage, not global slicing, is what keeps the small SKCM/BRCA CCLE
  pools from being starved.** Splitting each lineage separately means even BRCA (69 CCLE lines
  total) still contributes to val/test (10 each) instead of getting swallowed by the larger SKCM
  pool in a global shuffle.
- **The pooled-train scaler means (~11.0, ~2.7, ~8.8 for the first 3 HVGs) sit between the CCLE and
  TCGA domain means** rather than at either — which is exactly what makes z-scoring the right fix
  for the Day 4 domain-scale gap. Fitting on the pooled train set centres both domains toward a
  shared origin instead of privileging the larger TCGA distribution.
- **A plain `set`-based groupby split is deterministic across reruns only if the RNG is re-seeded
  and lineage iteration is sorted.** `groupby(sort=True)` + `default_rng(seed)` per domain makes
  `splits.json` byte-identical run to run — required so the Day 7+ training set is reproducible.

## Key Decisions

- **Split each domain independently with per-lineage stratification, not a single pooled split.**
  CCLE and TCGA have very different lineage mixes (CCLE is SKCM-heavy: 110/259; TCGA is BRCA-heavy:
  1215/2264). A pooled split would let the dominant domain/lineage skew the held-out sets. Per-
  domain, per-lineage stratification guarantees every split sees all 3 lineages of both domains.
- **Fit exactly one shared per-gene scaler on `CCLE_train ∪ TCGA_train`, not one per domain.**
  The contrastive objective compares CCLE and TCGA embeddings in the *same* latent space, so both
  towers must receive inputs on a common scale. A single pooled scaler (fit on train only) removes
  the domain-scale offset while leaving genuine lineage signal; per-domain scalers would re-encode
  the domain identity we are trying to strip out. `test_scaler_fit_on_train_only` locks the
  train-only boundary in place.
- **Sampler yields index dicts (`ccle_indices`/`tcga_indices`), not tensors.** Keeps the sampler
  decoupled from batching/collation so the Day 7 trainer can index the datasets and stack tensors
  however it needs; also makes the lineage-balance and no-replacement properties directly testable.
- **CCLE with replacement, TCGA without, per epoch.** With N_CCLE (183 train) << N_TCGA (1586
  train), oversampling CCLE keeps every batch lineage-balanced without exhausting the small pool,
  while drawing TCGA without replacement means one epoch is a genuine single pass over patients.
  Epoch length is bounded by the smallest TCGA lineage pool (`min(pool // per_lineage)`).

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
....................                                                     [100%]
20 passed, 3 deselected in 4.24s

$ uv run pctrans-preprocess --raw-dir data/raw/ --out-dir data/processed/ --n-hvgs 2000 --split
CCLE: 259 cell lines x 19193 genes
  SKCM: 110 cell lines
  LUAD: 80 cell lines
  BRCA: 69 cell lines
TCGA: 2264 patients x 20530 genes
  BRCA: 1215 patients
  LUAD: 576 patients
  SKCM: 473 patients
Common genes (CCLE & TCGA): 16568
Selected top 2000 HVGs (union-rank method)
Top 10 HVGs by mean rank: ['KRT19', 'TYR', 'AGR2', 'RPS4Y1', 'SOX10', 'MLANA', 'KRT7', 'DCT', 'PLP1', 'TFF1']
Saved ccle_2k.parquet, tcga_2k.parquet, gene_list.txt to data/processed/
Stratified split (train / val / test):
  CCLE: train 183 | val 38 | test 38
  TCGA: train 1586 | val 339 | test 339
Scalers fit on 1769 pooled train samples (2000 genes)
Saved splits.json, scalers.pkl to data/processed/
```

Post-run integrity check (leakage / coverage / per-lineage balance):

```
=== CCLE ===
  leakage tr&va: set()  tr&te: set()  va&te: set()
  covers all: True
  train n=  183  {'SKCM': 78, 'LUAD': 56, 'BRCA': 49}
  val   n=   38  {'SKCM': 16, 'LUAD': 12, 'BRCA': 10}
  test  n=   38  {'SKCM': 16, 'LUAD': 12, 'BRCA': 10}
=== TCGA ===
  leakage tr&va: set()  tr&te: set()  va&te: set()
  covers all: True
  train n= 1586  {'BRCA': 851, 'LUAD': 404, 'SKCM': 331}
  val   n=  339  {'BRCA': 182, 'LUAD': 86, 'SKCM': 71}
  test  n=  339  {'BRCA': 182, 'LUAD': 86, 'SKCM': 71}
=== SCALER ===
  type: StandardScaler  n genes: 2000
  mean_[:3]: [11.0354, 2.7177, 8.8194]  scale_[:3]: [5.312, 5.1853, 5.4144]
```

## Numbers

| Split | CCLE (n) | CCLE lineage mix | TCGA (n) | TCGA lineage mix |
|---|---|---|---|---|
| Train | 183 | SKCM 78 / LUAD 56 / BRCA 49 | 1,586 | BRCA 851 / LUAD 404 / SKCM 331 |
| Val   | 38  | SKCM 16 / LUAD 12 / BRCA 10 | 339   | BRCA 182 / LUAD 86 / SKCM 71 |
| Test  | 38  | SKCM 16 / LUAD 12 / BRCA 10 | 339   | BRCA 182 / LUAD 86 / SKCM 71 |
| **Total** | **259** | | **2,264** | |

| Metric | Value |
|---|---|
| Split ratio (train/val/test) | 70 / 15 / 15 (per-lineage) |
| Pooled train samples for scaler fit | 1,769 (183 CCLE + 1,586 TCGA) |
| Scaler features (genes) | 2,000 |
| Cross-split leakage (both domains) | 0 IDs |
| Split coverage | 100% (partitions every sample exactly once) |
| Sampler batch composition (batch_size=48) | 24 CCLE + 24 TCGA (8 per lineage per domain) |
| Day 5 tests added / total fast tests | 11 / 20 passing |

## Next Up (Day 6 — Dual-Tower Architecture & Loss)

- Implement `MLPBlock`, `CCLEEncoder`, `TCGAEncoder` (`[2000→1024→512→256→128→64]`, BN+ReLU+Dropout,
  linear projection head, no BN/activation on the last layer).
- Implement `DualTowerModel` (`forward`, `encode_ccle`, `encode_tcga`) with L2 normalisation onto
  the unit hypersphere.
- Implement `SupConInfoNCELoss` with learnable `log_tau = nn.Parameter(log(1/init_tau))`.
- Write `configs/model.yaml` (hidden dims, dropout, embed_dim, init_tau) and the Day 6 tests
  (encoder output shape, L2 unit-norm, loss is positive scalar, temperature positive).
- Confirm total parameter count ≈ 8.6M across the two towers.
