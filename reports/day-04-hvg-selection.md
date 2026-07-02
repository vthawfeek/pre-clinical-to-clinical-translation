# Day 4: Feature Synchronisation & Union-Rank HVG Selection

**Date:** 2026-07-02
**Commit:** `day 4: feature synchronisation, union-rank HVG selection, EDA notebook`

## What Was Built

- `pctrans/data/preprocessor.py` — implemented `FeatureSynchroniser`:
  - `load_ccle(raw_dir)`: reads `Model.csv` + the expression matrix, filters to LUAD/BRCA/SKCM
    via the existing `filter_lineages`, strips the `" (ENTREZ_ID)"` suffix from expression
    column names (`"EGFR (1956)"` → `"EGFR"`), and joins the resolved per-row lineage code onto
    the expression frame as a trailing `lineage` column.
  - `load_tcga(raw_dir)`: reads the phenotype table, filters to the 3 lineages, then reads
    *only* the matched sample columns out of the ~1.2 GB Xena expression file (header is read
    first to intersect with the wanted sample-ID set, then that set is passed to
    `pd.read_csv(usecols=...)`) — cuts real load time from ~230 s to ~11 s. Drops one duplicated
    gene row (`SLC35E2` appears twice in the raw file) and transposes to samples × genes.
  - `find_common_genes(ccle_genes, tcga_genes)`: sorted set intersection.
  - `select_hvgs(ccle_expr, tcga_expr, common_genes, n_hvgs)`: union-rank method — per-gene
    variance ranked independently within each domain, averaged, ties broken alphabetically by
    gene symbol for determinism.
  - `save_filtered(...)`: writes `ccle_2k.parquet`, `tcga_2k.parquet` (genes + `lineage` column)
    and `gene_list.txt` (one HUGO symbol per line, HVG-rank order).
- `pctrans/scripts/preprocess.py` — implemented the `pctrans-preprocess` CLI (`--raw-dir`,
  `--out-dir`, `--n-hvgs`; `--split` still raises `NotImplementedError` pending Day 5's
  `DataSplitter`). Prints per-lineage sample counts, common-gene count, and the top 10 HVGs.
- `notebooks/01_eda.ipynb` — 5-section EDA notebook (sample counts, expression distributions,
  "before" PCA domain-gap plot, top-20 HVG biological sanity check, variance spectrum), with
  markdown cells recording the real numbers from today's run (code cells are not pre-executed —
  no `nbconvert`/`nbformat` in this project's dependencies — see Verification below for how the
  numbers were produced).
- `tests/test_data.py` — added `test_find_common_genes_sorted`, `test_hvg_selection_count`,
  `test_hvg_selection_deterministic`, `test_hvg_tie_break_is_alphabetical`.
- `data/processed/ccle_2k.parquet` (259 × 2001), `data/processed/tcga_2k.parquet`
  (2264 × 2001), `data/processed/gene_list.txt` (2000 genes) — real output, gitignored.

## What Was Learned

- **TCGA's Xena expression file is *not* purely HUGO symbols**, despite PLAN.md's assumption —
  most gene rows are symbols, but a subset (mostly ambiguous/unmapped probes, all clustered at
  the top of the file) are bare Entrez IDs (e.g. `100130426`). Rather than build a separate
  Entrez→symbol mapping, `find_common_genes` just takes the set intersection of CCLE's stripped
  symbols against TCGA's row IDs as-is — the symbol-format TCGA rows match directly, the
  Entrez-format ones simply don't intersect and get correctly excluded. Net result: 16,568
  common genes (vs. PLAN's speculative ~18,000 — same order of magnitude, lower because DepMap's
  file is protein-coding-only at 19,193 genes).
- **CCLE per-lineage counts dropped from Day 2's meta-only numbers (LUAD 91, BRCA 92, SKCM 134,
  total 317) to today's expression-joined numbers (LUAD 80, BRCA 69, SKCM 110, total 259)** —
  432 of 2,105 total DepMap models have no row in the expression matrix (not every profiled cell
  line has RNA-seq). This is a real data characteristic, confirmed by directly checking
  `ModelID` membership, not a bug in the join.
- **The raw "before" PCA domain gap is partly a normalisation-scale artefact, not pure
  biology**: CCLE is log2(TPM+1) and TCGA is log2(normalized_count+1) — different upstream
  pipelines on the same log-scale family. Observed means differ by domain far more than by
  lineage (e.g. SKCM: CCLE mean 1.89 vs TCGA mean 5.88 on the same 2,000 genes). PC1 (37.7% of
  variance) separates almost purely by domain (CCLE mean PC1 -198.5 vs TCGA +22.7), while
  lineage only shows up weakly on PC2 (12.6%). This makes per-gene z-scoring (Day 5) a
  correctness requirement, not just good practice — without it the contrastive model could learn
  to exploit the scale offset as a shortcut instead of the shared lineage signal.
  This is the "before" picture PLAN.md's Day 11 UMAP is explicitly meant to answer with an
  "after."
- **Mean expression is a poor HVG criterion, and the pooled per-lineage top-10-by-mean-expression
  table proves it**: it's dominated by collagen/stromal/immune genes (COL1A1, COL1A2, COL3A1,
  FN1, VIM, CD74) that are highly expressed in *every* solid-tumour patient sample — the TME
  artefact PLAN.md's Context section calls out. The union-rank *variance* method instead
  surfaces real lineage markers (SOX10, MLANA, TYR, DCT for SKCM; SFTPB, SLC34A2 for LUAD) that
  a naive top-mean-expression list would miss.

## Key Decisions

- **Read only the phenotype-matched sample columns from the TCGA expression file**, via a
  two-pass approach (read header first, intersect with target sample IDs, then pass that plain
  `set` — not a lambda — to `pd.read_csv(usecols=...)`). A lambda-based `usecols` predicate is
  correctness-equivalent but ~20x slower (227 s vs 11 s in a side-by-side timing test) because
  pandas can't take its optimized path for a callable. Necessary because ~92 of the 2,356
  lineage-filtered phenotype samples have no matching expression column, so a plain `set` passed
  directly raises `ValueError: Usecols do not match columns`.
- **Tie-break HVG ranking alphabetically by gene symbol** rather than leaving order
  pandas-arbitrary, so `gene_list.txt` (and therefore the whole downstream feature space) is
  byte-identical across reruns — verified by `test_hvg_selection_deterministic` and a dedicated
  constructed-tie test (`test_hvg_tie_break_is_alphabetical`).
- **Kept `gene_list.txt` in HVG-rank order (most variable first) rather than alphabetical**,
  since the rank order is directly useful for reporting "top N HVGs" (used above) and the
  alphabetical case is already covered by `find_common_genes`'s sort.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
.........                                                                [100%]
9 passed, 3 deselected in 0.31s

$ uv run pctrans-preprocess --raw-dir data/raw/ --out-dir data/processed/ --n-hvgs 2000
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
```

Post-run sanity check (no NaNs, correct shapes):

```
ccle (259, 2001)   tcga (2264, 2001)   n genes 2000
any na ccle False  any na tcga False
```

### EDA notebook figures (the numbers behind `notebooks/01_eda.ipynb`'s markdown cells)

**Section 3 — PCA "before" domain gap (ASCII summary):**

```
PC1 (37.7% var)              CCLE cluster                         TCGA cluster
                          x x x  x                         o o o o o o o o o o o
                        x x x x x x                      o o o o o o o o o o o o o
                          x x  x x                          o o o o o o o o o o
            <-------------------------------|---------------------------------->
         PC1 mean(CCLE) = -198.5                          PC1 mean(TCGA) = +22.7
```

PC1 (37.7%) separates almost entirely by domain; lineage only shows up (weakly) on PC2 (12.6%,
SKCM mean +62.3 vs LUAD/BRCA around -18 to -19).

**Section 4 — top 20 HVGs, annotated:**

| Gene | Biology |
|---|---|
| TYR, SOX10, MLANA, DCT, PRAME, MIA, TYRP1 | Melanocyte/melanoma markers (SKCM) |
| KRT19, KRT7, AGR2, TFF1, EPCAM, S100P, S100A14, CEACAM6, ELF3, RAB25, SLPI | Epithelial/carcinoma markers (LUAD+BRCA) |
| RPS4Y1 | Y-chromosome gene — likely a sex-composition confound, flagged for Day 9 |

## Numbers

| Metric | Value |
|---|---|
| CCLE samples (LUAD/BRCA/SKCM, RNA-seq available) | 259 (SKCM 110, LUAD 80, BRCA 69) |
| TCGA samples (LUAD/BRCA/SKCM) | 2,264 (BRCA 1,215, LUAD 576, SKCM 473) |
| Common genes (CCLE ∩ TCGA) | 16,568 |
| Selected HVGs | 2,000 |
| CCLE variance range across selected HVGs (rank 1 → 2000) | 20.80 → 0.79 |
| TCGA variance range across selected HVGs (rank 1 → 2000) | 47.26 → 1.68 |
| PCA PC1 / PC2 explained variance (raw, pre-training) | 37.7% / 12.6% |
| TCGA expression file targeted-column read time | ~11 s (vs. ~227 s for a full/lambda read) |

## Next Up

- Implement `DataSplitter.stratified_split` — per-domain, per-lineage stratified train/val/test
  split (no leakage across splits).
- Implement `DataSplitter.fit_scalers` / `apply_scalers` — per-gene `StandardScaler` fit on
  pooled CCLE_train + TCGA_train only, applied to val/test. Section 2's domain-scale gap above
  is exactly what this step needs to remove.
- Implement `CCLEDataset` / `TCGADataset` (PyTorch) and `StratifiedContrastiveBatchSampler`.
- Wire `--split` into `pctrans-preprocess` (currently raises `NotImplementedError`).
- Write `reports/day-05-splits.md` with the split size table and per-lineage balance check.
