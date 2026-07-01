# Day 3: TCGA Data Download

**Date:** 2026-07-01
**Commit:** `day 3: TCGA download client, Xena phenotype parse, lineage filter`

## What Was Built

- `pctrans/data/tcga_client.py` — `TCGAClient` class with `download_expression()` (streams the
  gzipped Xena PANCAN expression matrix with a tqdm progress bar, gunzips it in place, then
  deletes the intermediate `.gz`) and `download_phenotype()` (the phenotype/survival table is
  served uncompressed already, so it's a straight streamed download). Both are idempotent
  (skip if the final file exists unless `force=True`) and verified against a per-file minimum
  byte size, mirroring `CCLEClient`. Also adds `filter_tcga_lineages(df_pheno, lineages)`,
  which returns a boolean mask over the `cancer type abbreviation` column — no alias table
  needed since TCGA already uses the LUAD/BRCA/SKCM codes directly.
- `pctrans/scripts/download.py` — implemented the `tcga` sub-command of `pctrans-download`:
  downloads both files, prints phenotype shape and per-lineage patient counts, then reports the
  expression matrix's gene/sample counts and first 3 gene identifiers **without loading the
  ~1.1 GB matrix into memory** (line-counts the file and reads only the first column for a
  3-row peek).
- `tests/conftest.py` — added `tiny_tcga_meta` fixture (6 rows: LUAD, BRCA, SKCM, plus GBM and
  COAD as non-target lineages to test exclusion).
- `tests/test_data.py` — added `test_tcga_client_filter_lineages`,
  `test_tcga_client_filter_lineages_matched_labels`, and two `integration`-marked tests
  (`test_tcga_expression_gene_count`, `test_tcga_phenotype_has_target_lineages`) that run
  against the real downloaded files when present.
- `data/raw/tcga/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena` (1.1 GB) and
  `data/raw/tcga/Survival_SupplementalTable_S1_20171025_xena_sp.tsv` (2.4 MB) — real UCSC Xena
  PANCAN data, downloaded and gitignored.

## What Was Learned

- **PLAN.md's Xena host (`tcga-xena-hub.s3...`) returns 403 for both files.** The actual data
  lives in the `tcga-pancan-atlas-hub` S3 bucket — confirmed by HEAD-requesting both hosts
  before writing any download code (the same "plan URL doesn't match reality" pattern Day 2 hit
  with DepMap's API). The phenotype file is also not gzipped despite PLAN.md's `.gz` URL — it's
  served as a plain, unnamed-extension tab-separated file directly.
- **The expression matrix's row identifiers are Entrez-style numeric IDs (e.g. `100130426`),
  not HUGO gene symbols** as PLAN.md's Dataset Specifications claimed ("Gene IDs: HUGO symbols,
  no Entrez ID suffix"). Verified by reading the first data row directly. This matters for
  Day 4's gene harmonisation step: CCLE columns strip down to HUGO symbols, so TCGA's numeric
  IDs will need to go through a symbol map first. A matching probemap file
  (`probeMap/hugo_gencode_good_hg19_V24lift37_probemap`) exists in the same S3 bucket and
  returns 200 OK — flagged here so Day 4 doesn't have to rediscover it.
- **Real per-lineage patient counts (LUAD 641, BRCA 1,236, SKCM 479) are all higher than
  PLAN.md's estimates** (515 / 1,093 / 469). The phenotype table also covers all 33 TCGA
  cohorts (12,591 rows total), not just the three target lineages — expected, since it's the
  pan-cancer survival table, and `filter_tcga_lineages` correctly narrows to just LUAD/BRCA/SKCM.
- Reading a 1.1 GB tab-separated file fully into pandas just to report `.shape` would be wasteful
  and slow; counting newlines and peeking at the first column instead keeps the CLI's
  verification step fast and low-memory.

## Key Decisions

- **`filter_tcga_lineages` returns a boolean mask** (not the matched lineage labels), consistent
  with `filter_lineages` from Day 2 and with how it's used in `download.py` to count patients
  per lineage. PLAN.md's suggested test (`set(filtered.unique()).issubset({...})`) only makes
  sense if `filtered` were the label Series, so the test was adapted to apply the mask first
  (`tiny_tcga_meta.loc[filtered, "cancer type abbreviation"]`) before asserting the label set —
  same intent as PLAN.md's test, correct against the documented `pd.Series[bool]` return type.
- **Verify expression matrix metadata without a full `pd.read_csv` load** (line-count + 3-row
  peek instead), given the file is ~1.1 GB uncompressed — avoids unnecessary memory pressure for
  a verification step that only needs shape and a handful of values.
- **Delete the intermediate `.gz` after gunzipping** the expression matrix rather than keeping
  both compressed and uncompressed copies on disk, since only the uncompressed file is used
  downstream (Day 4 preprocessing) and disk space isn't free.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
.....                                                                    [100%]
5 passed, 3 deselected in 0.18s

$ uv run pytest tests/ -q -m "not slow"   # includes both CCLE and TCGA integration tests, data present
........                                                                 [100%]
8 passed in 0.68s

$ uv run pctrans-download tcga --out-dir data/raw/tcga/
Phenotype: data\raw\tcga\Survival_SupplementalTable_S1_20171025_xena_sp.tsv shape=(12591, 34)
  LUAD: 641 patients
  BRCA: 1236 patients
  SKCM: 479 patients
Expression matrix: data\raw\tcga\EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena genes=20531 samples=11069
First 3 genes: [100130426, 100133144, 100134869]
```

## Numbers

| Item | Value |
|---|---|
| Phenotype table rows (all 33 TCGA cohorts) | 12,591 |
| Phenotype table columns | 34 |
| Expression matrix shape | 20,531 genes × 11,069 samples |
| Expression matrix file size (uncompressed) | 1,177,513,907 bytes (~1.10 GiB) |
| Expression matrix file size (gzipped download) | ~331 MB |
| Phenotype file size | 2,419,504 bytes (~2.31 MiB) |
| LUAD patients | 641 |
| BRCA patients | 1,236 |
| SKCM patients | 479 |
| Total target-lineage patients | 2,356 |
| Download + gunzip time | ~28 s (gzip) + a few seconds (phenotype + gunzip) |

## Next Up

- Day 4: `pctrans/data/preprocessor.py` — `FeatureSynchroniser` (`load_ccle`, `load_tcga`,
  `find_common_genes`, union-rank `select_hvgs`, `save_filtered`) and the `pctrans-preprocess` CLI.
- Resolve the TCGA gene-ID mismatch discovered today: map the expression matrix's numeric
  Entrez-style row IDs to HUGO symbols via the `hugo_gencode_good_hg19_V24lift37_probemap` file
  before intersecting with CCLE's (already-HUGO) gene list.
- Build `notebooks/01_eda.ipynb` with sample counts, gene distributions, and the "before
  training" PCA domain-gap plot.
- Verify common-gene overlap between CCLE (~19,193 genes) and TCGA (~20,531 genes) is close to
  the plan's ~18,000 estimate once symbols are harmonised.
