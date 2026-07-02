# 01 · Data Pipeline

How raw CCLE and TCGA expression matrices become a single, shared 2,000-gene
feature space that both encoder towers can consume.

Code: [`pctrans/data/ccle_client.py`](../pctrans/data/ccle_client.py),
[`pctrans/data/tcga_client.py`](../pctrans/data/tcga_client.py),
[`pctrans/data/preprocessor.py`](../pctrans/data/preprocessor.py).
CLI: `pctrans-download`, `pctrans-preprocess`.

## 1. Sources

### CCLE (cell lines) — DepMap 24Q4, hosted on Figshare

| Artefact | URL | Local filename |
|---|---|---|
| Expression | `https://ndownloader.figshare.com/files/51065489` | `OmicsExpressionProteinCodingGenesTPMLogp1.csv` |
| Metadata | `https://ndownloader.figshare.com/files/51065297` | `Model.csv` |

Expression values are **log1p(TPM)** for protein-coding genes; rows are cell lines
(`ModelID`, e.g. `ACH-000001`), columns are genes. The download client streams the
file and validates it is `≥ 100 MB` (the real file is ~500 MB) so a truncated
download or an HTML error page is caught and deleted rather than silently used.

### TCGA (patients) — UCSC Xena Pan-Cancer Atlas hub, hosted on S3

| Artefact | URL | Local filename |
|---|---|---|
| Expression | `.../download/EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz` | `EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena` |
| Phenotype | `.../download/Survival_SupplementalTable_S1_20171025_xena_sp` | `Survival_SupplementalTable_S1_20171025_xena_sp.tsv` |

Host prefix: `https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com`. The
expression matrix is the batch-corrected (`EB++AdjustPANCAN`) RSEM matrix, gzip on
the wire and decompressed on write; it is genes-as-rows, samples-as-columns
(~1.2 GB uncompressed), so the loader reads only the phenotype-matched sample
columns. Values are Xena-normalised log2 expression.

> **Note on the URLs.** PLAN.md originally pointed at the `tcga-xena-hub` bucket,
> which returns HTTP 403 for these files. The matrices actually live in
> `tcga-pancan-atlas-hub` (confirmed with a HEAD request against both hosts); the
> client uses the working bucket.

### Lineage filter

Both domains are filtered to three lineages: **LUAD** (lung adenocarcinoma),
**BRCA** (breast carcinoma), **SKCM** (cutaneous melanoma).

- **TCGA** is easy: the phenotype table's `cancer type abbreviation` column already
  uses these exact codes, so `filter_tcga_lineages` is a direct `.isin([...])`.
- **CCLE** needs an alias table. DepMap 24Q4's `OncotreePrimaryDisease` has no
  single "Lung Adenocarcinoma" or "Breast Cancer" bucket (NSCLC lumps LUAD+LUSC;
  breast splits across subtypes), so `filter_lineages` resolves
  `OncotreePrimaryDisease` first, then falls back to the finer `OncotreeSubtype`
  (see `LINEAGE_ALIASES` in `ccle_client.py`). Melanoma resolves at the
  primary-disease level.

## 2. Gene-ID harmonisation (step by step)

The two domains do **not** share a gene-column convention. Harmonisation happens in
`FeatureSynchroniser.load_ccle` / `.load_tcga` / `.find_common_genes`:

1. **CCLE columns are `"SYMBOL (ENTREZ_ID)"`** — e.g. `EGFR (1956)`, `TP53 (7157)`.
   Strip the trailing Entrez ID with the regex `` \(\d+\)$`` (`CCLE_ENTREZ_SUFFIX`)
   so the column becomes the bare HUGO symbol `EGFR`.
2. **TCGA rows are already HUGO symbols** (`EGFR`, `TP53`, …) after transposing the
   genes-as-rows matrix to samples-as-rows. A handful of symbols (e.g. `SLC35E2`)
   appear twice in the Xena matrix; the loader keeps the first occurrence so the
   gene index is unique.
3. **Attach lineage.** Each domain's expression frame gets a trailing `lineage`
   column, joined from its metadata by sample/model ID.
4. **Common-gene intersection.** `find_common_genes` takes the **set intersection**
   of the two symbol lists and returns it **sorted** (reproducibility). Only genes
   measured in *both* domains survive.

Worked micro-example:

```
CCLE columns : "EGFR (1956)", "TP53 (7157)", "MLANA (2315)"
  -> stripped: "EGFR",        "TP53",        "MLANA"
TCGA symbols : "EGFR", "TP53", "KRAS"
common genes : sorted({EGFR,TP53,MLANA} ∩ {EGFR,TP53,KRAS}) = ["EGFR", "TP53"]
```

## 3. HVG selection — the union-rank method

Feeding both towers every common gene would let TCGA's much larger sample count
dominate any pooled-variance ranking. Instead `select_hvgs` gives **each domain
equal weight** by ranking variance *within* each domain and averaging the ranks:

```
var_ccle[g] = Var over CCLE samples of gene g   (ddof=1)
var_tcga[g] = Var over TCGA samples of gene g   (ddof=1)
rank_ccle   = var_ccle.rank()      # 1 = lowest variance, N = highest
rank_tcga   = var_tcga.rank()
mean_rank[g] = (rank_ccle[g] + rank_tcga[g]) / 2
```

Genes are sorted by `mean_rank` **descending**, ties broken **alphabetically by
symbol** (a stable `mergesort`), and the top `n_hvgs = 2000` are kept. Alphabetical
tie-breaking makes the output fully deterministic (verified by
`test_hvg_tie_break_is_alphabetical`).

### Worked example — top 5 genes and their ranks

Suppose four common genes with these within-domain variance ranks (higher rank =
more variable; `N` = number of common genes):

| Gene | rank_ccle | rank_tcga | mean_rank | Selected? |
|---|---|---|---|---|
| `MLANA` | N | N−1 | (N + N−1)/2 | ✅ 1st |
| `EGFR` | N−2 | N | (2N−2)/2 | ✅ 2nd |
| `TP53` | N−1 | N−3 | (2N−4)/2 | ✅ 3rd |
| `ACTB` | 3 | 5 | 4 | ❌ (housekeeping, low variance) |

`MLANA` (a melanocyte marker) ranks near the top in **both** domains, so it survives
even though no single-domain ranking would guarantee it. Housekeeping genes like
`ACTB` have low variance in both domains and drop out. This is exactly the
behaviour we want: keep genes that carry lineage signal in cell lines **and**
patients.

## 4. Outputs

`save_filtered` writes, to `data/processed/`:

- `ccle_2k.parquet` — CCLE samples × 2,000 HVGs + `lineage` (indexed by `ModelID`)
- `tcga_2k.parquet` — TCGA samples × 2,000 HVGs + `lineage` (indexed by `sample`)
- `gene_list.txt` — the 2,000 HUGO symbols in HVG-rank order

Splitting and scaling (`--split`) are covered in
[02_feature_engineering.md](02_feature_engineering.md).
