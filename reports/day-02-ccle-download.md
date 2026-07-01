# Day 2: CCLE Data Download

**Date:** 2026-07-01
**Commit:** `day 2: CCLE download client, lineage filter, Model.csv metadata parse`

## What Was Built

- `pctrans/data/ccle_client.py` — `CCLEClient` class with `download_expression()` and
  `download_metadata()`, both idempotent (skip if file already exists unless `force=True`),
  streamed with a tqdm progress bar, and verified against a per-file minimum byte size to
  catch truncated downloads or HTML error pages. Also adds `filter_lineages(df_meta, lineages)`,
  which maps `OncotreePrimaryDisease` (falling back to `OncotreeSubtype`) to `{LUAD, BRCA, SKCM}`
  via an explicit alias table.
- `pctrans/scripts/download.py` — implemented the `ccle` sub-command of `pctrans-download`:
  downloads both files, then prints metadata shape, per-lineage cell-line counts, expression
  matrix shape, and a NaN count for auditability. `tcga` sub-command left as a Day 3 stub.
- `tests/conftest.py` — added `tiny_ccle_meta` fixture (6 rows spanning LUAD/BRCA/SKCM plus two
  non-target diseases — Glioblastoma and Lung Squamous Cell Carcinoma — to test exclusion).
- `tests/test_data.py` — added `test_ccle_client_filter_lineages`,
  `test_ccle_client_filter_lineages_excludes_other_diseases`, and an `integration`-marked
  `test_ccle_expression_no_nan` that runs against the real downloaded file when present.
- `data/raw/ccle/OmicsExpressionProteinCodingGenesTPMLogp1.csv` (507 MB) and
  `data/raw/ccle/Model.csv` (646 KB) — real DepMap 24Q4 data, downloaded and gitignored.

## What Was Learned

- **DepMap 24Q4's `OncotreePrimaryDisease` column does not contain "Lung Adenocarcinoma" or
  "Breast Cancer" as literal values**, contradicting the alias examples sketched in PLAN.md.
  Lung cancers are bucketed under the primary disease `"Non-Small Cell Lung Cancer"`, which
  conflates adenocarcinoma (LUAD) and squamous cell carcinoma (LUSC) — mapping the whole bucket
  to LUAD would have silently pulled in non-adenocarcinoma cell lines and broken the
  lineage-matching premise of the whole project. The finer-grained `OncotreeSubtype` column
  does carry `"Lung Adenocarcinoma"` as an exact value, so `filter_lineages` resolves
  `OncotreePrimaryDisease` first (catches `"Melanoma"` directly) and falls back to
  `OncotreeSubtype` for LUAD/BRCA, where DepMap splits carcinoma subtypes into several buckets
  (`"Invasive Breast Carcinoma"`, `"Breast Invasive Ductal Carcinoma"`, etc.).
- Real per-lineage counts (LUAD 91, BRCA 92, SKCM 134) differ from PLAN.md's speculative
  estimates (LUAD 97–112, BRCA 125–145, SKCM 62–78) — BRCA is lower and SKCM is notably higher
  than expected. This is a genuine composition difference in the 24Q4 release vs. whatever
  release the plan's numbers were guessed from, not a bug in the filter (see exclusion test).
- DepMap's download API endpoint (`depmap.org/portal/api/download/files`) is behind a Cloudflare
  Turnstile bot check and cannot be curled directly. The actual file bytes are served from
  Figshare (`ndownloader.figshare.com`), which is unauthenticated and scriptable — the
  `figshare.com/articles/dataset/DepMap_24Q4_Public/27993248` page resolves to article ID
  `27993248`, and `api.figshare.com/v2/articles/27993248` lists every file's direct
  `ndownloader` URL and exact byte size.
- The plan's blanket "verify file size > 1 MB" check doesn't fit both files: the real
  `Model.csv` is 646 KB, under 1 MB. Used a per-file minimum instead (100 KB for metadata,
  100 MB for the expression matrix) so a genuinely small legitimate file doesn't get rejected.

## Key Decisions

- **Lineage filter uses `OncotreePrimaryDisease` with an `OncotreeSubtype` fallback**, not
  `OncotreePrimaryDisease` alone as PLAN.md literally specified — necessary because the real
  24Q4 schema has no primary-disease value for lung adenocarcinoma or a single breast-cancer
  bucket. Verified with a dedicated exclusion test that Glioblastoma and squamous-cell lung
  lines are correctly left out.
- **Per-file minimum byte size instead of a single 1 MB threshold** for the "is this a real
  download or an error page" check, since Model.csv (646 KB) is legitimately under 1 MB.
- **`test_ccle_expression_no_nan` marked `integration` and made skip-if-absent** rather than
  requiring the 507 MB file in CI, consistent with the `not slow and not integration` gate the
  project already runs in `.github/workflows/ci.yml`.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
...                                                                      [100%]
3 passed, 1 deselected in 0.26s

$ uv run pytest tests/ -q -m "not slow"   # includes the integration NaN check, data present
....                                                                     [100%]
4 passed in 0.47s

$ uv run pctrans-download ccle --out-dir data/raw/ccle/
Metadata: data\raw\ccle\Model.csv shape=(2105, 47)
  LUAD: 91 cell lines
  BRCA: 92 cell lines
  SKCM: 134 cell lines
Expression matrix: data\raw\ccle\OmicsExpressionProteinCodingGenesTPMLogp1.csv shape=(1673, 19193)
NaN values in expression matrix: 0
```

## Numbers

| Item | Value |
|---|---|
| Model.csv rows (all DepMap models) | 2,105 |
| Model.csv columns | 47 |
| Expression matrix shape | 1,673 cell lines × 19,193 genes |
| Expression matrix file size | 506,628,654 bytes (~483 MiB) |
| Model.csv file size | 645,696 bytes (~631 KiB) |
| LUAD cell lines | 91 |
| BRCA cell lines | 92 |
| SKCM cell lines | 134 |
| Total target-lineage cell lines | 317 |
| NaN values in expression matrix | 0 |
| Download time (expression, ~17 MB/s) | ~30 s |

## Next Up

- Implement `pctrans/data/tcga_client.py` — `TCGAClient` for the UCSC Xena Pan-Cancer expression
  matrix + survival/phenotype table.
- Add `filter_tcga_lineages` for the `cancer type abbreviation` column (should map directly to
  LUAD/BRCA/SKCM without the alias gymnastics CCLE needed).
- Run `pctrans-download tcga --out-dir data/raw/tcga/` and verify sample counts (~515 LUAD,
  ~1,093 BRCA, ~469 SKCM expected).
- Watch for the same kind of schema drift seen today — verify Xena's actual column names/values
  before trusting PLAN.md's speculative figures.
