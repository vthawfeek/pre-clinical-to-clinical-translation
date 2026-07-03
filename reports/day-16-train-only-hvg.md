# Day 16: Train-Only Feature Selection (Close the Last Leakage Path)

**Date:** 2026-07-03
**Commit:** `day 16: train-only HVG selection, leakage-delta analysis`

## What Was Built

- **`pctrans/data/preprocessor.py`**:
  - `FeatureSynchroniser.select_hvgs(..., train_ids=None)` — when `train_ids` is a
    `{"ccle": [...], "tcga": [...]}` dict of TRAIN sample IDs, per-gene variance is
    computed on that slice only; `train_ids=None` (default) keeps the Phase-1
    all-sample behaviour byte-for-byte.
  - `FeatureSynchroniser.save_filtered(...)` / `DataSplitter.save_splits(...)` — output
    filenames are now overridable (`ccle_name`, `tcga_name`, `gene_list_name`,
    `splits_name`, `scalers_name`) so a train-only run writes a parallel artefact set
    instead of overwriting the Phase-1 files.
- **`pctrans/scripts/preprocess.py`** — new `--hvg-on all|train` flag. `train` splits
  sample IDs *first*, ranks HVG variance on the train slice only, and writes
  `ccle_2k_trainhvg.parquet`, `tcga_2k_trainhvg.parquet`, `gene_list_trainhvg.txt`,
  `splits_trainhvg.json`, `scalers_trainhvg.pkl` — the Phase-1 artefacts are untouched.
- **`pctrans/scripts/train.py`** / **`pctrans/scripts/evaluate.py`** — added
  `--ccle-file/--tcga-file/--splits-file/--scalers-file` (and `--checkpoint-path` on
  train) overrides so both CLIs can point at an alternate artefact set without any
  change to their Phase-1 defaults.
- **`tests/test_data.py`** — two new tests:
  - `test_hvg_train_only_ignores_test` — selecting via `train_ids` matches selecting on
    the train slice directly, and is unchanged even after corrupting every held-out row.
  - `test_hvg_flag_reproduces_phase1` (`@pytest.mark.integration`) — `--hvg-on all`
    (`train_ids=None`) reproduces the committed Day-4 `gene_list.txt` exactly, against
    the real raw CCLE/TCGA data.
- **New data artefacts** (not overwriting Phase-1): `data/processed/ccle_2k_trainhvg.parquet`,
  `tcga_2k_trainhvg.parquet`, `gene_list_trainhvg.txt`, `splits_trainhvg.json`,
  `scalers_trainhvg.pkl`, `models/best_model_trainhvg.pt`, `reports/eval_summary_trainhvg.json`.

## What Was Learned

- **Leakage was negligible — the honest, expected outcome.** Gene-list Jaccard overlap
  between all-sample and train-only HVG selection is 0.9512 (1,950/2,000 genes shared,
  50 swapped each way out of 16,568 common genes). Test kNN@5 stays at 100.0% and TFS
  moves by −0.0009 (0.8915 → 0.8906) — within noise of a single checkpoint, not a
  meaningful regression.
- **kNN@1 actually improved slightly (97.4% → 100.0%).** With one anchor's nearest
  neighbour flipping to correct, this is consistent with the Δ being noise-sized, not a
  systematic effect in either direction — exactly what "negligible leakage" should look
  like.
- **The same biological outlier is the hardest case in both runs.** ACH-000264 (Calu-6,
  anaplastic NSCLC — already flagged as the UMAP outlier on Day 11) is the lowest-TFS
  test cell line under both gene sets (0.662 all-sample, 0.779 train-only). Its retrieval
  actually got easier under train-only HVG (kNN match fraction 0.8 → 1.0) but its
  silhouette contribution stayed the lowest of any test point (0.048 → 0.118) — i.e. this
  is a genuine hard-to-place biological case, not an artefact of which genes happened to
  leak into the HVG ranking.
- **The split itself never moves.** `stratified_split` runs on sample IDs before any gene
  is touched, so `splits.json` and `splits_trainhvg.json` are identical (verified
  index-for-index) — the only variable isolated by this comparison is which 2,000 genes
  get selected, exactly as intended.

## Key Decisions

- **Split-then-select, always, behind a flag rather than a rewrite.** `--hvg-on train`
  forces `split=True` internally (train-only HVG selection is meaningless without a
  split to define "train"), while `--hvg-on all` keeps computing variance across every
  sample before splitting — preserving the exact Phase-1 code path so the backward-compat
  test can assert byte-for-byte equality.
- **Parallel artefact names (`*_trainhvg`) instead of a `--hvg-on`-suffixed output
  directory.** Keeps both runs' files sitting side-by-side in `data/processed/` for easy
  diffing (as this report's Jaccard/delta analysis does) without needing two full data
  directory trees.
- **CLI filename overrides on `train.py`/`evaluate.py` rather than a second script.** The
  training/evaluation logic itself doesn't change between artefact sets — only which
  files it reads — so parameterising the four filenames (mirroring `preprocess.py`'s own
  pattern) avoided duplicating ~150 lines of pipeline code for a one-off comparison run.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
84 passed, 6 deselected in 24.40s

$ uv run pytest tests/test_data.py::test_hvg_flag_reproduces_phase1 -q
1 passed in 33.63s

$ uv run pctrans-preprocess --hvg-on train --seed 42 --val-frac 0.15 --test-frac 0.15
CCLE: 259 cell lines x 19193 genes  (SKCM 110, LUAD 80, BRCA 69)
TCGA: 2264 patients x 20530 genes   (BRCA 1215, LUAD 576, SKCM 473)
Common genes (CCLE & TCGA): 16568
Selected top 2000 HVGs (union-rank method, variance on train slice only)
Stratified split: CCLE train 183 | val 38 | test 38 -- TCGA train 1586 | val 339 | test 339

$ uv run pctrans-train --ccle-file ccle_2k_trainhvg.parquet ... --checkpoint-path models/best_model_trainhvg.pt
Best val kNN@5:      0.9737 (epoch 2)
DECISION: PASS -> proceed to Week 2

$ uv run pctrans-evaluate --model models/best_model_trainhvg.pt --ccle-file ccle_2k_trainhvg.parquet ...
Overall kNN@5 Accuracy:  100.0%   (Wilson 90.8-100.0%, n=38)
Silhouette Score:  +0.56   [boot 95% CI +0.55, +0.58]
TFS (composite):   0.89
DECISION: DEPLOY   [PASS (>=70% -> deploy path, Days 11-14)]
```

## Numbers

**Leakage-delta table (all-sample → train-only HVG, identical seed-42 split):**

| Metric | All-sample HVG | Train-only HVG | Δ |
|---|---|---|---|
| Gene-list Jaccard | — | 0.9512 (1,950/2,050) | 50 genes swapped each way |
| Test kNN@5 | 100.00% | 100.00% | +0.00 pts |
| Test kNN@1 | 97.37% | 100.00% | +2.63 pts |
| Cross-domain silhouette | +0.5662 | +0.5625 | −0.0036 |
| TFS (composite) | 0.8915 | 0.8906 | −0.0009 |
| PCA+kNN baseline | 65.79% | 65.79% | +0.00 pts |
| Best val kNN@5 (checkpoint) | 0.9474 (epoch 2) | 0.9737 (epoch 2) | +0.0263 |

**Interpretation:** small Δ across every metric → the Day-4 all-sample HVG selection was
not meaningfully leaking test information into the Phase-1 headline numbers. Train-only
HVG selection is retained as the new default artefact set going forward (`gene_list_trainhvg.txt`
etc.) since it's the methodologically clean choice, but Phase-1's original 100%/0.89 claims
stand — they were not an artefact of the leakage this closes.

## Next Up

- Day 17 — real baselines (Harmony/ComBat/Scanorama) run on identical test data, plus a
  supervised CCLE→TCGA cross-domain classifier ceiling.
- Fold the train-only HVG artefacts into the Day 15 multi-seed harness (currently the
  seed sweep holds the 2,000-gene set fixed) as a later refinement, per Day 15's report.
- Carry `gene_list_trainhvg.txt` forward as the canonical feature set for Days 18+
  (15-lineage scale-up) rather than re-deriving it per lineage config.
