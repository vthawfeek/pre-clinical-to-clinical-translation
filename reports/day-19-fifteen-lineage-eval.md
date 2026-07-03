# Day 19: 15-Lineage Evaluation — Headroom & Error Structure

**Date:** 2026-07-03
**Commit:** `day 19: 15-lineage evaluation, error-structure biology analysis`

## What Was Built

- `pctrans/evaluation/knn.py` — `knn_accuracy_from_embeddings`/`knn_retrieval_accuracy` accept
  optional `idx_to_lineage`/`lineage_order` overrides (default to the Phase-1 3-lineage module
  constants, so every existing caller is unaffected). New `KNOWN_CONFUSABLE_PAIRS` (LUAD/LUSC,
  GBM/LGG, COAD/READ, LUSC/HNSC), `top_confusions()` (ranked off-diagonal `(true, pred, count)`
  triples), and `confusable_pair_mass()` (fraction of off-diagonal error mass on named pairs).
- `pctrans/evaluation/baselines.py` — `pca_knn` threads the same `idx_to_lineage`/`lineage_order`
  overrides through so the no-alignment baseline scores correctly on an arbitrary lineage count.
- `pctrans/scripts/evaluate.py` — new `--data-config` flag (mirrors Day 18's `pctrans-train`
  convention); builds the label-id map via `build_lineage_maps` and threads it through dataset
  construction, `knn_retrieval_accuracy`, and `pca_knn`. The random baseline is now computed as
  `1/len(lineages)` instead of the hardcoded 3-lineage `1/3`, and `lineages` is recorded in
  `eval_summary.json`.
- `pctrans/scripts/visualize.py` — same `--data-config` plus `--ccle-file/--tcga-file/
  --splits-file/--scalers-file` overrides as `evaluate.py`, plus a new `--output-suffix` so the
  15-lineage UMAP/before-after/TFS-ranking/embeddings artefacts land alongside the Phase-1 ones
  (`*_15` suffix) instead of overwriting them.
- `pctrans/evaluation/viz.py` — `LINEAGE_COLORS` extended from 3 to all 15 Phase-2 lineages (the
  original 3 keep their Phase-1 hex values); `lineage_domain_scatter`, `lineage_domain_scatter_static`,
  and `before_after_panel` accept optional `lineage_order`/`idx_to_lineage` overrides. New
  `confusion_matrix_heatmap()` — static, row-normalised-by-default matplotlib heatmap for any
  square confusion matrix.
- `tests/test_evaluation.py` — 6 new tests: `knn_accuracy_from_embeddings` with a custom 5-lineage
  map, a synthetic 15-lineage `eval15` fixture, confusion-matrix shape, `confusable_pair_mass`
  (majority on named pairs; zero when no named pair is present), and `top_confusions` ordering.
- `tests/test_viz.py` — 3 new tests: `lineage_domain_scatter` with a lineage-order override,
  `confusion_matrix_heatmap` figure/labels, and its row-normalisation.
- `notebooks/03_evaluation.ipynb` — fixed a latent bug in Section 5 (the baseline bar chart read a
  `summary["baselines"]["harmony_reference"]` key that Day 17 removed when the baselines dict was
  restructured; it now reads the real Harmony/PCA/supervised-ceiling numbers from
  `reports/baselines.json`). Added **Section 6 — 15-Lineage Evaluation** (metrics table + 3-vs-15
  comparison, per-lineage kNN@5, confusion heatmap, `top_confusions`/`confusable_pair_mass`
  printout, 15-lineage UMAP, and a biology-read markdown cell). Executed end-to-end via
  `jupyter nbconvert --execute` (ephemeral `--with nbconvert`, not added as a project dependency,
  matching Day 4's precedent) — exit 0, 0 error outputs across all 26 cells.
- Real Day 19 output: `reports/eval_summary_15.json` (from
  `pctrans-evaluate --data-config configs/data_15.yaml ...`), `reports/confusion_matrix_15.png`,
  `reports/umap_test_set_15.{html,png}`, `reports/umap_before_after_15.png`,
  `reports/tfs_ranking_15.html` (from `pctrans-visualize --output-suffix _15 ...`).

## What Was Learned

- **The errors are exactly the ones the plan predicted, and they're clean.** LGG and READ score
  0% on the test set — but 100% of their misses land on their named partner (all 3 LGG anchors →
  GBM, both READ anchors → COAD), not scattered noise. LUSC (25% correct) sends its other 3 misses
  entirely to LUAD and HNSC, its two named partners. This is the strongest evidence so far that the
  model learned real lineage biology rather than a shortcut: a model exploiting an artefact would
  have no reason to fail in exactly the biologically adjacent direction.
- **Off-diagonal error mass is heavily concentrated, not proof by majority.** 45.8% of all 24
  test-set misclassifications fall on the 4 named pairs, which occupy only 8 of 210 possible
  off-diagonal cells (3.8%) — a 12x enrichment over random chance. It falls just short of a literal
  majority (>50%), so the report states the real number rather than rounding up to "most errors."
- **A Day-17 regression had gone unnoticed until the notebook was actually executed.** When the
  baselines dict was restructured to add real Harmony/ComBat/Scanorama numbers, the
  `harmony_reference` placeholder key the notebook's Section 5 depended on was silently dropped —
  nothing caught it because no day between 17 and 19 re-ran the notebook. This is exactly the
  failure mode Day 19's "notebook executes end-to-end" verification step exists to catch.
  `nbconvert` still isn't a project dependency (per Day 4's decision to keep it minimal); it was
  installed ephemerally via `uv run --with nbconvert` for the execute-and-verify step.
  Restricting the daily quality gate's ruff/pytest run to `pctrans/ tests/` means the notebook's
  correctness is only checked by explicit execution, not by every day's automated gate — worth
  keeping in mind for later days that touch shared JSON schemas the notebook reads.
- **The random baseline was silently wrong for any non-3-lineage run.** `evaluate.py` imported the
  hardcoded `RANDOM_BASELINE = 1/3` constant from `baselines.py`; running it against the
  15-lineage config would have printed "33.3%" instead of the correct "6.7%" chance rate. Caught
  and fixed while generalising the CLI, before it ever produced a real report.

## Key Decisions

1. **Every generalised function takes `idx_to_lineage`/`lineage_order` as optional keyword
   arguments defaulting to the Phase-1 3-lineage module constants**, rather than making the
   3-lineage defaults implicit or removing them. This keeps every existing call site (including
   the `pipeline` test fixture, which never passes these) byte-identical to Phase-1 behaviour
   while letting `--data-config configs/data_15.yaml` opt into 15 lineages — the same pattern Day
   18 established for `build_lineage_maps` itself.
2. **The confusion-matrix heatmap row-normalises by default.** Lineage test-set counts here range
   from n=2 (READ) to n=16 (SKCM); a raw-count heatmap would visually bury the small-lineage
   confusions (READ's 2/2 miss-to-COAD) under BRCA's larger absolute numbers. Raw counts remain
   available via `normalize=False` for anyone who wants them.
3. **Fixed the stale Section 5 baseline cell instead of leaving it broken or deleting it.** The
   plan's Day 19 verification step requires the whole notebook to execute cleanly, and the honest
   fix (reading `reports/baselines.json` for the real Day 17 numbers) is strictly better than the
   literature placeholder it replaced — consistent with the project's "replace placeholders with
   real numbers" pattern from Day 17 itself.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
........................................................................ [ 69%]
...............................                                          [100%]
103 passed, 6 deselected in 10.55s

$ uv run pytest tests/ -q -m "slow and not integration"
..                                                                       [100%]
2 passed, 107 deselected, 2 warnings in 25.44s
```

`pctrans-evaluate --model models/best_model_15.pt --ccle-file ccle_2k_15.parquet --tcga-file
tcga_2k_15.parquet --splits-file splits_15.json --scalers-file scalers_15.pkl --data-config
configs/data_15.yaml --output reports/eval_summary_15.json`:

```
Test set: CCLE 111 + TCGA 1070

==============================================
           GATE 1 EVALUATION REPORT
==============================================
Overall kNN@5 Accuracy:   78.4%   (threshold: 70%)
  Wilson 95% CI:    69.8- 85.0%  (n=111)
  Bootstrap 95% CI:  70.3- 85.6%
Per-lineage kNN@5:
  BLCA:   60.0%   BRCA:   90.0%   COAD:   80.0%   GBM:   66.7%   HNSC:  100.0%
  KIRC:  100.0%   LGG:     0.0%   LIHC:  100.0%   LUAD:   66.7%   LUSC:   25.0%
  OV:    80.0%    PAAD:   87.5%   READ:    0.0%   SKCM:  100.0%  STAD:   80.0%
kNN@k table:  k=1:75.7%  k=3:78.4%  k=5:78.4%  k=10:79.3%
Silhouette Score:  +0.70   (boot 95% CI +0.68, +0.73)
TFS (composite):   0.82
Random baseline:   6.7%   (1/15 lineages)
PCA+kNN baseline:  25.2%
==============================================
DECISION: DEPLOY   [PASS (>=70% -> deploy path, Days 11-14)]
==============================================
```

Error-structure analysis (`top_confusions` / `confusable_pair_mass` on the real confusion matrix):

```
Top off-diagonal confusions (true -> pred : count):
  LGG -> GBM : 3   *named confusable pair*
  GBM -> LGG : 2   *named confusable pair*
  LUAD -> PAAD : 2
  LUSC -> LUAD : 2   *named confusable pair*
  OV -> LUAD : 2
  READ -> COAD : 2   *named confusable pair*
  BLCA -> HNSC : 1   BLCA -> PAAD : 1   BRCA -> HNSC : 1   COAD -> LUAD : 1

Named-pair share of off-diagonal error mass: 45.8% (11 of 24 total misclassifications)
Named pairs occupy 8/210 possible off-diagonal cells (3.8%) -> 12.0x enrichment over random chance
```

Notebook: `jupyter nbconvert --to notebook --execute --inplace notebooks/03_evaluation.ipynb`
exits 0; 26 cells, 0 error outputs (checked programmatically after execution).

## Numbers

| Metric | 3-lineage (Phase 1) | 15-lineage (Day 19) |
|---|---|---|
| Test set (CCLE / TCGA) | 38 / 339 | 111 / 1070 |
| Overall kNN@5 | 100.0% (Wilson CI 90.8–100.0%, n=38) | 78.4% (Wilson CI 69.8–85.0%, n=111) |
| kNN@1 / kNN@3 / kNN@10 | — | 75.7% / 78.4% / 79.3% |
| Silhouette | +0.57 | +0.70 |
| TFS (composite) | 0.89 | 0.82 |
| Random baseline | 33.3% | 6.7% |
| PCA+kNN baseline | 65.8% | 25.2% |

Weakest lineages (test kNN@5): LGG 0.0% (n=3), READ 0.0% (n=2), LUSC 25.0% (n=4), BLCA 60.0% (n=5),
GBM 66.7% (n=9), LUAD 66.7% (n=12) — every one of these is either a named confusable pair or the
smallest cohorts in the set. Strongest: HNSC/KIRC/LIHC/SKCM all 100.0%.

Per-lineage cluster tightness (mean within-lineage cosine, pooled test embeddings): 0.876 (GBM,
lowest) to 0.981 (LIHC, highest) — every lineage stays well above 0, i.e. no lineage collapses into
noise even at 15-way scale.

## Next Up

- Day 20: tumour-purity confounder analysis — rule out that the alignment axis is secretly a
  purity axis (pure cell line vs. stroma-contaminated tumour) rather than cancer identity.
- Assemble TCGA ESTIMATE/ABSOLUTE purity scores keyed by barcode; assign CCLE lines purity ≈ 1.0.
- Purity-stratified retrieval (high- vs. low-purity TCGA halves) and purity-residualised
  silhouette, per `PLAN-phase2.md`'s three-part Day 20 analysis.
