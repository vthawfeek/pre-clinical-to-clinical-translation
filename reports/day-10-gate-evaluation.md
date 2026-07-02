# Day 10: kNN Retrieval Evaluation [GATE 1 — PASS, DEPLOY]

**Date:** 2026-07-02
**Commit:** `day 10: kNN evaluation, silhouette, TFS computation, Gate 1 outcome: DEPLOY`

## What Was Built

- **`pctrans/evaluation/knn.py`** — `knn_retrieval_accuracy(model, ccle_loader, tcga_loader, k=5)`
  embeds the frozen towers and scores cross-domain retrieval; the array-level
  `knn_accuracy_from_embeddings(...)` does the work (majority-vote kNN, per-lineage accuracy,
  3×3 confusion matrix, kNN@{1,3,5,10} table, per-CCLE match fraction for TFS) so it is testable
  on synthetic embeddings without a model. `embed_loader` shares the callback's embedding logic.
- **`pctrans/evaluation/silhouette.py`** — `cross_domain_silhouette(...)` (pooled CCLE+TCGA,
  clustered by **lineage** not domain) + `silhouette_contributions(...)` for the per-sample values
  the per-cell-line TFS consumes. Returns 0.0 when <2 lineages present.
- **`pctrans/evaluation/tfs.py`** — `translational_fidelity_score(knn, sil)` = `0.5·knn + 0.5·(sil+1)/2`
  and elementwise `per_cell_line_tfs(match_fraction, silhouette_contribution)`.
- **`pctrans/scripts/evaluate.py`** — `pctrans-evaluate` CLI: loads `best_model.pt` + test split,
  runs kNN + silhouette + TFS + the PCA-then-kNN no-alignment baseline, prints the Gate 1 report,
  applies the ≥70% DEPLOY threshold, writes `reports/eval_summary.json`.
- **`tests/test_evaluation.py`** (10 tests) + **`tests/test_scripts.py`** (1) — random-model kNN ≈ 33%,
  perfect-embeddings kNN = 100% with diagonal confusion matrix, silhouette = 1.0 on perfect clusters,
  TFS formula/range, per-cell-line TFS, end-to-end model path, and `pctrans-evaluate --help` exit 0.
- **`reports/eval_summary.json`** — full machine-readable Gate 1 output incl. per-cell-line TFS ranking
  (feeds Day 11 UMAP / outlier analysis and Day 12 Streamlit app).

## What Was Learned

- **Test-set retrieval is perfect at k≥3 (100% kNN@5), and 97.4% at k=1** — exactly one of 38 cell
  lines has a nearest TCGA patient of the wrong lineage, and majority vote over 5 neighbours corrects
  it. The confusion matrix is fully diagonal (LUAD 12 / BRCA 10 / SKCM 16). This clears the 70% gate
  by a wide margin, but it is a **small test set (38 CCLE cell lines)** — 100% means "0/38 majority
  errors", not a large-sample guarantee; reported honestly rather than as a headline accuracy.
- **SKCM is the *tightest* lineage on test (mean TFS 0.860), not the hardest** — the mirror image of
  Day 9, where one melanoma val cell line capped SKCM at 0.9375. On the held-out test split every
  SKCM line retrieves SKCM patients, consistent with melanoma's strong melanocyte-identity markers
  producing a compact cluster once that one val outlier is out of the picture.
- **The alignment genuinely beats the baselines:** PCA(50)+kNN with no alignment reaches 65.8% and
  Harmony sits at ~63% (literature) — both well above random (33.3%) because lineage signal survives
  even unaligned, but the contrastive dual-tower closes the remaining domain gap to 100%.
- **Silhouette +0.57 confirms it is real cohesion, not kNN luck** — pooled cross-domain, within-lineage
  samples are markedly more similar to each other than to other lineages. TFS composite = 0.89.
- **Lowest-fidelity cell line: ACH-000264 (LUAD), TFS 0.662** — the Day 11 outlier to investigate; its
  majority vote is still correct but a minority of its k=5 neighbours cross the lineage boundary.

## Key Decisions

- **DEPLOY.** Test kNN@5 = 100% ≥ 70% → PLAN.md's deploy path (Days 11–14). No debug protocol needed.
- **PCA+kNN computed live; Harmony taken from literature.** `harmonypy` is not a project dependency
  (adds a heavy install for one baseline number); the PCA no-alignment baseline is computed on the
  actual test features, and Harmony's ~63% is cited as the reference point the plan specifies.
- **Silhouette uses `euclidean` on L2-normalised embeddings.** On the unit hypersphere euclidean
  distance is a monotonic function of cosine distance, so the ranking (and sign) matches the
  cosine-metric intent while staying in scikit-learn's default fast path.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
..................................................                       [100%]
50 passed, 3 deselected in 11.22s
```

`pctrans-evaluate --model models/best_model.pt --data-dir data/processed/`:

```
Test set: CCLE 38 + TCGA 339

==============================================
           GATE 1 EVALUATION REPORT
==============================================
Overall kNN@5 Accuracy:  100.0%   (threshold: 70%)
Per-lineage kNN@5:
  LUAD:  100.0%
  BRCA:  100.0%
  SKCM:  100.0%
kNN@k table:  k=1:97.4%  k=3:100.0%  k=5:100.0%  k=10:100.0%
Silhouette Score:  +0.57   (> 0 = good alignment)
TFS (composite):   0.89    (> 0.6 = deploy)
Random baseline:   33.3%
PCA+kNN baseline:  65.8%
Harmony baseline:  ~63%  (literature)
==============================================
DECISION: DEPLOY   [PASS (>=70% -> deploy path, Days 11-14)]
==============================================
Wrote reports\eval_summary.json
```

## Numbers (if applicable)

| Metric | Value | Threshold / baseline |
|---|---|---|
| Test set size | CCLE 38 / TCGA 339 | — |
| Checkpoint | epoch 2 (`best_model.pt`), val kNN@5 0.9474 | — |
| kNN@1 / @3 / @5 / @10 | 97.4% / 100% / 100% / 100% | ≥70% gate |
| Per-lineage kNN@5 | LUAD 100% · BRCA 100% · SKCM 100% | 33.3% random |
| Confusion matrix | `[[12,0,0],[0,10,0],[0,0,16]]` (LUAD/BRCA/SKCM) | diagonal = perfect |
| Silhouette (lineage, pooled) | +0.566 | > 0 = aligned |
| TFS (composite) | 0.891 | > 0.6 = deploy |
| PCA+kNN baseline | 65.8% | — |
| Harmony baseline | ~63% (literature) | — |

**Per-cell-line TFS (38 cell lines):**

| Lineage | n | mean TFS | min TFS |
|---|---|---|---|
| SKCM | 16 | 0.860 | 0.808 |
| BRCA | 10 | 0.818 | 0.799 |
| LUAD | 12 | 0.814 | 0.662 |

- Top 3: ACH-000881 (SKCM, 0.868), ACH-001522 (SKCM, 0.868), ACH-001975 (SKCM, 0.867)
- Bottom 3: ACH-000223 (BRCA, 0.799), ACH-000276 (BRCA, 0.799), **ACH-000264 (LUAD, 0.662)**

## Next Up

- **Day 11 (UMAP + TFS analysis):** implement `pctrans/evaluation/viz.py` (UMAP projection,
  lineage/domain scatter, TFS ranking bar); render `reports/umap_test_set.{html,png}`.
- Build the **before/after** panel: PCA of raw test features (domain gap) vs. UMAP of embeddings.
- Investigate the low-TFS outlier **ACH-000264 (LUAD)** and the bottom BRCA lines — hypermutation,
  mis-classified primary disease, or extreme passage.
- Build `notebooks/03_evaluation.ipynb` (metrics table, UMAP, TFS ranking, baseline comparison).
