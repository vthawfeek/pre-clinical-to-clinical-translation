# 05 · Evaluation

The Gate 1 metrics: kNN retrieval, cross-domain silhouette, the composite TFS, the
UMAP settings, and how we beat the baselines.

Code: [`pctrans/evaluation/knn.py`](../pctrans/evaluation/knn.py),
[`silhouette.py`](../pctrans/evaluation/silhouette.py),
[`tfs.py`](../pctrans/evaluation/tfs.py),
[`viz.py`](../pctrans/evaluation/viz.py).
CLI: `pctrans-evaluate`, `pctrans-visualize`.

All numbers below are on the **held-out test split** (`splits.json` test IDs),
which the model never saw during training or checkpoint selection.

## 1. kNN retrieval — the primary metric

The clinical question made quantitative: *given a drug tested on this cell line,
are its nearest human analogues the right patient population?* For each CCLE test
cell line, find its `k` nearest TCGA test patients in the 64-d embedding space
(Euclidean on unit-norm vectors), take the **majority-vote lineage**, and check it
matches the cell line's true lineage. Averaged over all test cell lines = accuracy.

The core (`knn_accuracy_from_embeddings`) runs on plain arrays so it is unit-tested
on synthetic embeddings: random ⇒ ~0.33 (chance over 3 lineages), one-hot perfect ⇒
1.0 (`test_knn_accuracy_random_model`, `test_knn_accuracy_perfect_embeddings`).

### kNN@k table (test set)

| k | Accuracy |
|---|---|
| 1 | **97.4 %** |
| 3 | **100 %** |
| 5 | **100 %** |
| 10 | **100 %** |

Per-lineage kNN@5 is 100 % for LUAD, BRCA, and SKCM. `k = 5` is the gate metric;
the k=1 miss is a single hard cell line (see §5).

## 2. Silhouette score

`cross_domain_silhouette` computes the silhouette of the **pooled** CCLE+TCGA test
embeddings using **lineage** as the cluster label. It measures whether lineage
clusters are tight and well-separated *after* pooling both domains — i.e. did
alignment produce lineage structure that survives mixing cell lines with patients?

- Range `[−1, +1]`; `> 0` = good alignment, `≈ 0` = overlapping, `< 0` = wrong
  neighbourhood. Returns `0.0` for the degenerate single-lineage case.
- Perfect one-hot clusters score `1.0` (`test_silhouette_perfect_clusters`).

**Test silhouette = +0.57.** `silhouette_contributions` also returns the per-sample
silhouette, which feeds per-cell-line TFS.

## 3. TFS — the composite Translational Fidelity Score

kNN answers "are the neighbours the right lineage?" and silhouette answers "are the
clusters clean?". TFS blends both onto one `[0, 1]` scale:

```
TFS = 0.5 · knn_accuracy  +  0.5 · (silhouette + 1) / 2
```

kNN accuracy is already `[0, 1]`; the silhouette `[−1, 1]` is rescaled to `[0, 1]`
via `(s + 1)/2`. Worked check (`test_tfs_formula`):
`0.5·0.8 + 0.5·(0.4+1)/2 = 0.40 + 0.35 = 0.75`.

Interpretation bands: **> 0.70** high fidelity · **0.50–0.70** moderate ·
**< 0.50** poor (domain artefacts dominate).

**Overall test TFS = 0.89** — `0.5·1.00 + 0.5·(0.566+1)/2`.

The identical formula scores each cell line from *its* neighbour match fraction and
*its* silhouette contribution (`per_cell_line_tfs`), giving a ranked "which cell
lines translate best?" list.

## 4. UMAP hyperparameters

For the Day 11 figures (`umap_projection`), 64-d test embeddings are projected to
2-d with:

- **`n_neighbors = 15`** — balances local vs global structure. Lower over-fragments
  into tiny local clusters; higher washes out the lineage separation we want to
  *show*. 15 is UMAP's default and appropriate for a few-hundred-point test set.
- **`min_dist = 0.1`** — lets same-lineage points pack tightly so clusters read
  cleanly, while keeping the three lineages visibly apart.
- Fixed `random_state = 42` for reproducible layouts.

UMAP is used **only for visualisation** — never for any reported metric. Metrics are
computed in the full 64-d space.

## 5. Baselines & the Gate 1 decision

| Metric | Random | PCA + kNN | Harmony (lit.) | **This work** |
|---|---|---|---|---|
| kNN@5 accuracy | 33.3 % | 65.8 % | ~63 % | **100 %** |
| Silhouette | — | — | — | **+0.57** |
| TFS | — | — | — | **0.89** |

- **Random (33.3 %)** — chance over 3 balanced lineages; the floor.
- **PCA + kNN (65.8 %)** — the honest no-alignment baseline: PCA on pooled raw
  features then the same cross-domain kNN. It captures variance but does **not**
  close the domain gap, so it lands far below the trained towers
  (`_pca_knn_baseline` in `evaluate.py`).
- **Harmony (~63 %)** — literature reference for a classic batch-integration method
  (harmonypy is not a project dependency).

**Gate 1 threshold: kNN@5 ≥ 70 % ⇒ DEPLOY.** At 100 % the model clears it
decisively → **DEPLOY** (`_decision` bands in `evaluate.py`; summary written to
`reports/eval_summary.json`).

### Hardest cell line

Ranked by per-cell-line TFS, the lowest is **`ACH-000264` (Calu-6, LUAD, TFS
0.662)** — an anaplastic/undifferentiated NSCLC line whose expression profile sits
near the LUAD/other boundary, dragging its per-sample silhouette down (0.048) even
though its neighbour match fraction is still 0.8. It's the single k=1 miss and a
sensible "use with caution" flag rather than a model failure.
