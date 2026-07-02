# Day 11: UMAP Visualisation & TFS Analysis

**Date:** 2026-07-02
**Commit:** `day 11: UMAP visualisation, TFS per cell line, outlier analysis, evaluation notebook`

## What Was Built

- **`pctrans/evaluation/viz.py`** — implemented the three planned functions plus two static
  renderers:
  - `umap_projection(embeddings, n_neighbors=15, min_dist=0.1, n_components=2, seed=42)` — fits
    UMAP on the pooled CCLE+TCGA test embeddings; `n_neighbors` is clamped to `n-1` so it also
    runs on the tiny test/unit sets.
  - `lineage_domain_scatter(coords, lineage_labels, domain_labels, title, sample_ids, tfs_scores)`
    — interactive Plotly scatter, colour = lineage (LUAD blue / BRCA pink / SKCM brown), marker =
    domain (TCGA ● / CCLE ✕), hover shows sample ID + TFS. Accepts int-coded *or* string labels.
  - `tfs_ranking_bar(cell_line_ids, tfs_scores, top_n=10)` — horizontal bar of the top/bottom 10
    cell lines, bars coloured by TFS band (green >0.70, yellow 0.50–0.70, red <0.50).
  - `lineage_domain_scatter_static(...)` + `before_after_panel(...)` — matplotlib renderers for the
    publication PNGs (no `kaleido` binary needed).
- **`pctrans/scripts/visualize.py`** — `pctrans-visualize` CLI: loads `best_model.pt` + the test
  split, embeds both towers, saves `data/processed/embeddings_test.npz` (for the Day 12 app), and
  writes the UMAP (html+png), the before/after panel (png), and the TFS ranking (html). Prints
  per-lineage cluster tightness. Registered as a `[project.scripts]` entry point.
- **`reports/umap_test_set.html` / `.png`** — interactive + static UMAP of the 38 CCLE + 339 TCGA
  test embeddings.
- **`reports/umap_before_after.png`** — Blog-Post-2 panel: PCA of raw features (domain gap) beside
  UMAP of aligned embeddings (lineage clusters).
- **`reports/tfs_ranking.html`** — interactive top/bottom cell-line TFS bar.
- **`data/processed/embeddings_test.npz`** — pooled test embeddings + ids/labels for the app.
- **`notebooks/03_evaluation.ipynb`** — 5 sections: Gate-1 metrics + confusion matrix, interactive
  UMAP, TFS ranking table, biological outlier analysis (joins `Model.csv`), baseline comparison.
  Executes end-to-end.
- **`tests/test_viz.py`** (7 tests) + a `pctrans-visualize --help` smoke test in `test_scripts.py`;
  `matplotlib>=3.8` added to dependencies.

## What Was Learned

- **The "before" PCA is a pure domain axis.** In raw HVG space, PC-1 separates *cell line vs.
  patient*, not disease: all 38 CCLE lines sit at PC-1 ≈ −60…−85, every one of the 339 TCGA
  patients at PC-1 ≈ −30…+25, with zero overlap. After alignment the UMAP axes become lineage —
  the domain gap the project exists to close is visible in a single side-by-side panel.
- **The lowest-TFS line is biologically ambiguous, not a bug.** ACH-000264 is **Calu-6**, a
  textbook *anaplastic / undifferentiated* NSCLC line from a **metastatic pleural effusion**. It
  lacks clean lung-adenocarcinoma differentiation identity, so it lands on the lineage boundary
  (silhouette 0.048; 1 of its 5 nearest TCGA patients is off-lineage). Low TFS correctly *flags* an
  ambiguous line rather than hiding it — the honest-science moment for the blog.
- **The bottom BRCA lines are all triple-negative/basal** (HCC38, HCC1937, HCC70, MDA-MB-157).
  TCGA BRCA is ER+/luminal-dominated, so the basal programme pulls these to the near edge of the
  BRCA cloud — partial support for the plan's TNBC↔proliferative hypothesis (they never actually
  cross into LUAD; BRCA kNN@5 stays 100%).
- **Cluster-tightness ≠ silhouette.** By mean within-lineage cosine, **BRCA is tightest (+0.861)**,
  then SKCM (+0.802), then LUAD (+0.787) — so the plan's "SKCM is the tightest cluster" hypothesis
  is only half right: SKCM *is* tighter than LUAD, but BRCA is tightest overall. SKCM still wins on
  *silhouette* (best separation from the other lineages), which is why it tops the TFS ranking.

## Key Decisions

- **Static PNGs rendered with matplotlib, not plotly+kaleido.** Publication PNGs use
  `lineage_domain_scatter_static` / `before_after_panel`; interactive figures use Plotly (HTML).
  This avoids adding the `kaleido` binary (heavy, historically flaky on Windows) for one export
  path, and matplotlib is now an explicit dependency.
- **`umap_projection` shape test marked `slow`.** UMAP's first call triggers a ~25 s numba JIT; the
  fast quality-gate suite stays snappy while the projection is still covered under `-m slow` (and
  exercised for real by `pctrans-visualize`).
- **Embeddings persisted to `embeddings_test.npz`.** The Day 12 Streamlit app loads pre-computed
  embeddings instead of running model inference on CPU per request, as the plan specifies.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
.........................................................                [100%]
57 passed, 4 deselected in 20.19s

$ uv run pytest tests/test_viz.py -q -m slow
1 passed, 6 deselected, 1 warning in 36.18s   # umap_projection shape
```

`pctrans-visualize` (warnings filtered):

```
Test set: CCLE 38 + TCGA 339
Wrote data\processed\embeddings_test.npz
Wrote reports\umap_test_set.html
Wrote reports\umap_test_set.png
Wrote reports\umap_before_after.png
Wrote reports\tfs_ranking.html

Per-lineage cluster tightness (mean within-lineage cosine, pooled):
  LUAD: +0.787
  BRCA: +0.861
  SKCM: +0.802
```

`notebooks/03_evaluation.ipynb` executes end-to-end via `jupyter nbconvert --execute` (exit 0).

**Static UMAP (`reports/umap_test_set.png`, described):** three well-separated lineage clusters.
Brown SKCM top-left (patients) with the SKCM cell-line crosses just above; pink BRCA lower-left;
blue LUAD lower-right. Each domain forms a tight own-cluster, but every CCLE cross group sits
nearest its correct lineage's patient cloud — consistent with kNN@5 = 100% in the full 64-d space
(2-D UMAP exaggerates the residual cell-line/patient offset that the retrieval metric does not see).

## Numbers (if applicable)

| Metric | Value |
|---|---|
| Test set | CCLE 38 / TCGA 339 |
| kNN@5 / silhouette / TFS | 100% / +0.566 / 0.892 |
| UMAP coords | (377, 2), n_neighbors=15, min_dist=0.1, seed=42 |

**Per-lineage TFS (38 cell lines):**

| Lineage | n | mean TFS | min | max | tightness (cosine) |
|---|---|---|---|---|---|
| SKCM | 16 | 0.860 | 0.808 | 0.868 | +0.802 |
| BRCA | 10 | 0.818 | 0.799 | 0.840 | **+0.861** |
| LUAD | 12 | 0.814 | 0.662 | 0.838 | +0.787 |

- **Top 3:** ACH-000881, ACH-001522, ACH-001975 — all SKCM (~0.868); melanocyte-identity markers
  give a compact, well-separated cluster.
- **Bottom 3 (with hypothesis):**
  - **ACH-000264 / Calu-6 (LUAD, 0.662)** — anaplastic/undifferentiated NSCLC, metastatic pleural
    effusion; lacks LUAD differentiation identity → boundary embedding (silhouette 0.048).
  - **ACH-000276 / HCC38 (BRCA, 0.799)** — triple-negative/basal; diverges from ER+-dominated TCGA
    BRCA.
  - **ACH-000223 / HCC1937 (BRCA, 0.799)** — triple-negative/basal (BRCA1-mutant); same basal-edge
    effect.

## Next Up

- **Day 12 (Streamlit app + Blog Post 2):** build `app/streamlit_app.py` loading
  `embeddings_test.npz` — live UMAP with selected cell line ★ + nearest-5 TCGA patients ⬡, TFS
  gauge, nearest-neighbour table.
- Pre-compute **all** CCLE cell-line embeddings → `data/processed/ccle_embeddings.npz`.
- Draft `reports/blog-02-results.md` around the before/after panel + the Calu-6 outlier case study.
- Draft LinkedIn Post 2 ("irreconcilable → clustered by disease") with the static before/after image.
