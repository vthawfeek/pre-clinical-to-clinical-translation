# Day 12: Streamlit App + Pre-Computed Embeddings

**Date:** 2026-07-02
**Commit:** `day 12: Streamlit app, pre-computed embeddings, blog-02 draft, LinkedIn-02 draft`

## What Was Built

- **`app/streamlit_app.py`** — the interactive demo. A sidebar dropdown selects a CCLE
  cell line (grouped by lineage: LUAD, BRCA, SKCM, labelled with the real cell-line name
  and ModelID). Three main panels:
  - **Live UMAP** — the pooled test-set manifold (38 CCLE + 339 TCGA) with every point
    dimmed for context; the selected cell line is drawn as a gold-ringed star and its five
    nearest TCGA patients as ringed hexagons. Colour = lineage, marker = domain.
  - **TFS gauge** — a 0-1 dial coloured on the fidelity bands (green > 0.70, yellow
    0.50-0.70, red < 0.50) with a threshold mark at 0.70, plus a plain-language caption.
  - **Nearest patient neighbours table** — rank, TCGA sample ID, lineage, cosine similarity,
    tumour stage, and histology.
  - Footer with the GitHub link, the dual-tower architecture one-liner, and the method summary.
  All data/plot logic lives in importable pure functions (`load_bundle`, `pooled_umap`,
  `nearest_patients`, `build_umap_figure`, `build_tfs_gauge`, ...); the UI in `main()` is
  guarded by `if __name__ == "__main__"` so the module imports cleanly for testing.
- **`pctrans/scripts/precompute.py`** (`pctrans-precompute` CLI, registered in `pyproject.toml`)
  — embeds every CCLE cell line with the frozen `CCLEEncoder` and writes:
  - `data/processed/ccle_embeddings.npz` — the full catalogue (259 cell lines x 64-d:
    `z`, lineage code `y`, ModelID `ids`, display `names`).
  - `data/processed/app_meta.json` — deploy-safe metadata distilled once from the raw files:
    ModelID -> stripped cell-line name (all 259) and each TCGA test patient -> {stage, histology}
    (339 patients). This lets the app show real names and a clinical column without shipping
    `data/raw/`.
- **`data/processed/ccle_embeddings.npz`**, **`data/processed/app_meta.json`** — generated
  artefacts (git-ignored under `data/`; regenerable via `pctrans-precompute`).

## What Was Learned

- The strong Gate 1 result means the honest edge case the plan asked for (a cell line with
  TFS < 0.40) does not exist in the held-out set: the lowest test TFS is CALU6 / ACH-000264
  at **0.662** (the Day 11 anaplastic-NSCLC outlier), which lands in the yellow "moderate"
  band, not red. The gauge's red band was instead verified against a synthetic 0.35 value.
- Nearest-neighbour retrieval in the app is a dot product, not a distance search: the saved
  embeddings are already L2-normalised by the dual tower, so cosine similarity is `z_tcga @ z_i`
  directly. Top-1 cosine similarities are modest in absolute terms (0.54-0.65) even for 5/5
  correct-lineage neighbours, a reminder that the manifold separates lineages by *relative*
  angle, not by packing each cluster into a tight cone.
- 334 of 339 TCGA test patients carry an AJCC tumour stage in the Xena phenotype table; the
  five blanks degrade to "-" in the table rather than breaking the row.
- Streamlit 1.58 has deprecated `use_container_width`; switched all charts/tables to
  `width="stretch"` to keep the app warning-free and current.

## Key Decisions

- **The app runs on the held-out *test* embeddings, not the full catalogue.** `embeddings_test.npz`
  (Day 11) is the evaluated subset that has per-cell-line TFS from Gate 1, so the UMAP,
  neighbours, and gauge are all backed by honest, held-out numbers. `ccle_embeddings.npz`
  (all 259 cell lines) is produced per the plan as the deployment/precompute deliverable and
  kept for future full-catalogue expansion.
- **A separate `app_meta.json` instead of reading `data/raw/` at serve time.** Streamlit Cloud's
  free tier cannot hold the ~500 MB raw expression file, and the plan requires the app to load
  only small precomputed artefacts. Distilling names + clinical annotation into one small JSON
  keeps the deployed app dependent solely on `data/processed/` files.
- **`pctrans-precompute` as a first-class CLI** rather than an ad-hoc script, matching the
  project's CLI-driven style and keeping the artefacts reproducible from a fresh clone.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!
$ uv run ruff check app/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
.........................................................                [100%]
57 passed, 4 deselected in 35.53s

$ uv run python -m pctrans.scripts.precompute
Wrote data\processed\ccle_embeddings.npz  (259 cell lines, dim 64)
Wrote data\processed\app_meta.json  (259 names, 339 TCGA annotations)
CCLE cell lines per lineage: {'LUAD': 80, 'BRCA': 69, 'SKCM': 110}

# Streamlit AppTest (in-process render, no browser) — 3 cell lines + edge case
bundle: CCLE (38, 64) TCGA (339, 64) tfs entries 38
LUAD ACH-000021 name=NCIH1693 TFS=0.819 match=5/5 topcos=0.540 stage='Stage IB'
BRCA ACH-000097 name=ZR751   TFS=0.823 match=5/5 topcos=0.599 stage='Stage IIB'
SKCM ACH-000219 name=A375    TFS=0.856 match=5/5 topcos=0.651 stage='Stage IIIC'
lowest TFS: ('ACH-000264', 0.662) color #f0a500 band Moderate fidelity
synthetic 0.35: #d62728 Poor fidelity
render clean across 4 selections; subheaders: ['Live UMAP', 'Translational Fidelity', 'Nearest human patient neighbours']
```

## Numbers

| Item | Value |
|---|---|
| Cell lines in `ccle_embeddings.npz` | 259 (LUAD 80 / BRCA 69 / SKCM 110), 64-d |
| App dropdown options (test cell lines) | 38 |
| TCGA patients in the app manifold | 339 (test) |
| TCGA patients with a tumour stage | 334 / 339 |
| Per-cell-line TFS scores available | 38 (from `reports/eval_summary.json`) |
| Lowest test TFS | CALU6 / ACH-000264 = 0.662 (moderate) |
| Highest test TFS (verified sample) | A375 / ACH-000219 = 0.856 (SKCM) |
| UMAP pooled points | 377 (38 CCLE + 339 TCGA) |
| App render check | AppTest, 0 exceptions across 4 selections |

## Next Up

- Day 13: complete the 5 `docs/` files (data pipeline, feature engineering, architecture,
  training, evaluation).
- Day 13: raise test coverage to >= 80% — add `test_inference.py`, `test_scripts.py`, and
  extend model/loss/eval tests.
- Day 13: write `README.md` with the results table, architecture diagram, and quick start.
- Day 13: run the coverage report (`pytest --cov=pctrans --cov-report=term-missing`) and note
  coverage per module in the daily report.
