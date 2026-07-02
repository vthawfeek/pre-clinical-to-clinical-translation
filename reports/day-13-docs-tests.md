# Day 13: Documentation, Tests & README

**Date:** 2026-07-02
**Commit:** `day 13: all 5 docs complete, test coverage 85%, README with results table`

## What Was Built

- **`docs/01_data_pipeline.md`** — CCLE (DepMap 24Q4 Figshare) + TCGA (UCSC Xena
  Pan-Cancer S3) URLs and filenames, step-by-step gene-ID harmonisation (strip the
  `" (ENTREZ)"` suffix from CCLE columns; TCGA already HUGO), and the union-rank HVG
  algorithm with a worked top-5 example.
- **`docs/02_feature_engineering.md`** — why z-score *after* log1p, the
  lineage-stratified 70/15/15 split, and the leakage boundary (scalers fit on pooled
  train only, frozen for val/test/inference).
- **`docs/03_architecture.md`** — encoder layer diagram with dimensions, BatchNorm
  justification (small-N/large-N domain imbalance), L2 unit-sphere rationale, ~5.5M
  param count, and why the towers stay asymmetric even with harmonised features.
- **`docs/04_training.md`** — the 5-equation SupCon-InfoNCE derivation, learnable
  `log(1/τ)` temperature rationale, the stratified batch-construction math
  (48 → 8 CCLE + 8 TCGA per lineage), cosine-with-warmup LR schedule, early stopping.
- **`docs/05_evaluation.md`** — kNN@{1,3,5,10} table, silhouette interpretation, the
  TFS composite formula, UMAP `n_neighbors=15 / min_dist=0.1` choices, baseline
  comparison, and the hardest-cell-line note.
- **`README.md`** — tagline, ASCII architecture diagram, test-set results table
  (Random / PCA+kNN / Harmony / This Work), quick-start CLI walkthrough, programmatic
  `TranslationEmbedder` example, docs index, Streamlit demo link.
- **`pctrans/inference/api.py`** — implemented `TranslationEmbedder` (was a stub):
  loads a checkpoint + processed catalogue, `embed_cell_line` → `(1, 64)`,
  `query_patients` → k nearest TCGA patients over the precomputed gallery. Reuses the
  frozen train scalers (no re-fit at serve time).
- **`pctrans/inference/__init__.py`** — export `TranslationEmbedder`.
- **`pctrans/scripts/query.py`** — implemented the `pctrans-query` CLI (was a stub).
- **`tests/conftest.py`** — session-scoped `pipeline` fixture: builds synthetic
  lineage-separable processed data, runs `pctrans-train` for one epoch, writes
  `embeddings_test.npz` — the shared substrate for the end-to-end CLI tests.
- **`tests/test_inference.py`** (new) — checkpoint load, `(1, 64)` embed shape,
  unknown-id error, k-neighbour retrieval + k-clamping.
- **`tests/test_pipeline.py`** (new) — end-to-end `evaluate` / `precompute` / `query`
  (fast) and `visualize` (slow, UMAP) driving the real `main()` bodies.
- **`tests/test_scripts.py`** — extended to `--help`-smoke all 7 CLIs.
- **`tests/test_data.py`** — added download-client idempotency (skip-when-present) for
  CCLE + TCGA, and a `save_filtered` parquet/gene-list round-trip.

## What Was Learned

- **Typer command functions are directly callable.** `@app.command()` returns the
  undecorated function, so the tests call `evaluate.main(...)` etc. directly (real
  return values to assert on) rather than only through `CliRunner` — this is what let
  the script bodies get covered honestly instead of via `--help` alone.
- **BatchNorm makes single-sample inference valid.** `embed_cell_line` embeds one
  cell line at a time; that only works because `model.eval()` switches BatchNorm to
  running statistics, so a batch of 1 is fine. Worth stating explicitly in docs/03.
- **The remaining 15% uncovered is genuinely network/raw-file code.** `download.py`
  (26%), `preprocess.py` (14%), and the client `_download` bodies all require the
  multi-hundred-MB live downloads; they're covered by the `integration`-marked tests
  when the real files are present, not by unit tests.
- **UMAP's numba JIT dominates test time** (~55s for the one visualize test), so the
  UMAP-bearing pipeline test is marked `slow` and kept out of the fast quality gate
  while still counting toward the coverage run.

## Key Decisions

- **Build the checkpoint by actually running `pctrans-train` in the fixture** rather
  than hand-crafting a state_dict. It costs ~1s on the tiny synthetic data and means
  `train.py` (74 stmts) is exercised for real, lifting it from 0% → 99%.
- **Write `embeddings_test.npz` directly in the fixture (no UMAP)** so the fast
  `precompute`/`query`/inference tests don't pull in the numba JIT; only the explicit
  `slow` visualize test does.
- **Set the tiny model's `embed_dim = 64`** (with small `input_dim`/hidden dims) so
  the inference test asserts the literal `(1, 64)` shape from PLAN.md, not a
  scaled-down stand-in.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
74 passed, 5 deselected in 26.39s

$ uv run pytest tests/ --cov=pctrans --cov-report=term-missing -q
...
Name                               Stmts   Miss  Cover
----------------------------------------------------------------
pctrans\inference\api.py              53      0   100%
pctrans\scripts\query.py              15      1    93%
pctrans\scripts\train.py              74      1    99%
pctrans\scripts\evaluate.py          113      4    96%
pctrans\scripts\visualize.py         104      2    98%
pctrans\scripts\precompute.py         80     22    72%
pctrans\scripts\download.py           42     31    26%   (network-only)
pctrans\scripts\preprocess.py         44     38    14%   (raw-file-only)
pctrans\data\preprocessor.py         100     30    70%   (load_ccle/load_tcga need raw files)
...
TOTAL                               1214    181    85%
79 passed, 2 warnings in 63.69s
```

## Numbers (if applicable)

- **Test coverage: 85 %** (1214 statements, 181 missed) — target ≥ 80 % ✅
- **Tests: 79 passed** (74 in the fast gate; +5 slow/integration in the full run).
- Coverage jumped from **54 % → 85 %** on the day.
- Fully covered today (0% → high): `inference/api.py` 100 %, `scripts/train.py` 99 %,
  `scripts/evaluate.py` 96 %, `scripts/visualize.py` 98 %, `scripts/query.py` 93 %,
  `scripts/precompute.py` 72 %.
- Remaining gaps are download/preprocess/loader code that needs live data
  (`download.py` 26 %, `preprocess.py` 14 %).
- Docs: 5 files, ~1,000 lines total; README with the test-set results table
  (kNN@5 100 %, kNN@1 97.4 %, silhouette +0.57, TFS 0.89).

## Next Up

Day 14 — GitHub Release & Public Launch:

- Final code review: remove hardcoded paths, add `requirements.txt`
  (`uv pip freeze`), add `notebooks/colab_quickstart.ipynb`.
- Make the repo public; cut GitHub release **v0.1.0** with kNN@5 accuracy + demo
  links; add repo topics.
- Deploy the Streamlit app (Python 3.11, `app/streamlit_app.py`), commit the small
  `*_embeddings.npz` artefacts, verify 3 cell lines render.
- Publish Blog Post 1 + 2, schedule LinkedIn posts, post the X thread.
- Tag `v0.1.0` and verify a fresh-clone `uv run pctrans-evaluate` reproduces the Gate
  1 report.
