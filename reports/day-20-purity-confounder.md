# Day 20: Confounder Analysis — Tumour Purity

**Date:** 2026-07-03
**Commit:** `day 20: tumour-purity confounder analysis (stratified + residualised)`

## What Was Built

- `pctrans/data/tcga_client.py` — `download_purity()` + `PURITY_URL`/`PURITY_FILENAME`/`PURITY_MIN_BYTES`.
  Source: the TCGA PanCanAtlas "mastercalls" ABSOLUTE purity/ploidy table (Carter et al. 2012
  method), fetched from GDC's stable file-UUID endpoint. Keyed by `array` = sample barcode
  (`TCGA-OR-A5J1-01`), the same convention as our processed frame index, so no ID munging is
  needed at join time. Downloaded to `data/raw/tcga/TCGA_mastercalls.abs_tables_JSedit.fixed.txt`
  (10,786 samples, 901 KB).
- `pctrans/evaluation/confounders.py` — the three Day 20 analyses:
  - `load_purity(path, ids_ccle, ids_tcga)` — joins purity onto pooled sample IDs; CCLE cell lines
    get `CCLE_PURITY = 1.0` (pure monoculture), unmatched TCGA barcodes get `NaN`.
  - `domain_axis_purity_correlation(...)` — Pearson r between each sample's projection onto the
    CCLE→TCGA centroid axis and its purity.
  - `purity_stratified_knn(...)` — cross-domain kNN@k recomputed separately within TCGA
    high-/low-purity halves (split at the median finite purity); CCLE anchor set unchanged in
    both (no "low-purity cell line" stratum exists by construction).
  - `residualise_purity(...)` / `purity_residualised_silhouette(...)` — per-dimension OLS
    regression of purity out of pooled embeddings, then the lineage silhouette on the residuals.
- `pctrans/evaluation/viz.py` — `purity_confounder_panel(...)`: two-panel figure (domain-axis
  projection vs. purity scatter; kNN@5 bar per stratum with a reference line).
- `pctrans/scripts/confounders.py` (`pctrans-confounders` CLI) — loads the Day 11
  `embeddings_test.npz`, downloads/reads the purity table, runs all three analyses, prints the
  report, writes `reports/confounder_purity.json` + `reports/confounder_purity.png`.
- `pctrans/scripts/download.py` — `purity` subcommand (`pctrans-download purity`) for consistency
  with the existing `ccle`/`tcga` download commands.
- `tests/test_confounders.py` (10 tests), `tests/test_data.py` (+1 idempotency test),
  `tests/test_viz.py` (+1 panel test).

## What Was Learned

- **The domain axis is a moderate, not dominant, purity axis.** `r = -0.455` (n=333) — real but far
  from `r ≈ -1`. The scatter (`reports/confounder_purity.png`, left panel) shows why: samples fall
  into three horizontal bands (one per lineage) with only mild within-band purity structure — the
  dominant source of variance along that axis is lineage identity, not purity, which is exactly
  the shape we want.
- **Retrieval is saturated, so the stratified test is a weaker probe than it looks.** Both purity
  halves score kNN@5 = 100.0% (n=153 / n=142). That is consistent with "purity doesn't break
  retrieval," but the 3-lineage test set was already at 100% unstratified (Day 10) — there is no
  headroom left for a stratification to fail on. The 15-lineage test set (Day 19, kNN@5 78.4%) would
  be the sharper version of this test; scoping it there is flagged as a Day 20+ follow-up rather
  than done today, since the plan's Day 20 experiment is defined against the 3-lineage
  `embeddings_test.npz` artefact.
- **Purity residualisation costs some silhouette, but lineage cohesion clearly survives.**
  +0.566 → +0.500 (a 0.066 drop, ~12% relative). If the alignment were "secretly" a purity effect,
  removing purity should have collapsed the silhouette toward zero; instead it stays strongly
  positive.
- **ABSOLUTE purity coverage isn't universal.** 295/339 (87%) TCGA test patients have a called
  purity estimate; the other 44 have no ABSOLUTE solution (typically low-tumor-content or QC-failed
  samples) and are dropped from every purity-conditioned analysis rather than imputed.

## Key Decisions

- **Used ABSOLUTE (not ESTIMATE) as the purity source.** The plan allows either. ABSOLUTE's
  PanCanAtlas mastercalls table is a single, stably-hosted, unrestricted file (GDC file UUID) keyed
  by the exact sample-barcode format our processed frames already use — no aliasing needed. Verified
  by direct download + join-coverage check (2,041/2,264 of all downloaded TCGA samples matched)
  before writing any analysis code.
- **Kept the analysis scoped to the 3-lineage `embeddings_test.npz` artefact**, not a fresh
  15-lineage embedding run. The plan's Day 20 tasks describe joining purity onto "the test
  embeddings" — the artefact that name unambiguously refers to is the one Day 11 already produces
  and the Streamlit app already consumes. Re-deriving 15-lineage purity-stratified embeddings is
  legitimate future work, noted above, not silently substituted for the specified scope.
- **CCLE purity is a hardcoded constant (1.0), not a fitted or looked-up value.** Cell lines are
  monocultures — there is no stromal/immune contamination to estimate — so "purity" isn't a
  measured quantity for them; treating it as ground-truth 1.0 is the well-established convention in
  the tumour-purity literature (Aran et al. 2015 uses the same assumption for cell-line controls).

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
........................................................................ [ 63%]
..........................................                               [100%]
114 passed, 6 deselected in 28.22s
```

```
$ uv run pctrans-confounders
Loaded data\processed\embeddings_test.npz: CCLE 38 + TCGA 339
TCGA patients with an ABSOLUTE purity call: 295/339

====================================================
       DAY 20 — TUMOUR-PURITY CONFOUNDER REPORT
====================================================
(a) corr(domain-axis projection, purity): r = -0.455  (n=333)
(b) Overall kNN@5 (unstratified): 100.0%
    high_purity : 100.0%   (n=153)
    low_purity  : 100.0%   (n=142)
(c) Silhouette before purity residualisation: +0.566
    Silhouette after purity residualisation:  +0.500
====================================================
Wrote reports\confounder_purity.png
Wrote reports\confounder_purity.json
```

## Numbers

| Metric | Value |
|---|---|
| ABSOLUTE purity table | 10,786 samples, purity range [0.08, 1.00], mean 0.634 |
| TCGA test patients with a purity call | 295 / 339 (87.0%) |
| domain-axis ↔ purity correlation | r = −0.455 (n = 333) |
| kNN@5, high-purity stratum | 100.0% (n = 153) |
| kNN@5, low-purity stratum | 100.0% (n = 142) |
| kNN@5, unstratified (reference) | 100.0% |
| Silhouette, before purity residualisation | +0.566 |
| Silhouette, after purity residualisation | +0.500 (Δ = −0.066) |

**Gate 2 (G2-4) read:** PASS — retrieval holds in both purity strata and lineage cohesion survives
residualisation; the domain-axis/purity correlation is real but moderate, and the embedding is not
primarily a purity axis.

## Next Up

- Day 21: label-shuffle permutation negative control — shuffle the CCLE↔TCGA lineage
  correspondence, retrain/re-evaluate, and show retrieval collapses to the ~33% chance level
  (empirical p < 0.01).
- Carry the purity-confounder JSON into the Day 24 Gate 2 evidence table (`G2-4`).
- (Follow-up, not blocking) consider re-running the purity stratification on the 15-lineage test
  set, where unstratified accuracy has real headroom (78.4%) to actually drop under stratification.
