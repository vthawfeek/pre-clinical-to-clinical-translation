# Day 26: Drug-Response Transfer Positioning (CODE-AE)

**Date:** 2026-07-03
**Commit:** `day 26: drug-response-transfer probe + CODE-AE positioning`

## What Was Built

- `pctrans/casestudy/braf_vemurafenib.py`: `drug_signal_retained(embeddings, raw_expr, braf_status,
  vemurafenib_auc, n_splits=5, seed=0)` — within-CCLE, k-fold cross-validated out-of-fold R²/Spearman
  ρ for predicting vemurafenib AUC from three feature blocks: BRAF status alone (1-d indicator), raw
  2,000-gene HVG expression, and the 64-d contrastive embedding. Helper `_cv_r2_spearman` fits a
  per-fold `RidgeCV` (standardised features, built-in alpha sweep) rather than an unregularised or
  single-alpha model, so the high-dimensional raw-expression block (2,000 features, ~33 training rows
  per fold) doesn't blow up.
- `pctrans/evaluation/viz.py`: `braf_casestudy_panel` gains an optional `drug_signal_result` argument
  — when supplied, the static Figure F6 grows a third panel (grouped bar chart of R²/Spearman ρ per
  feature block) instead of the Day 23 two-panel layout. Backward compatible: omitting the argument
  reproduces the exact Day 23 figure (verified by the existing two-axes test still passing unchanged).
- `pctrans/scripts/casestudy_analysis.py`: extended to also (a) load `ccle_2k.parquet` raw HVG
  expression for the 41 SKCM lines with a vemurafenib readout, run `drug_signal_retained`, and print a
  Day 26 results block; (b) fit a descriptive-only `ElasticNetCV` (CCLE raw expression → AUC) and
  apply it to the 65 TCGA-SKCM patients' raw expression (`tcga_2k.parquet`) as a proximity-free
  reference point (`_ccle_to_patient_reference`, local to the script — explicitly not a validated
  metric, since no ground-truth patient AUC exists); (c) writes the new
  `reports/drug_transfer_positioning.json`; (d) passes the probe result into the Figure F6 call.
- `tests/test_casestudy.py`: `test_drug_signal_probe_runs_and_bounds` (synthetic planted-signal check:
  embedding arm recovers ρ>0.8/R²>0.5 when the target is generated from the embedding) and
  `test_drug_signal_probe_requires_matching_row_counts`.
- `tests/test_viz.py`: `test_braf_casestudy_panel_adds_third_axis_with_drug_signal_result`.
- `reports/preprint-outline.md`: new Results §4.9 (drug-signal-retained probe), a new evidence-table
  row, a "Relation to CODE-AE (Day 26 update)" paragraph in §5 Discussion summarising CODE-AE's
  deconfounding-autoencoder + drug-supervised design and why a lineage-supervised, drug-agnostic
  embedding is not expected to transfer to drug response, a new Limitations bullet on the probe's own
  small-n power, an updated Figure F6 description, and the Day 26 checklist item marked done.

## What Was Learned

- The embedding does **not** clearly underperform raw expression at this task: both are weak/negative
  out-of-fold R² (embedding −0.333, raw expression −0.041) at n=41. That rules out the cleanest
  story — "alignment selectively destroyed drug-response signal that raw expression retained" — since
  raw expression itself doesn't recover the signal either. The honest reading is *inconclusive on
  information loss*, not a rescue of the Day 23 Part-B null.
- A single categorical driver call (BRAF status alone, R²=+0.226) nominally beat both continuous
  high-dimensional regressions. With n=41 and folds of ~8 held-out samples, this is exactly the regime
  where a 1-parameter model can out-generalise a 2,000- or 64-dimensional one even with ridge
  shrinkage — a useful illustration of why the case study's sample size, not the representation, is
  the binding constraint.
- None of the three CV blocks reached significance (p=0.24–0.42), so this probe cannot itself resolve
  the Part-B question — it can only rule out the strongest version of the information-loss story.
  That distinction (ruling out one explanation vs. confirming another) is worth stating explicitly
  rather than letting a negative-but-not-significant R² read as a confirmed null.
- The ElasticNet CCLE→patient reference (only 7 of 2,000 genes selected, so not a meaningfully
  interpretable gene signature at this n) predicted patient AUC entirely inside the CCLE training
  range — a weak sanity check that the model isn't extrapolating wildly, but explicitly not a
  validated prediction since no ground-truth patient AUC exists to score against.

## Key Decisions

- Used per-fold `RidgeCV` (not `ElasticNetCV`) inside `drug_signal_retained` for the reusable,
  tested probe: ridge tolerates n≪p without a sparsity search per fold, keeping the CV loop fast and
  convergence-warning-free at 5 folds × 3 blocks. `ElasticNetCV` was reserved for the one-shot,
  non-cross-validated CCLE→patient reference in task 2, where its sparsity (7 nonzero genes) is a
  useful, human-scale description of what the reference model leaned on — a property RidgeCV's dense
  coefficients wouldn't offer for that illustrative use.
- Extended the existing Day 23 `pctrans-casestudy-analysis` CLI in place rather than adding a new
  `pctrans-drug-transfer` entry point. The Day 26 probe reuses the exact same loaded table, cell-line/
  patient frames, and `with_auc` subset Day 23 already computes, and both days write into the same
  Figure F6 — a second script would either duplicate that loading/filtering logic or require passing
  intermediate state through a file, neither of which the plan's "add the R²-retained panel to Figure
  F6" wording implied.
- Made `drug_signal_result` an optional keyword (default `None`) on `braf_casestudy_panel` rather than
  a required argument or a new function, so the Day 23 two-panel call site (and its test) needed zero
  changes — confirmed by re-running `test_braf_casestudy_panel_returns_two_axes` unmodified.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
........................................................................ [ 53%]
..............................................................           [100%]
134 passed, 6 deselected, 1 warning in 30.82s
```

Real end-to-end run (`uv run python -m pctrans.scripts.casestudy_analysis`):

```
============================================================
   DAY 23 -- BRAF PLACEMENT + VEMURAFENIB RESPONSE LINK
============================================================
Part A: median dist mutant=0.840 vs WT=0.870  (n_mut=47, n_wt=14)
        Mann-Whitney p=0.04724, effect size=0.649 (95% CI [0.465, 0.834])
Part B: Spearman rho(proximity, vemurafenib AUC) = 0.209 (95% CI [-0.109, 0.493]); p=0.19; n=41 (33 mutant / 8 WT)
============================================================

============================================================
   DAY 26 -- DRUG-SIGNAL RETAINED (within-CCLE CV, n=41)
============================================================
  BRAF status alone    R^2=+0.226  rho=+0.130  p=0.416
  raw HVG expression   R^2=-0.041  rho=-0.167  p=0.297
  64-d embedding       R^2=-0.333  rho=-0.187  p=0.241
  CCLE-to-patient ElasticNet reference: predicted patient AUC 0.780 +/- 0.107 (range [0.560, 1.073]);
  training range [0.560, 1.502] -- descriptive only, no ground truth
============================================================
Wrote reports\braf_vemurafenib.png
Wrote reports\braf_vemurafenib.html
Wrote reports\braf_casestudy.json
Wrote reports\drug_transfer_positioning.json
```

## Numbers

| Feature block | R² (out-of-fold) | Spearman ρ | p | n | folds |
|---|---|---|---|---|---|
| BRAF status alone | +0.226 | +0.130 | 0.42 | 41 | 5 |
| Raw HVG expression (2,000-d) | −0.041 | −0.167 | 0.30 | 41 | 5 |
| 64-d contrastive embedding | −0.333 | −0.187 | 0.24 | 41 | 5 |

ElasticNet CCLE→patient reference: trained on 41 CCLE SKCM lines (AUC range [0.560, 1.502], 7 of
2,000 genes selected), applied to 65 TCGA-SKCM patients: predicted AUC 0.780 ± 0.107, range
[0.560, 1.073] (inside training range; descriptive only).

## Next Up

- Day 27: fill `reports/preprint-outline.md` into a full manuscript draft (`reports/preprint-draft.md`
  + figures), inserting the Day 25 Celligner numbers into §4.7/Table T3 and the Day 26 probe into
  §4.9 (done today, verify it reads well in the full draft), finalise Abstract numbers, run the
  pre-submission checklist.
- `/blog-draft 27` → `reports/blog-03-validation.md` ("I Got 100% Accuracy. Then I Tried to Break
  It.") covering the full stress-test arc including today's honest inconclusive drug-signal probe.
- Draft LinkedIn Post 3 + X thread anchored on the permutation-null or 15-lineage confusion figure.
- Final full-suite run + push; Phase 2 complete.
