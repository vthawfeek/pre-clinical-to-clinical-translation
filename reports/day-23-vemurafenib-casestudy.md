# Day 23: BRAF/Vemurafenib Placement + Response-Link Case Study

**Date:** 2026-07-03
**Commit:** `day 23: BRAF/vemurafenib placement + response-link case study`

## What Was Built

- `pctrans/casestudy/braf_vemurafenib.py` — new Part A / Part B analysis functions, added
  alongside the Day 22 data-assembly code in the same module (per the plan's module map):
  - `braf_mutant_patient_centroid` — mean embedding of BRAF-mutant TCGA SKCM patients.
  - `distance_to_centroid` — Euclidean distance of a batch of embeddings to a fixed centroid.
  - `pairwise_greater_fraction` — a common-language effect size: the fraction of
    `(greater_group, less_group)` pairs where the ordering actually holds (1.0 = perfect
    separation, 0.5 = none, ties count 0.5).
  - `_bootstrap_two_sample_ci` — generic two-sample percentile bootstrap (independent resampling
    of each group), used to attach a CI to the effect size.
  - `braf_placement_test` (**Part A**) — one-sided Mann-Whitney U test (mutant-line distances
    stochastically less than WT-line distances) + bootstrapped effect-size CI.
  - `spearman_with_ci` — paired-bootstrap CI around a Spearman rho.
  - `braf_response_link` (**Part B**) — correlates cell-line proximity to the Part-A centroid with
    vemurafenib AUC among the cell lines with a PRISM readout.
- `pctrans/evaluation/viz.py` — two new figure builders:
  - `braf_casestudy_panel` — static matplotlib two-panel figure (placement scatter + response
    scatter with OLS fit and bootstrap CI band), for the PNG export.
  - `braf_casestudy_panel_interactive` — plotly two-subplot equivalent, for the HTML export.
  - `_bootstrap_fit_band` — private helper: OLS fit line + bootstrap CI band over a grid.
- `pctrans/scripts/casestudy_analysis.py` — new `pctrans-casestudy-analysis` CLI: loads the Day 22
  `braf_vemurafenib.parquet`, runs Part A + Part B, prints a report, and writes
  `reports/braf_casestudy.json` + `reports/braf_vemurafenib.{png,html}`.
- `pyproject.toml` — registers `pctrans-casestudy-analysis`; adds `scipy` as an explicit
  dependency (previously only pulled in transitively via scikit-learn/umap-learn, but now
  imported directly for `mannwhitneyu`/`spearmanr`).
- `tests/test_casestudy.py` — `test_centroid_distance_math` (centroid + distance computation on
  known synthetic points, including a full `braf_placement_test` run with known ordering),
  `test_placement_test_requires_both_groups`, `test_response_correlation_runs_and_bounds` (rho
  bounds + CI keys via `spearman_with_ci`, plus an end-to-end `braf_response_link` run).
- `tests/test_viz.py` — `test_braf_casestudy_panel_returns_two_axes` and
  `test_braf_casestudy_panel_interactive_has_two_subplots`, matching this project's existing
  one-test-per-panel-builder convention.
- `notebooks/03_evaluation.ipynb` — Section 7 (renumbered from the plan's placeholder "Section 6",
  since Day 19 already claimed that number): headline numbers, Part A/B code cells, the
  interactive figure, and an honest "Reading the result" cell with caveats. Executed end-to-end
  via `jupyter nbconvert --execute` (exit 0, ephemeral `--with nbconvert` per the Day 4/19
  decision to keep it out of project dependencies).

## What Was Learned

- **Part A is real but weak, not the clean separation the lineage-retrieval numbers show.**
  BRAF-mutant SKCM cell lines are closer to the BRAF-mutant patient centroid than WT lines
  (Mann-Whitney p = 0.047, just under the conventional 0.05 threshold), but the bootstrapped
  effect-size 95% CI is `[0.465, 0.834]` — the lower bound sits right at "no effect" (0.5). The
  model captured *some* within-melanoma BRAF-linked structure, but it is a soft signal riding on
  top of the (very clean) lineage signal, not a sharp mutant sub-cluster the UMAP shows by eye.
- **Part B is a genuine null result, and the plan explicitly asked for it to be reported as
  such rather than buried.** Spearman rho(proximity, vemurafenib AUC) = 0.209, 95% CI
  `[-0.109, 0.493]` — not significant (p = 0.19), and the *sign* is even the opposite of the
  positive-control hypothesis (closer-to-mutant-centroid should mean lower/more-sensitive AUC;
  observed rho is positive). With n = 41 (33 mutant / 8 WT) this is not strong evidence of
  "no relationship" either — it is evidence that this analysis, at this N, cannot detect one.
- **Placement and response are separable questions, and Day 23 shows why that distinction
  matters.** A model can recover a driver-defined subgroup (Part A, weak positive) without that
  same geometry predicting a downstream functional phenotype (Part B, null) — vemurafenib
  sensitivity depends on more than BRAF status alone (NRAS co-mutation, MITF levels, feedback
  reactivation), none of which the embedding was ever trained to see.
- **The UMAP panel shows the domain gap dominates the local geometry even within one lineage.**
  Restricting to SKCM only, cell lines and patients still separate into two visually distinct
  UMAP-1 bands (as in the full 3-lineage embedding); BRAF status shows up as a *within-band*
  color mixture rather than a second axis of separation, consistent with the modest Part-A effect
  size.

## Key Decisions

1. **Effect size = a hand-rolled common-language statistic (`pairwise_greater_fraction`), not a
   library rank-biserial-correlation call.** It has a direct, testable interpretation ("fraction
   of WT-vs-mutant pairs where the WT line truly is farther away") and its exact behaviour under
   perfect separation / reversal / ties is trivial to hand-verify in
   `test_centroid_distance_math`, unlike relying on the sign convention of a particular
   `scipy.stats.mannwhitneyu` statistic.
2. **Proximity defined as `-distance`, correlated directly against raw AUC (not a derived
   "sensitivity" variable).** Keeps the Part-B computation to one un-transformed real-world
   quantity (PRISM AUC, lower = more sensitive) and pushes the sign interpretation into the
   printed/notebook text ("expected negative correlation") rather than inventing a second
   sign-flipped column that could get out of sync with the raw data.
3. **New `pctrans-casestudy-analysis` CLI instead of extending Day 22's `pctrans-casestudy`
   command.** Typer apps in this project use one `@app.command()` per script so the CLI is
   callable with no subcommand name; adding a second command to the existing app would have
   forced `pctrans-casestudy main`/`pctrans-casestudy analyze` naming and broken the Day 22
   report's documented invocation. A second script matches the Day 20/21 precedent (new analyses
   built on a prior day's assembled data get their own CLI).
4. **Notebook section labelled "Section 7", not "Section 6" as the plan's task list literally
   says.** Day 19 already added a "Section 6" to `notebooks/03_evaluation.ipynb`; the plan's Day
   23 text predates that numbering. Sequential numbering (7) matches the notebook's actual state
   rather than silently overwriting/duplicating Day 19's section number.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
........................................................................ [ 56%]
........................................................                 [100%]
128 passed, 6 deselected, 1 warning in 18.54s
```

Real run (`pctrans-casestudy-analysis`, on the Day 22 `braf_vemurafenib.parquet`):

```
Loaded data/processed/braf_vemurafenib.parquet: 61 cell lines + 65 patients (SKCM)

============================================================
   DAY 23 -- BRAF PLACEMENT + VEMURAFENIB RESPONSE LINK
============================================================
Part A: median dist mutant=0.840 vs WT=0.870  (n_mut=47, n_wt=14)
        Mann-Whitney p=0.04724, effect size=0.649 (95% CI [0.465, 0.834])
Part B: Spearman rho(proximity, vemurafenib AUC) = 0.209 (95% CI [-0.109, 0.493]); p=0.19; n=41 (33 mutant / 8 WT)
============================================================
Wrote reports\braf_vemurafenib.png
Wrote reports\braf_vemurafenib.html
Wrote reports\braf_casestudy.json
```

Notebook: `jupyter nbconvert --to notebook --execute --inplace notebooks/03_evaluation.ipynb`
(ephemeral `--with nbconvert`) — exit 0, all Section 7 cells produced their expected output with
no errors.

## Numbers

| Quantity | Value |
|---|---|
| SKCM cell lines with a BRAF call (Part A) | 61 (47 mutant / 14 WT) |
| SKCM patients with a BRAF call (centroid source) | 65 (32 mutant / 33 WT) |
| Part A — median distance to centroid, mutant lines | 0.840 |
| Part A — median distance to centroid, WT lines | 0.870 |
| Part A — Mann-Whitney p (one-sided, mutant < WT) | 0.0472 |
| Part A — effect size (95% CI) | 0.649 [0.465, 0.834] |
| SKCM cell lines with a vemurafenib AUC (Part B) | 41 (33 mutant / 8 WT) |
| Part B — Spearman rho (95% CI) | 0.209 [-0.109, 0.493] |
| Part B — p-value | 0.190 (not significant) |
| Bootstrap resamples (both parts) | 2,000 |

## Next Up

- Day 24: assemble all Phase-2 evidence (Days 15/17/20/21 + this case study) into
  `reports/phase2-summary.md`, extend `/gate-check` with the Gate 2 decision, and update
  README/CLAUDE headline claims — including reporting Day 23's Part A (weak positive) and
  Part B (null) results honestly rather than only the positive half.
- Day 25: Blog Post 3 — the "what survives rigorous validation" story, closing with this case
  study as the first (partial) tie to real drug response.
