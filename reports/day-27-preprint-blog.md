# Day 27: Preprint Assembly + Blog Post 3

**Date:** 2026-07-03
**Commit:** `day 27: preprint draft assembled, blog-03 validation story, Phase 2 complete`

## What Was Built

- `reports/multiseed_baselines_panel.png` (Figure F2): the one figure the Phase-2 evidence chain was
  missing. Left panel is a strip plot of the 10 per-seed multi-seed kNN@5 values (`multiseed_results.json`)
  with the mean, 95% CI band, and the Gate-2 90% stability threshold marked. Right panel is a bar
  chart of random / PCA+kNN / Harmony / supervised-ceiling / contrastive on the identical 3-lineage
  test set (`baselines.json`), reusing the exact colour palette already established by
  `celligner_comparison_panel`/`permutation_null_panel` in `pctrans/evaluation/viz.py` so it reads as
  one figure family with F4/F7. Built from a small one-off script (not added to the package, since it
  is a single manuscript figure assembled from two already-committed JSON artefacts, not a reusable
  evaluation pipeline step).
- `reports/preprint-draft.md`: full manuscript draft (Abstract through Back matter, ~4,000 words) built
  out from `reports/preprint-outline.md`'s bullet skeleton into complete prose. Abstract numbers
  cross-checked against `reports/phase2-summary.md` and the Day 15/17/19/20/21/23/25/26 reports before
  being finalised. All seven planned figures (F1-F7) are referenced with their actual file paths;
  Tables T2 (evidence table) and T3 (reference-method comparison) are populated with real numbers, not
  placeholders.
- `reports/preprint-outline.md`: F2's description updated from "planned" to done with its file path;
  pre-submission checklist gained a Day-27 line marking the manuscript draft complete and noting the
  remaining externally-gated items (ORCID, venue selection, final citation pass) that only the author
  can complete.
- `reports/blog-03-validation.md`: "I Got 100% Accuracy. Then I Tried to Break It." (1,328 words, inside
  the 1,200-1,500 target). Walks through all seven Phase-2 controls in narrative order (multi-seed →
  real baselines/supervised ceiling → 15-lineage difficulty → purity confounder → label-shuffle →
  Celligner attempt → vemurafenib case study), reporting the supervised-ceiling near-tie and the
  vemurafenib Part-B null as plainly as the passing checks.
- `reports/linkedin-03.txt` and `reports/x-thread-03.txt`: LinkedIn Post 3 anchored on the label-shuffle
  permutation-null figure (`permutation_null.png`); X thread anchored on the 15-lineage confusion-matrix
  heatmap (`confusion_matrix_15.png`), per the Day 27 plan's instruction to use one of those two figures
  for each piece of content.

## What Was Learned

- Writing the full-prose manuscript surfaced that the outline's Day 25/26 additions (§4.7, §4.9) already
  contained essentially publication-ready numbers and framing; the Day 27 work was genuinely assembly
  and finalisation, not new analysis, exactly as the plan intended for this day.
- The evidence table reads differently in full-manuscript form than as a bullet list: laying out
  §4.1-4.9 as numbered subsections before the table forces every claim to be justified in prose first,
  which made the §4.3 "near-trivial margin over the supervised ceiling" result impossible to soften by
  placement, it has to lead the Results section on its own merits, which is where it belongs given the
  paper's stated purpose.
- F2 was the one figure never actually produced during Days 15-26, because Day 15's tasks were
  scoped to statistics and a JSON artefact, not a plot. Assembling the preprint is what exposed the
  gap; it is a good illustration of why a dedicated "assemble the manuscript" day catches
  documentation debt that day-by-day execution does not.

## Key Decisions

- Built F2 as a standalone script rather than a new `pctrans/evaluation/viz.py` function with a CLI
  entry point and tests. Every other figure in the paper (F1, F3-F7) is tied to a reusable evaluation
  pipeline step that gets re-run on future data; F2 is purely a manuscript-assembly figure reading two
  already-finalised JSON files with no future re-run path, so adding package code and a test suite for
  it would be scope creep beyond what Day 27's "export a manuscript-formatted draft + figures" task
  actually calls for.
- Kept `reports/preprint-outline.md` and the new `reports/preprint-draft.md` as separate files rather
  than overwriting the outline. The outline remains the working skeleton with its bullet-form
  reasoning and the section-by-section "what Day N contributed" annotations; the draft is the
  submission-formatted artefact. A reviewer or future collaborator benefits from both: the draft to
  read, the outline to see how each number was arrived at day by day.
- Did not attempt a fresh bibliographic search for exact Celligner/PRECISE/CODE-AE/SupCon/CLIP
  citation details (venue, page numbers, DOI). The outline's citations were assembled during the Day
  25/26 research pass on general knowledge of the field; flagged as an explicit unchecked
  pre-submission item rather than fabricating citation metadata that could be wrong.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
........................................................................ [ 53%]
..............................................................           [100%]
134 passed, 6 deselected, 1 warning in 32.80s
```

```
$ uv run python make_f2.py
wrote reports/multiseed_baselines_panel.png
mean=95.00 sd=3.39 ci=(93.15131578947367, 97.10526315789473)
```

No source code under `pctrans/` or `tests/` changed today; the quality gate reruns the unchanged Day-26
suite to confirm the writing-only day introduced no regressions.

## Numbers (if applicable)

- `reports/preprint-draft.md`: ~4,000 words across Abstract, Introduction, Related Work (§2), Methods
  (§3.1-3.8), Results (§4.1-4.9 + Tables T2/T3), Discussion (§5), Limitations (§6, 10 bullets),
  Conclusion (§7), Back matter, Figures (F1-F7), Tables (T1-T3), pre-submission checklist.
- `reports/blog-03-validation.md`: 1,328 words.
- Figure F2: multi-seed mean 95.0%, sd 3.4pp, 95% CI [93.2%, 97.1%] (10 seeds); baselines panel:
  random 33.3%, PCA+kNN 65.8%, Harmony 84.2%, supervised ceiling 97.1%, contrastive 100.0%.
- Total reports directory: 7 figures referenced in the manuscript (F1 `umap_before_after.png`, F2
  `multiseed_baselines_panel.png` new today, F3 `confusion_matrix_15.png`, F4 `permutation_null.png`,
  F5 `confounder_purity.png`, F6 `braf_vemurafenib.png`, F7 `celligner_comparison.png`), all present on
  disk and git-tracked.

## Next Up

- Phase 2 (Days 15-27) is complete. Optional future Phase 3, named explicitly out of scope in the
  Limitations section of both the preprint and `PLAN-phase2.md`: Rung 3 (external cohorts,
  cross-platform validation, patient-derived xenografts, single-cell RNA-seq) and, only with
  collaborators, the Rung 5 prospective/regulatory path.
- Before actual submission: register an ORCID, pick a target venue (bioRxiv now; PLOS ONE/PeerJ or a
  workshop like MLCB/ML4H for peer review), run a fresh citation check against primary sources, and
  verify the venue against DOAJ/COPE.
- If a numeric Celligner comparison becomes a submission requirement, build the package from GitHub
  source on a Colab/Linux runtime with R installed and rerun `pctrans-celligner-compare` unchanged, the
  comparison code path is already written and tested.
- A systematic (not single-case) drug-response transfer study would be the natural first Phase-3 slice,
  since Day 26 already narrows the open question to "is n=41 the binding constraint," which a larger
  drug panel or a multi-lineage case study could directly test.
