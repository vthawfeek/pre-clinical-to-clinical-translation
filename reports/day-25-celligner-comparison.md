# Day 25: Celligner Head-to-Head Benchmark

**Date:** 2026-07-03
**Commit:** `day 25: Celligner head-to-head benchmark on identical retrieval metric`

## What Was Built

- `pctrans/evaluation/celligner_compare.py` — `run_celligner(ccle_expr, tcga_expr)` (fits Celligner on
  the CCLE reference and aligns TCGA onto it, guarded by an import try/except like Day 17's
  ComBat/Scanorama), `retrieval_on_embedding(joint_emb, ccle_ids, tcga_ids, lineages, k=5, ...)`
  (scores any externally-produced joint embedding with the identical `knn_accuracy_from_embeddings` +
  `cross_domain_silhouette` functions our own model is scored with).
- `pctrans/scripts/celligner_compare.py` — `pctrans-celligner-compare` CLI. Loads the raw (unscaled)
  HVG-filtered CCLE/TCGA test matrices for both the 3-lineage and 15-lineage variants, attempts
  Celligner, and places the result next to the already-computed contrastive/PCA/Harmony/
  supervised-ceiling numbers pulled from `reports/eval_summary*.json` and `reports/baselines.json`.
  Writes `reports/celligner_comparison.json` + `reports/celligner_comparison.png`.
- `pctrans/evaluation/viz.py::celligner_comparison_panel` — Figure F7: one grouped bar chart per
  lineage variant, skipping any method with no real number rather than plotting a fabricated zero.
- `tests/test_celligner_compare.py` — `test_celligner_compare_skips_without_dep` (the plan's required
  test: `run_celligner` returns `None` cleanly, not a raised exception, when the dependency chain is
  absent) and `test_retrieval_on_embedding_matches_our_metric` (the plan's required test: on a fixed
  synthetic array, `retrieval_on_embedding` reproduces `knn_accuracy_from_embeddings` exactly), plus
  a shape/range test.
- `pyproject.toml` — registered the new CLI entry point; documented (but deliberately did **not**
  add) `celligner` in the `baselines` extra, since adding it would make the whole extra unresolvable
  by pip/uv (see below).
- `reports/preprint-outline.md` — filled §4.7 and the Table T2/T3 rows, updated the Discussion's
  Celligner paragraph, added a Limitations bullet, updated the F7/T3 figure descriptions, and checked
  off the Day 25 pre-submission-checklist item as "attempted, not obtained" rather than leaving it
  blank or falsely marking it done.

## What Was Learned

- **Celligner cannot be pip/uv-installed on *any* platform, not just this Windows dev box.** Its
  published PyPI release (1.1.0) declares a dependency on a package literally named `umap` — not
  `umap-learn` — which has no installable release on PyPI at all. `uv pip install celligner --dry-run`
  fails immediately with an unsatisfiable-dependency error before any platform-specific build step is
  even reached. This is a stronger, more universal blocker than the Day 17 ComBat/Scanorama case
  (those need a C/C++ toolchain, which *some* machines have); Celligner needs a hand-patched install
  or a from-GitHub-source build regardless of OS.
- The from-source install path (per the Celligner README) additionally requires R plus a bundled
  `mnnpy` build with no prebuilt wheel — confirmed neither `R` nor `Rscript` is on this machine's
  PATH, a second independent blocker on top of the packaging issue.
- Because the honest path here is "attempted, dependency unavailable" rather than "ran and lost" or
  "ran and won," the module is built so a reader who *does* have a working Celligner install (e.g. on
  Colab/Linux with R) can call `run_celligner` + `retrieval_on_embedding` directly and drop the number
  into the same `reports/celligner_comparison.json` structure — the comparison infrastructure is
  real and tested even though today's number is not.
- Framing check from the plan held up on inspection: since Celligner is unsupervised, a comparable
  number from it would validate the "coarse lineage is nearly trivial" reading already established in
  §4.3 (supervised ceiling 97.1%), not contradict it — so the missing number is a reporting gap, not a
  missing threat to the paper's central claim.

## Key Decisions

- **Did not add `celligner` to the `baselines` extra.** Unlike `scanorama`/`inmoose` (Day 17), which
  are legitimately installable on Colab/Linux and only fail on this dev box, `celligner`'s PyPI
  metadata is broken everywhere. Listing it in an installable extra would make `uv sync --extra
  baselines` fail unconditionally for every user, which is strictly worse than the status quo. Left a
  detailed comment pointing at the from-source workaround instead.
- **Raw (unscaled) expression as Celligner's input, not the project's z-scored features.** Celligner
  does its own centering/PCA-based alignment internally and its published usage takes TPM-like
  values; feeding it the same train-only-fit z-scored features the contrastive model consumes would
  double-apply normalisation assumptions it isn't designed for. `ccle_2k.parquet`/`tcga_2k.parquet`
  already store raw log2 HVG expression (confirmed by inspection), so the CLI reads those directly.
- **Skipped a 15-lineage Harmony/supervised-ceiling row.** Day 17's real-baseline sweep was 3-lineage
  only (`reports/baselines.json` has no 15-lineage counterpart), so the 15-lineage panel only shows
  random/PCA/contrastive — consistent with what actually exists rather than re-running Day 17's full
  baseline suite as an unplanned side quest.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
........................................................................ [ 54%]
...........................................................              [100%]
131 passed, 6 deselected, 1 warning in 15.97s

$ uv run pctrans-celligner-compare
==========================================================
      DAY 25 — CELLIGNER HEAD-TO-HEAD (identical kNN@5 metric)
==========================================================

3-lineage (CCLE 38 + TCGA 339):
  Random                33.3%
  PCA+kNN               65.8%
  Harmony+kNN           84.2%
  Celligner+kNN          n/a (dep not installed -- see module docstring)
  Supervised ceiling    97.1%
  Contrastive (ours)   100.0%  (Wilson 90.8-100.0%)

15-lineage (CCLE 111 + TCGA 1070):
  Random                 6.7%
  PCA+kNN               25.2%
  Celligner+kNN          n/a (dep not installed -- see module docstring)
  Contrastive (ours)    78.4%  (Wilson 69.8-85.0%)

==========================================================
Wrote reports\celligner_comparison.json
Wrote reports\celligner_comparison.png
```

## Numbers

| Method | 3-lineage kNN@5 | 15-lineage kNN@5 |
|---|---|---|
| Random | 33.3% | 6.7% |
| PCA+kNN | 65.8% | 25.2% |
| Harmony+kNN | 84.2% | n/a (Day 17 was 3-lineage only) |
| **Celligner+kNN** | **n/a — dependency unresolvable** | **n/a — dependency unresolvable** |
| Supervised ceiling | 97.1% | n/a (Day 17 was 3-lineage only) |
| Contrastive (ours) | 100.0% (Wilson CI 90.8–100.0%, n=38) | 78.4% (Wilson CI 69.8–85.0%, n=111) |

`uv pip install celligner --dry-run` fails with: *"Because umap was not found in the package registry
and all versions of celligner depend on umap>=0.1, we can conclude that all versions of celligner
cannot be used."* — confirmed independent of R/mnnpy availability.

## Next Up

- Day 26 (per the updated `PLAN-phase2.md`): drug-response transfer positioning against CODE-AE —
  `drug_signal_retained` probe (embedding vs. raw expression vs. BRAF-alone predictability of
  vemurafenib AUC) plus CODE-AE literature positioning in the Discussion.
- Day 27: assemble the preprint draft (`reports/preprint-draft.md`) from the now-more-complete
  `reports/preprint-outline.md`, and the Blog Post 3 / LinkedIn / X-thread content deliverables that
  were originally slated for Day 25 before the plan's Phase 2E extension.
- If a numeric Celligner numbers is ever required for submission: build it from GitHub source on a
  Colab/Linux runtime with R installed, then re-run `pctrans-celligner-compare` unchanged — the
  comparison code path is already written and tested, only the environment is missing.
