# Day 17: Real Baselines + Supervised Ceiling

**Date:** 2026-07-03
**Commit:** `day 17: Harmony/ComBat/Scanorama baselines, supervised cross-domain ceiling`

## What Was Built

- `pctrans/evaluation/baselines.py`: `pca_knn` (moved here from `evaluate.py` to
  remove duplication), `harmony_knn` (harmonypy batch-integration, domain =
  batch, then cross-domain kNN), `combat_knn` (inmoose `pycombat_norm`),
  `scanorama_knn` (Scanorama integration + dimred), and `supervised_ceiling`
  (logistic-regression lineage classifier trained on CCLE, scored cross-domain
  on TCGA test — the "how easy is the problem, really" number).
- `pctrans/scripts/baselines.py`: new `pctrans-baselines` CLI. Loads the same
  processed test split `pctrans-evaluate` scores, runs every baseline + the
  ceiling, folds in the contrastive model's kNN@5 (+ Wilson CI) from
  `reports/eval_summary.json` when present, and writes `reports/baselines.json`
  with `best_real_baseline`, `beats_best_baseline_by`, and
  `matches_supervised_ceiling` verdict fields.
- `pctrans/scripts/evaluate.py`: dropped the hardcoded `HARMONY_BASELINE = 0.63`
  literature constant and the duplicated `_pca_knn_baseline` helper; now
  imports `pca_knn`/`RANDOM_BASELINE` from `baselines.py` and points the Gate 1
  report at `pctrans-baselines` for the full real comparison.
- `pyproject.toml`: new `[project.optional-dependencies] baselines` extra
  (`harmonypy`, `scanorama`, `inmoose`) kept separate from `dev`.
- `.github/workflows/ci.yml`: `uv sync --all-extras` → `uv sync --extra dev`
  (see Key Decisions).
- `tests/test_baselines.py`: `test_supervised_ceiling_beats_random`,
  `test_baseline_knn_shapes_and_range` (every baseline returns `None` or a
  `[0,1]` accuracy), `test_harmony_knn_runs_for_real` (harmonypy must not
  silently skip).

## What Was Learned

- **Harmony (84.2%) clears PCA (65.8%) by ~18 points but sits far below the
  contrastive model's 100%** — real batch correction genuinely helps over "no
  alignment," but multi-positive contrastive InfoNCE recovers cross-domain
  lineage structure that a general-purpose integration method (not trained on
  a lineage objective) still misses.
- **The supervised ceiling is 97.1%** — a plain logistic regression trained
  only on CCLE and evaluated cross-domain on TCGA test, with *no* alignment
  step at all, is already almost as good as the contrastive model. This is the
  most important honest finding of the day: on the 3-lineage task, raw
  lineage signal is close to linearly separable across domains, so most of
  the "difficulty" this benchmark measures is retrieval structure (kNN@5 on
  unlabelled neighbours) rather than recoverability of the label itself. The
  contrastive model's real value-add here is the last ~3 points plus the fact
  that it needs no labels on the TCGA side to do it — the ceiling model
  is fully supervised cross-domain, which the contrastive model is not.
  This is exactly why Day 18's 15-lineage task matters: it is designed to
  reopen headroom this 3-lineage ceiling has nearly closed.
- **`harmonypy`'s default `nclust = round(N/30)` collapses to 1 on small test
  sets (N≈38) and crashes** (`sigma.astype` on a bare float — a latent bug in
  the library for tiny inputs). Worked around by forcing `nclust = max(2,
  min(round(N/30), 100))`; harmless on production-sized inputs where the
  library's own default already exceeds 1.
- **`scanorama` and `inmoose` ship no prebuilt wheels for this Python/OS
  combination** and need a C/C++ toolchain to build from source; both failed
  to install on the Windows dev box (`Microsoft Visual C++ 14.0 or greater is
  required`). Rather than force that dependency onto every contributor (and
  onto CI), both are gated behind an import guard that returns `None` — the
  ComBat/Scanorama rows in the local report and `baselines.json` honestly read
  "n/a (dep not installed)" instead of a fabricated number. Real numbers for
  both would need to be reproduced under `pip install pctrans[baselines]` on
  Colab/Linux (already the project's established secondary environment).

## Key Decisions

- **Moved `--all-extras` → `--extra dev` in CI.** Adding `scanorama`/`inmoose`
  to any extras group that `--all-extras` picks up means every CI run now
  attempts a from-source native build with no prebuilt wheel to fall back on —
  a real risk of breaking the pipeline on a dependency this project doesn't
  actually require for lint/test. Scoping CI to the existing `dev` extra keeps
  today's change additive: `harmonypy` (already installed, pure Python/PyTorch,
  no native build) still runs for real in CI; the two fragile deps stay
  strictly opt-in.
- **Implemented `combat_knn`/`scanorama_knn` against real libraries
  (`inmoose.pycombat.pycombat_norm`, `scanorama.correct`) rather than a
  hand-rolled reimplementation of ComBat.** ComBat's empirical-Bayes shrinkage
  has enough subtlety that a from-scratch port risks silently wrong numbers —
  worse for a rigor-focused phase than an honest "not installed here" gap that
  a Colab run can fill in.
- **`pctrans-baselines` is a separate CLI from `pctrans-evaluate`, not folded
  into it.** Harmony/ComBat/Scanorama are optional-dependency, slower to run,
  and orthogonal to the trained-model Gate 1 report; keeping them in their own
  command means `pctrans-evaluate` stays fast and dependency-light, while
  `pctrans-baselines` reads `eval_summary.json` to fold the contrastive number
  into the same table without recomputing it.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
........................................................................ [ 82%]
...............                                                          [100%]
87 passed, 6 deselected in 11.91s
```

Real run against the Phase-1 processed data (`data/processed/`) and trained
`models/best_model.pt`:

```
$ uv run pctrans-evaluate
Test set: CCLE 38 + TCGA 339
Overall kNN@5 Accuracy:  100.0%   (threshold: 70%)
  Wilson 95% CI:    90.8-100.0%  (n=38)
...
DECISION: DEPLOY   [PASS (>=70% -> deploy path, Days 11-14)]

$ uv run pctrans-baselines
Test set: CCLE 38 + TCGA 339
==================================================
      REAL BASELINES + SUPERVISED CEILING
==================================================
Random                33.3%
PCA+kNN               65.8%
ComBat+kNN             n/a (dep not installed)
Harmony+kNN           84.2%
Scanorama+kNN          n/a (dep not installed)
Supervised ceiling    97.1%  (CCLE train -> TCGA test, no alignment)
Contrastive (ours)   100.0%  (Wilson 90.8-100.0%)
  beats best real baseline by +15.8 pts
  matches supervised ceiling: False
==================================================
Wrote reports\baselines.json
```

## Numbers

| Method | Test kNN@5 |
|---|---|
| Random | 33.3% |
| PCA+kNN | 65.8% |
| ComBat+kNN | n/a (native build unavailable locally; see What Was Learned) |
| Harmony+kNN | **84.2%** |
| Scanorama+kNN | n/a (native build unavailable locally; see What Was Learned) |
| Supervised ceiling (CCLE→TCGA classifier) | 97.1% |
| Contrastive (ours) | **100.0%** (Wilson CI 90.8–100%, n=38) |

Contrastive beats the best real baseline (Harmony) by +15.8 points, and beats
the fully-supervised, no-alignment ceiling by +2.9 points while requiring no
TCGA labels to do it.

## Next Up

- Day 18: make lineage IDs config-driven, add `configs/data_15.yaml`, scale to
  ~15 lineages including confusable pairs (LUAD/LUSC, COAD/READ, GBM/LGG,
  HNSC), re-run preprocessing + training under the larger label set.
- Day 19: evaluate the 15-lineage model — per-lineage kNN@k, confusion matrix,
  and an honest read on whether the errors are biologically sensible.
- Day 20: tumour-purity confounder analysis (stratified retrieval +
  residualised silhouette).
- Day 21: label-shuffle permutation negative control.
