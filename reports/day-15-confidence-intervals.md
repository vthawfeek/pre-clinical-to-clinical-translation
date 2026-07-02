# Day 15: Confidence Intervals & Multi-Seed Reproducibility

**Date:** 2026-07-02
**Commit:** `day 15: bootstrap + Wilson CIs, multi-seed reproducibility harness`

## What Was Built

- **`pctrans/evaluation/stats.py`** — the statistical-hardening toolkit:
  - `wilson_ci(successes, n, alpha=0.05)` — analytic score interval for a binomial
    proportion; stays inside [0, 1] and keeps a finite width at p = 1 (the 38/38 case).
  - `bootstrap_ci(values, statistic, n_boot=2000, seed=0)` — generic percentile
    bootstrap; resamples per-unit values with replacement, no distributional assumption.
  - `bootstrap_metric_ci(match_fraction, ...)` — convenience wrapper that turns the
    per-CCLE `match_fraction` from `knn.py` into a kNN point estimate + 95% CI (hit =
    `match_fraction >= 0.5`, i.e. a strict majority for k = 5).
  - `aggregate_seeds(values, ...)` — mean / sd / min / max + bootstrap CI of the
    across-seed mean, used by the multi-seed runner.
- **`pctrans/scripts/multiseed.py`** (`pctrans-multiseed`) — re-runs
  **split → fit scalers → train → test-eval** end-to-end for seeds 42–51, reusing the
  default training config with no per-seed tuning. Varying the split seed changes
  *which* cell lines land in the 38-line test set, so the sweep is the honest test of
  small-test-set stability. Writes `reports/multiseed_results.json` (per-seed rows +
  aggregate). Per-seed checkpoints go to a temp dir so `models/best_model.pt` is never
  touched.
- **`pctrans/scripts/evaluate.py`** — extended to print Wilson + bootstrap CIs next to
  every point metric and to store them in `reports/eval_summary.json`
  (`knn_wilson_ci`, `knn_bootstrap_ci`, `silhouette_bootstrap_ci`).
- **`tests/test_stats.py`** — 8 tests (Wilson known value, bounds within [0,1], zero-n,
  bootstrap contains point, CI shrinks with n, metric-wrapper perfect/threshold,
  seed aggregation). Added `pctrans-multiseed` to the CLI `--help` smoke test.
- **`pyproject.toml`** — new `pctrans-multiseed` entry point.
- **`reports/multiseed_results.json`** — 10-seed reproducibility record.

## What Was Learned

- **The "100%" is a stable ceiling, not a lucky split.** Across 10 independent splits,
  kNN@5 = 0.950 ± 0.034 with a bootstrap CI of [0.932, 0.971] and a floor of 89.5% —
  the single-split 100% sits at the top of a tight, well-separated band, comfortably
  above the future Gate-2 G2-1 bar (CI lower bound ≥ 0.90).
- **A bare 100% on n = 38 is genuinely wide.** The Wilson interval for 38/38 is
  90.8–100.0%: the honest reading is "≥ 91% with 95% confidence," which the report now
  prints instead of a naked "100%."
- **The point metric is far more stable than the geometry.** kNN@5 spread is tiny
  (sd 0.034) but silhouette swings a lot across seeds (0.566 → 0.859, sd 0.113). The
  canonical Phase-1 model early-stopped at the val-kNN-optimal epoch 2 (silhouette
  +0.57), whereas seeds that ran a few epochs longer tightened clusters (up to +0.86)
  — silhouette is sensitive to the exact stopping epoch in a way retrieval accuracy is not.
- **Bootstrap vs Wilson agree where they should.** At 38/38 the bootstrap CI is
  degenerate [1.0, 1.0] (every anchor is a hit, so every resample is 1.0), which is why
  the analytic Wilson interval — not the bootstrap — is the honest interval to quote for
  a perfect proportion.

## Key Decisions

- **Wilson over Wald for the retrieval proportion.** The normal-approximation (Wald)
  interval is degenerate at p = 1 and can leave [0, 1] for small n; Wilson is the
  standard fix and gives the expected 90.8% lower bound at 38/38.
- **The seed sweep holds the 2,000-HVG feature space fixed and varies only the split.**
  This isolates the "is the result split-dependent?" question. Per-split, train-only HVG
  *re-selection* is the Day-16 refinement and is deliberately not folded in yet, so
  today's numbers answer one question cleanly.
- **Reuse the default config verbatim per seed; best-checkpoint scoring.** No per-seed
  tuning (that would be a leak of its own), and each run is evaluated at its best-val
  checkpoint, not final-epoch weights — matching how the canonical model was selected.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
83 passed, 5 deselected in 14.37s

$ uv run pctrans-evaluate            # excerpt
Overall kNN@5 Accuracy:  100.0%   (threshold: 70%)
  Wilson 95% CI:    90.8-100.0%  (n=38)
  Bootstrap 95% CI: 100.0-100.0%
Silhouette Score:  +0.57   (> 0 = good alignment)  [boot 95% CI +0.55, +0.58]

$ uv run pctrans-multiseed --n-seeds 10 --seed-start 42   # summary
kNN@5       mean 0.950 +/- 0.034  CI [0.932, 0.971]  (min 0.895, max 1.000)
kNN@1       mean 0.945 +/- 0.029  CI [0.929, 0.963]  (min 0.895, max 1.000)
Silhouette  mean 0.742 +/- 0.113  CI [0.675, 0.804]  (min 0.566, max 0.859)
TFS         mean 0.910 +/- 0.038  CI [0.888, 0.933]  (min 0.848, max 0.965)
```

## Numbers

| Metric | Single split (seed 42, n=38) | 10-seed mean ± sd | 95% CI | Range |
|---|---|---|---|---|
| kNN@5 | 100.0% (Wilson 90.8–100.0%) | 0.950 ± 0.034 | [0.932, 0.971] | 0.895–1.000 |
| kNN@1 | 97.4% | 0.945 ± 0.029 | [0.929, 0.963] | 0.895–1.000 |
| Silhouette | +0.57 (boot +0.55, +0.58) | +0.742 ± 0.113 | [0.675, 0.804] | 0.566–0.859 |
| TFS | 0.89 | 0.910 ± 0.038 | [0.888, 0.933] | 0.848–0.965 |

Per-seed kNN@5: 100.0, 97.4, 94.7, 92.1, 89.5, 94.7, 94.7, 100.0, 94.7, 92.1 (%).
Per-seed best epoch ranged 2–10; each run early-stops well before the 30-epoch cap.

## Next Up

- Day 16 — refactor `preprocessor.py` to compute union-rank HVG variance on the **train
  slice only** (stratified split on IDs first), behind a `--hvg-on all|train` flag.
- Re-run preprocess + split + train + Gate-1 eval under `--hvg-on train`; save
  `data/processed/gene_list_trainhvg.txt` without overwriting the Phase-1 gene list.
- Leakage-delta analysis: Δ in test kNN@5 / silhouette / TFS and gene-list Jaccard
  (all-sample vs train-only HVG) — expect a small Δ (leakage was negligible).
- Add `test_hvg_train_only_ignores_test` and `test_hvg_flag_reproduces_phase1`.
