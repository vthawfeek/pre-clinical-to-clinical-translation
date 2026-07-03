# Day 21: Label-Shuffle Negative Control (Permutation Test)

**Date:** 2026-07-03
**Commit:** `day 21: label-shuffle permutation negative control, empirical p-value`

## What Was Built

- `pctrans/evaluation/stats.py` — `permutation_test(real_value, null_generator, n_perm=20, seed=0)`.
  A deliberately generic entry point: `null_generator(rng)` is any callable that returns a
  float, so the same function scores both Day 21 variants (and any future permutation-style
  control) without `stats.py` importing anything about models, datasets, or the trainer.
  Returns the null distribution, its mean/max, and the empirical p-value (fraction of null
  draws `>= real_value`, Laplace/add-one corrected so a real value that beats every
  permutation still gets a finite, non-zero p rather than a false `0.0`).
- `pctrans/evaluation/viz.py` — `permutation_null_panel(results, chance_level=None, title=...)`:
  one histogram per variant (null values + a solid line at the real value + a dashed line at
  chance), p-value and n_perm printed in the subplot title.
- `pctrans/scripts/permutation_test.py` — new `pctrans-permutation-test` CLI implementing both
  variants from the plan:
  - **eval-only** (cheap): loads the already-computed test embeddings
    (`embeddings_test_15.npz`), shuffles the TCGA test label array, and recomputes kNN@5 with
    no retraining — isolates *metric-level* chance.
  - **retrain** (expensive): shuffles the TCGA `lineage` column (a per-sample relabelling that
    preserves per-class counts, so the stratified sampler is unaffected) independently within
    each of train/val/test, trains a fresh model for a short (5-epoch) schedule, and evaluates
    test kNN@5 under the same shuffled labels — the stronger claim that a full training run
    cannot learn a correspondence that isn't there.
  - Both nulls are scored against the same real value: the actual `best_model_15.pt`'s real
    (unshuffled) test kNN@5, recomputed from the committed embeddings so it uses the identical
    code path as the null.
  - Writes `reports/permutation_test.json` + `reports/permutation_null.png`.
- `pyproject.toml` — registers `pctrans-permutation-test`.
- `tests/test_stats.py` — `test_permutation_null_near_chance` (a synthetic shuffled-label
  matcher lands within 0.03 of 1/15 chance over 20 perms) and `test_permutation_pvalue_range`
  (p always in [0, 1]; a real-signal case gets p < 0.01 at n_perm=200; a real value sitting
  inside a wide null gets p > 0.1).
- `tests/test_viz.py` — `test_permutation_null_panel_one_axis_per_variant` (one matplotlib axis
  per variant dict entry).

## What Was Learned

- **The null landed almost exactly where the plan predicted, before any tuning.** The plan's
  illustrative numbers said "mean ~0.07 (~1/15), max 0.14 over 20 perms" — the real run (100
  perms) came back with null mean 7.0–7.7% against a true chance level of 6.67%, max 17–20%.
  That is about as clean a sanity check as this kind of control gets: the shuffled-label task
  behaves exactly like an unlearnable random-labelling problem should.
- **20 permutations cannot reach the plan's own p < 0.01 target.** The Laplace-corrected
  empirical p-value is `(count_exceeding + 1) / (n_perm + 1)`, so its floor at `n_perm=20` is
  1/21 ≈ 0.048 even when *zero* permutations come close to the real value — the first run
  (`--n-perm 20`, the CLI default that mirrors the plan's illustrative command) hit exactly this
  floor (p = 0.0476) despite the real value (78.4%) being nowhere near the null's max (18.0%).
  Re-running with `--n-perm 100` (floor 1/101 ≈ 0.0099) resolved this and both variants landed
  at p = 0.0099 — the small-sample floor, not weak signal, was the limiting factor.
- **Epochs are cheap at this scale, so the "expensive" retrain variant wasn't actually
  expensive.** A timing probe showed ~1.25s/epoch on the 15-lineage config once the ~30s of
  fixed `uv run` + import + data-load overhead is paid once (2 epochs: 35s; 6 epochs: 40s). Doing
  all 100 permutations inside one Python process (data loaded once, only the TCGA `lineage`
  column reshuffled per iteration) meant the full 100×5-epoch sweep finished in a few minutes of
  actual training compute, well inside a single background run.
- **The two variants agree almost exactly** (null mean 7.7% eval-only vs. 7.0% retrain; p = 0.0099
  both), which is itself informative: the "expensive" retrain null isn't finding extra structure
  the "cheap" eval-only null misses. A shuffled correspondence is unlearnable whether or not the
  model gets to try.

## Key Decisions

1. **Shuffle at the per-sample label level, not a whole-lineage relabelling.** The plan says
   "shuffle the mapping between CCLE lineage labels and TCGA lineage labels." A bijection over
   the label set itself (e.g. relabel all TCGA-LUAD as BRCA) only gives `n_lineages - 1`
   distinct derangements for a 15-class problem — nowhere near enough diversity for a 20-100
   draw null distribution. Permuting individual TCGA sample labels (`rng.permutation` of the
   `lineage` column) preserves per-class counts exactly (so the stratified batch sampler and
   split proportions are unaffected) while giving a combinatorially large, genuinely random null
   each draw — the standard permutation-test construction.
2. **Both variants are scored against one shared real value**, not two separately-recomputed
   ones. The real value is the actual deployed model's real test kNN@5, read off the committed
   `embeddings_test_15.npz` through the same `knn_accuracy_from_embeddings` call the nulls use.
   This keeps the comparison honest (same metric, same code path) and avoids two numbers in the
   report that should be identical but might drift by a rounding difference if computed two ways.
3. **Ran the control on the 15-lineage config (Day 18-19 pipeline), not the saturated 3-lineage
   one.** At 3 lineages the real value is already 100% against a 33% chance — a valid but far
   less informative test than 78.4% real vs. 6.7% chance at 15 lineages, which is also the
   evaluation the project currently treats as its headline number. `--ccle-file`/`--data-config`
   etc. all default-override to the 3-lineage artefacts if that comparison is wanted instead.
4. **Kept `--n-perm 20` as the CLI default** (matching the plan's illustrative command) but ran
   the actual Day 21 analysis with `--n-perm 100` to clear the plan's own p < 0.01 bar. Bumping
   the CLI's own default to 100 felt like overfitting the tool to one report's needs; a caller
   who wants the tighter bound passes `--n-perm 100` explicitly, same as this report did.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
........................................................................ [ 61%]
.............................................                            [100%]
117 passed, 6 deselected in 33.61s

$ uv run pytest tests/test_stats.py -q -k permutation
..
2 passed, 8 deselected in 3.02s
```

Real run (`pctrans-permutation-test --n-perm 100`):

```
Loaded data\processed\embeddings_test_15.npz: CCLE 111 + TCGA 1070
Real (unshuffled) test kNN@5: 78.4%   (chance = 6.7%)

Running eval-only label shuffle (100 permutations)...
Running retrain-based label shuffle (100 permutations x 5 epochs)...
    retrain perm  1: kNN@5   5.4%
    ... (100 permutations, range 0.9%-17.1%)

==========================================================
     DAY 21 — LABEL-SHUFFLE NEGATIVE CONTROL REPORT
==========================================================
Lineages: 15   Chance level: 6.7%
Real test kNN@5: 78.4%
Eval-only shuffle    null mean   7.7%  max  19.8%  (n_perm=100)  ->  p = 0.0099
Retrain shuffle      null mean   7.0%  max  17.1%  (n_perm=100)  ->  p = 0.0099
==========================================================
DECISION: PASS   [target p < 0.01 on the retrain variant]
==========================================================
```

## Numbers

| Quantity | Value |
|---|---|
| Lineages | 15 |
| Chance level (1/15) | 6.67% |
| Real test kNN@5 (unshuffled) | 78.4% |
| Eval-only null: mean / max / p | 7.7% / 19.8% / 0.0099 |
| Retrain null: mean / max / p | 7.0% / 17.1% / 0.0099 |
| Retrain permutations | 100 × 5 epochs |
| Wall time (100-perm real run) | ~26 min (background) |
| Wall time (20-perm smoke run) | ~5 min |

No shuffled-label permutation — eval-only or retrain, 120 draws total — came anywhere close to
the real value; the closest (19.8%) is still 3.9x below the real 78.4% and only 3x the 6.7%
chance level.

## Next Up

- Day 22: assemble vemurafenib response data (GDSC/DepMap) and BRAF mutation calls (MC3) for the
  first real-biology case study.
- Day 23: place vemurafenib-relevant cell lines/patients in the aligned embedding space and check
  whether BRAF-mutant melanoma cell lines land near BRAF-mutant melanoma patients.
- Day 24: Gate 2 evaluation — this negative control (G2-5-equivalent) is now one of the passing
  criteria alongside Day 15's CIs, Day 17's real baselines, and Day 20's purity confounder check.
- Day 25: Blog Post 3, "What Survives Rigorous Validation" — the label-shuffle control is a clean,
  visual (histogram) story element: real result vs. a wall of chance-level bars.
