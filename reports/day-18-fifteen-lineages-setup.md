# Day 18: Scale to ~15 Lineages — Data & Model

**Date:** 2026-07-03
**Commit:** `day 18: config-driven lineages, 15-lineage data + training run`

## What Was Built

- `pctrans/data/dataset.py` — `build_lineage_maps(lineages)` builds `{lineage: idx}` /
  `{idx: lineage}` maps from an ordered list (insertion order, not sorted, so the
  Phase-1 default reproduces the original hardcoded `{"LUAD": 0, "BRCA": 1, "SKCM": 2}`
  byte-for-byte). `_ExpressionDataset` (and `CCLEDataset`/`TCGADataset`) now accept an
  optional `lineage_to_idx` override, defaulting to the module-level map.
- `pctrans/data/ccle_client.py` — extended `LINEAGE_ALIASES` with 17 new
  primary-disease/subtype keys covering LUSC, COAD, READ, PAAD, STAD, LIHC, KIRC,
  HNSC, GBM, LGG, OV, BLCA. Deliberately left unmapped: generic "Renal Cell
  Carcinoma" (NOS, ambiguous vs papillary/chromophobe), generic "Colorectal
  Adenocarcinoma" (NOS, ambiguous vs COAD/READ), "Diffuse Intrinsic Pontine Glioma"
  (pediatric, not in the adult TCGA GBM/LGG cohorts).
- `pctrans/data/preprocessor.py` — `FeatureSynchroniser.__init__(lineages=None)`
  makes the lineage list a constructor argument instead of a module constant
  (`load_ccle`/`load_tcga` now filter on `self.lineages`). Added
  `drop_incomplete_genes(ccle_expr, tcga_expr, genes)` — see "What Was Learned".
- `pctrans/training/callbacks.py`, `pctrans/training/trainer.py` —
  `KNNValidationCallback`/`ContrastiveTrainer` accept an optional `idx_to_lineage`
  map so the per-lineage validation breakdown works for any lineage count, not just
  the hardcoded 3.
- `pctrans/scripts/preprocess.py` — new `--data-config` (lineage list + n_hvgs/
  val_frac/test_frac/seed defaults, CLI flags still override) and `--output-suffix`
  (explicit artefact-name suffix, e.g. `_15`) options; `_artefact_names()` replaces
  the old static `_ARTEFACT_NAMES` dict but reproduces its output byte-for-byte when
  `output_suffix` is left `None`.
- `pctrans/scripts/train.py` — new `--data-config` option builds the lineage map via
  `build_lineage_maps` and threads it through dataset construction and the trainer.
- `configs/data_15.yaml` — the 15-lineage set (BLCA, BRCA, COAD, GBM, HNSC, KIRC,
  LGG, LIHC, LUAD, LUSC, OV, PAAD, READ, SKCM, STAD); `configs/data.yaml` untouched.
- `configs/training_15.yaml` — `batch_size: 120` (4 CCLE + 4 TCGA per lineage × 15
  lineages, vs. the 3-lineage config's 48), `checkpoint_path: models/best_model_15.pt`,
  separate `mlflow_experiment`.
- `tests/test_data.py` — 8 new tests: config-driven lineage maps (contiguous ids,
  Phase-1-default reproduction), custom `lineage_to_idx` on `CCLEDataset`, the new
  CCLE alias resolutions, `FeatureSynchroniser(lineages=...)`, the sampler with 15
  lineages (`per_lineage == 4`, every batch covers all 15 from both domains), and
  `drop_incomplete_genes` (excludes-any-NaN + no-op-when-clean).
- Ran the real pipeline: `data/processed/{ccle,tcga}_2k_15.parquet`,
  `gene_list_15.txt`, `splits_15.json`, `scalers_15.pkl` (train-only HVG, per Day 16)
  and `models/best_model_15.pt` (not committed — gitignored per existing convention).

## What Was Learned

- **A real NaN bug surfaced only at 15 lineages.** The Xena PANCAN "batch-effect
  adjusted" TCGA matrix has genuine missing values for some gene/cohort
  combinations: 2,946 of 16,568 CCLE∩TCGA common genes (~18%) had at least one NaN
  once the 12 new TCGA cohorts were pulled in, versus **zero** NaN cells in the
  original LUAD/BRCA/SKCM subset. Train-only HVG selection (Day 16) ranks a gene
  using only the NaN-free train slice, so a gene with NaNs confined to val/test rows
  could still be selected — silently propagating NaN through scaling into the model,
  which surfaced as an opaque `sklearn` `NearestNeighbors` crash ("Input X contains
  NaN") on the very first validation epoch. Fixed with `drop_incomplete_genes()`
  as a completeness gate *before* HVG ranking, not a training-time workaround.
  Confirmed a no-op for the 3-lineage pipeline (`test_hvg_flag_reproduces_phase1`
  still reproduces the committed `gene_list.txt` byte-for-byte).
- **The harder task has genuine headroom, as intended.** Best val kNN@5 = 82.9%
  (epoch 23/28) at 15 lineages vs. Phase-1's ~97–100% at 3 lineages — the metric can
  now actually fail, which is the entire point of Rung 2.
- **Even pre-Day-19, the errors already look biological, not random.** The two
  worst-performing lineages (LGG 0%, READ 0% at the best checkpoint) are exactly
  the two smallest CCLE cohorts (17 and 13 lines) *and* exactly the two members of
  the plan's deliberately-confusable pairs (LGG↔GBM glioma, READ↔COAD colorectal).
  Promising early signal for Day 19's dedicated confusion-matrix analysis.
- **Real-world lineage counts don't round to nice guideline numbers.** READ landed
  at 13 CCLE lines after joining metadata to the expression matrix (15 before the
  join) — just under the plan's "~15" rule of thumb. Documented below as a
  deliberate keep, not silently swept under the rug.

## Key Decisions

1. **`build_lineage_maps` preserves insertion order, does not sort.** Sorting
   alphabetically would have changed the Phase-1 default from `{"LUAD":0,"BRCA":1,
   "SKCM":2}` to `{"BRCA":0,"LUAD":1,"SKCM":2}` — silently breaking the label
   convention baked into the already-committed `ccle_embeddings.npz` /
   `embeddings_test.npz` and the live Streamlit app. Insertion order keeps the
   Phase-1 default byte-identical while still letting the 15-lineage config choose
   its own (here: alphabetical, written directly into `configs/data_15.yaml`).
2. **`drop_incomplete_genes` is a first-class `FeatureSynchroniser` method, not a
   one-off script fix**, since the missing-value gap is a property of the raw Xena
   matrix that will recur for any future lineage expansion, not a Day-18-only issue.
3. **Retained READ at N=13 CCLE lines** (below the "~15" guideline) rather than
   dropping it. Unlike the plan's named PRAD exclusion (too few lines to be
   meaningful at all), READ is the deliberate confusable partner of COAD — dropping
   it would remove one of the three named confusable pairs the whole exercise exists
   to test, for a shortfall of only 2 lines.
4. **Batch size raised 48 → 120** for the 15-lineage config to hold the plan's
   "≥ 4 CCLE + 4 TCGA per lineage per batch" floor (15 lineages × 2 domains × 4 = 120)
   rather than let `per_lineage` silently collapse to 1.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
........................................................................ [ 75%]
.......................                                                  [100%]
95 passed, 6 deselected in 22.97s

$ uv run pytest tests/test_data.py -q -m integration -k hvg_flag
.                                                                        [100%]
1 passed, 36 deselected in 33.78s
```

Preprocessing (`pctrans-preprocess --data-config configs/data_15.yaml --hvg-on train
--output-suffix _15`):

```
Common genes (CCLE & TCGA): 16568
Dropped 2946 gene(s) with missing values in at least one domain; 13622 complete common genes remain
Selected top 2000 HVGs (union-rank method, variance on train slice only)
Stratified split (train / val / test):
  CCLE: train 512 | val 111 | test 111
  TCGA: train 4996 | val 1070 | test 1070
```

Training (`pctrans-train --config configs/training_15.yaml ... --data-config configs/data_15.yaml`):

```
Model: 5,500,288 params, init tau=0.0700
Sampler: 29 batches/epoch, 4 per lineage/domain
Epoch 1 train loss:   9.9361   val loss: 11.1033   val kNN@5: 0.2703
...
Epoch 23 val kNN@5: 0.8288  (best; early-stopped at epoch 28, patience 5)
DECISION: PASS -> proceed to Week 2
```

## Numbers

**CCLE cell lines by lineage (734 total, post expression-matrix join):**

| Lineage | N | Lineage | N | Lineage | N |
|---|---|---|---|---|---|
| SKCM | 110 | OV | 64 | LUSC | 27 |
| LUAD | 80 | GBM | 57 | LIHC | 23 |
| HNSC | 71 | PAAD | 51 | KIRC | 18 |
| BRCA | 69 | STAD | 35 | LGG | 17 |
| COAD | 66 | BLCA | 33 | READ | 13 |

**TCGA patients by lineage (7,136 total, post expression-matrix join):**

| Lineage | N | Lineage | N | Lineage | N |
|---|---|---|---|---|---|
| BRCA | 1215 | LUSC | 552 | STAD | 450 |
| KIRC | 606 | LGG | 529 | BLCA | 427 |
| LUAD | 576 | COAD | 492 | LIHC | 423 |
| HNSC | 566 | SKCM | 473 | OV | 308 |
| | | | | PAAD | 183 |
| | | | | READ | 170 |
| | | | | GBM | 166 |

No lineages dropped — all 15 planned lineages met the CCLE/TCGA count guard
(READ at 13 CCLE lines is the sole exception, retained per Key Decision 3).

**Best checkpoint (epoch 23/28, early-stopped) per-lineage val kNN@5:**

| Lineage | kNN@5 | Lineage | kNN@5 | Lineage | kNN@5 |
|---|---|---|---|---|---|
| GBM | 1.000 | BRCA | 0.900 | LIHC | 0.667 |
| KIRC | 1.000 | COAD | 0.900 | BLCA | 0.600 |
| STAD | 1.000 | OV | 0.900 | LUSC | 0.500 |
| SKCM | 0.938 | LUAD | 0.833 | LGG | 0.000 |
| HNSC | 0.909 | PAAD | 0.750 | READ | 0.000 |

Overall val kNN@5: **0.829** (15-lineage) vs. Phase-1's 0.974 (3-lineage) — real
headroom, as designed.

## Next Up

- Day 19: run `pctrans-evaluate` on the 15-lineage **test** set (not just val) with
  Wilson/bootstrap CIs, the full 15×15 confusion matrix, silhouette, TFS.
- Error-structure analysis: check whether LGG/READ's weak val performance
  concentrates on their named confusable partners (GBM, COAD) at test time too.
- Render a 15-lineage UMAP and add a 15-lineage section to
  `notebooks/03_evaluation.ipynb`.
- State plainly how much the headline metric dropped (97% → ~83%) and why that is
  the expected, honest outcome of a harder task.
