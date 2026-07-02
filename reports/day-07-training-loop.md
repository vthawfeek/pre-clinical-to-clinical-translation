# Day 7: Training Loop + Gate 0

**Date:** 2026-07-02
**Commit:** `day 7: ContrastiveTrainer, KNNValidationCallback, pctrans-train CLI, blog-01 draft`

## What Was Built

- `pctrans/training/callbacks.py` — `KNNValidationCallback`. Embeds all validation CCLE and TCGA
  samples with the frozen model, fits scikit-learn `NearestNeighbors` on the TCGA gallery, and for
  each CCLE sample takes a majority-vote over its k nearest TCGA neighbours in 64-dim L2 space.
  Returns `{"val_knn_accuracy", "per_lineage": {LUAD/BRCA/SKCM}, "k"}`. Ties in the majority vote
  resolve to the lowest lineage index (deterministic); `k` is capped at the TCGA gallery size.
- `pctrans/training/trainer.py` — `ContrastiveTrainer`. Adam over model + `log_tau`, per-epoch
  cosine LR schedule with a linear warmup (`LambdaLR`), grad-norm clip to 1.0, per-epoch kNN
  validation + a pooled-batch validation loss, best-checkpoint saving on val-kNN improvement, and
  early stopping (`patience`). MLflow logging (train_loss, val_loss, val_knn_accuracy, temperature,
  lr) is active only when a `mlflow_run_name` is supplied, so tests stay side-effect-free.
- `pctrans/training/__init__.py` — re-exports `ContrastiveTrainer`, `KNNValidationCallback`.
- `pctrans/scripts/train.py` — `pctrans-train` CLI. Loads `ccle_2k.parquet` / `tcga_2k.parquet` /
  `splits.json` / `scalers.pkl`, z-scores each split with the train-fit scaler, builds the
  stratified sampler + model + loss from `configs/*.yaml`, trains, and prints the Gate 0 sanity
  block. Flags: `--epochs`, `--config`, `--model-config`, `--data-dir`, `--mlflow/--no-mlflow`.
- `tests/conftest.py` — added `tiny_ccle_dataset`, `tiny_tcga_dataset`, `tiny_sampler`, and
  `small_model` (a 50-gene dual tower matching the tiny fixtures).
- `tests/test_training.py` — 6 tests: kNN callback on perfect one-hot embeddings (=1.0), one
  training epoch (finite losses, kNN in [0,1]), checkpoint saved on improvement, early stopping
  fires, checkpoint round-trip restores weights, LR warmup-then-cosine shape.
- `.gitignore` — added `mlflow.db` and `mlartifacts/` (MLflow created a SQLite backend on the real
  training run).
- `reports/blog-01-concept.md`, `reports/linkedin-01.txt` — Blog Post 1 drafts (via `/blog-draft 7`).
- `configs/training.yaml` — already carried the Day 7 values; no change needed.

## What Was Learned

- **Epoch-1 val kNN@5 is 81.6%, not the plan's ~33% estimate.** After a single epoch (41 batches,
  still inside LR warmup so effective lr ≈ 6e-5) the three lineages are already well separated in
  the learned space. The most likely cause is that LUAD, BRCA, and SKCM are transcriptionally very
  distinct, and per-gene z-scoring + BatchNorm + the SupCon signal pull them apart fast. This is a
  val-set proxy on 38 CCLE / 339 TCGA samples, not the held-out test set — Gate 1 (Day 10) is the
  real number, on TEST, after full training. Encouraging, but not to be reported as the result yet.
- **Loss magnitudes match the theory.** Symmetric SupCon = sum of two directional means; each
  direction is a softmax over ~24 TCGA keys, so a near-random start sits around `2 × log(24) ≈ 6.4`.
  The observed train 7.68 / val 8.78 are just above that (init is slightly worse than random
  alignment), confirming the loss is wired correctly and not collapsed.
- **Temperature barely moves in one epoch** (0.0700 → 0.0702), as expected — τ convergence toward
  the 0.02–0.05 range is a multi-epoch effect that Day 8-9 will track.
- **MLflow defaulted to a SQLite backend** (`mlflow.db`) in this environment rather than a bare
  `mlruns/` file store, so the ignore list needed both.

## Key Decisions

- **`log_tau` is left unclamped.** PLAN.md's Day 8 note suggests `clamp(log_tau, -4, 2)`, but (as the
  Day 6 report flagged) `log_tau` stores `log(1/τ)`, so that clamp would floor τ at 0.135 — above the
  target 0.02–0.05 range. Clamping is deferred until/unless τ actually destabilises during the full
  Day 8 run.
- **`test_one_training_epoch` uses a purpose-built `small_model` (input_dim=50), not `tiny_model`.**
  The plan snippet names `tiny_model`, but that fixture is 2000-dim to satisfy the Day 6 encoder
  tests while the tiny data has 50 genes. An input-matched small tower keeps the test honest and
  fast (<1s), consistent with the Day 6 report's fixture reasoning.
- **Validation loss is computed as one pooled SupCon batch** (all val CCLE vs all val TCGA) rather
  than a sampled mini-batch, so `val_loss` is deterministic across epochs and directly comparable.
- **MLflow logging is gated on `mlflow_run_name`.** The CLI passes a run name (logs an experiment);
  the unit tests pass `None` (no run, no `mlruns/` writes), keeping the suite hermetic.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
.......................................                                  [100%]
39 passed, 3 deselected in 26.44s

$ uv run pctrans-train --epochs 1
Train: CCLE 183 + TCGA 1586 | Val: CCLE 38 + TCGA 339
Sampler: 41 batches/epoch, 8 per lineage/domain
Model: 5,500,288 params, init tau=0.0700
Training for 1 epoch(s)...
==============================================
           GATE 0 SANITY CHECK
==============================================
Epoch 1 train loss:   7.6826
Epoch 1 val loss:     8.7848
Epoch 1 val kNN@5:   0.8158
Epoch 1 temperature:  0.0702
----------------------------------------------
Best val kNN@5:      0.8158 (epoch 1)
Final temperature:    0.0702
DECISION: PASS -> proceed to Week 2
==============================================
```

## Gate 0 Outcome

```
[Gate 0: Day 7]
├── 1 epoch completes            ✓  (41 batches, ~2m14s on CPU)
├── loss finite, no NaN/Inf      ✓  (train 7.6826, val 8.7848)
├── val kNN > 0                  ✓  (0.8158, well above random 0.333)
└── temperature logs correctly   ✓  (0.0702, positive)
DECISION: PASS → proceed to Week 2
```

## Numbers

- Train split: 183 CCLE + 1,586 TCGA. Val split: 38 CCLE + 339 TCGA.
- Sampler: 41 batches/epoch, 8 CCLE + 8 TCGA per lineage (batch of 48).
- Model params: 5,500,288 (unchanged from Day 6). Init τ = 0.0700.
- Epoch-1: train loss 7.6826, val loss 8.7848, val kNN@5 0.8158, τ 0.0702.
- Runtime: ~2m14s for 1 CPU epoch (≈3.3s/batch incl. validation).
- Tests: 6 new (training); 39 total passing, 3 deselected (slow/integration).

## Next Up

- Day 8: full 30-epoch run via `pctrans-train --config configs/training.yaml`; capture the learning
  curve (loss + val kNN at epochs 1/10/20/30) and τ evolution.
- Day 8: watch the early-warning signs (loss→0, loss stuck at log(24), τ<0.01, val kNN<33%).
- Day 8: confirm `models/best_model.pt` is saved on the best val-kNN epoch.
- Day 8: optional lr / batch_size / init_tau mini-sweep, compare val kNN at epoch 20.
