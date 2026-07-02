# Day 8: Training Run Execution

**Date:** 2026-07-02
**Commit:** `day 8: training run complete, best_model.pt saved, MLflow experiment pctrans-v1`

## What Was Built

- `models/best_model.pt` — best checkpoint from the full training run (dual-tower
  state dict + learnable-temperature loss state, epoch 2, val kNN@5 = 0.9474). Not
  tracked in git (`models/*.pt` is gitignored); regenerable via `pctrans-train`.
- `mlflow.db` (SQLite backend, gitignored) — experiment `pctrans-v1` now holds the
  30-epoch run `pctrans-v1` with per-epoch `train_loss`, `val_loss`,
  `val_knn_accuracy`, `temperature`, and `lr`.
- No source files changed — Day 8 is an execution/analysis day. The training path
  (`pctrans-train` → `ContrastiveTrainer` → `KNNValidationCallback`) was built on
  Days 5–7 and run here at full length.

## What Was Learned

- **The retrieval task is nearly saturated from initialisation.** Epoch-1 val
  kNN@5 was **0.868**, not the ~0.33 random baseline the plan projected. With only
  three well-separated lineages (LUAD/BRCA/SKCM) and L2-normalised 2,000-HVG
  inputs, cross-domain nearest-neighbour lineage voting already works before the
  encoders learn much. Contrastive training then nudges it to 0.947. This is the
  headline risk to carry into Gate 1: the metric may be easy to pass and a weak
  test of genuine alignment. Day 9/10 (silhouette + TFS) must add nuance.
- **Temperature did not collapse — it drifted slightly *up*.** The learnable
  log(1/τ) moved τ from 0.0700 → 0.0735 over 7 epochs, the opposite of the
  <0.01 collapse the plan flagged as a numerical-instability failure mode. The
  softmax sharpness is essentially stable; no clamp was needed.
- **Absolute loss values run far higher than the plan's ~2.8 estimate.** Train
  loss started at 7.76 and plateaued at ~4.17; val loss stayed ~8.0–8.9. The
  SupCon multi-positive InfoNCE over the full pooled validation set (38 CCLE +
  339 TCGA) has many more negatives than the plan's back-of-envelope log(B_t)
  figure assumed, so the magnitude is not comparable to that estimate — but the
  curve is monotonically decreasing on train, which is the signal that matters.
  Val loss and val kNN diverge (loss flat/high while kNN high), reinforcing that
  kNN retrieval, not loss magnitude, is the metric to trust here.
- **Early stopping fired at epoch 7.** Best val kNN@5 (0.9474) occurred at epoch 2;
  with `early_stop_patience = 5` and no strict improvement afterward, training
  halted after epoch 7. The full 30 epochs were never needed.

## Key Decisions

- **Ran the optional hyperparameter mini-sweep (plan task 4) but isolated its
  artifacts.** Sweep runs wrote checkpoints to the scratchpad and used
  `--no-mlflow`, so the canonical `models/best_model.pt` and the `pctrans-v1`
  MLflow experiment remain the default-config run (lr=3e-4, batch=48). This keeps
  the deliverable reproducible while still producing the comparison table.
- **Compared runs by best val kNN, not "at epoch 20" as the plan text suggested.**
  Every config early-stops well before epoch 20 (all by epoch ~7), so epoch-20 is
  unreachable; best val kNN is the meaningful comparison point.
- **Kept the default config (run_B) as canonical despite run_C scoring higher.**
  run_C's 0.9737 vs run_B's 0.9474 is a 1-sample difference on a 38-sample val set
  (1 sample ≈ 2.6%), i.e. within noise. The default is the pre-registered config;
  changing it on noise-level evidence would be overfitting to the val set. Revisit
  only if Gate 1 (test set) motivates it.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
.......................................                                  [100%]
39 passed, 3 deselected in 6.14s
```

Training CLI (default config, MLflow experiment `pctrans-v1`):
```
Train: CCLE 183 + TCGA 1586 | Val: CCLE 38 + TCGA 339
Sampler: 41 batches/epoch, 8 per lineage/domain
Model: 5,500,288 params, init tau=0.0700
Training for 30 epoch(s)...

==============================================
           GATE 0 SANITY CHECK
==============================================
Epoch 1 train loss:   7.7550
Epoch 1 val loss:     8.9655
Epoch 1 val kNN@5:   0.8684
Epoch 1 temperature:  0.0702
----------------------------------------------
Best val kNN@5:      0.9474 (epoch 2)
Final temperature:    0.0735
DECISION: PASS -> proceed to Week 2
==============================================
```

Best checkpoint metadata (`models/best_model.pt`):
```
display epoch 2 | val_knn 0.9474 | lr 0.0003 | batch 48
```

## Numbers

**Model / data**

| Item | Value |
|---|---|
| Parameters | 5,500,288 |
| Train samples | CCLE 183 + TCGA 1586 |
| Val samples | CCLE 38 + TCGA 339 |
| Batches/epoch | 41 (8 per lineage per domain) |
| Epochs run | 7 of 30 (early-stopped, patience 5) |
| Wall-clock (CPU) | ~51 s |

**Learning curve — default run (`pctrans-v1`)**

| Epoch | train_loss | val_loss | val_kNN@5 | τ | lr |
|------:|-----------:|---------:|----------:|------:|-------:|
| 1 | 7.7550 | 8.9655 | 0.8684 | 0.0702 | 1.0e-4 |
| 2 | 5.9275 | 8.3024 | **0.9474** | 0.0705 | 1.6e-4 |
| 3 | 4.7901 | 8.1067 | 0.9211 | 0.0711 | 2.2e-4 |
| 4 | 4.3396 | 8.5056 | 0.9211 | 0.0720 | 2.8e-4 |
| 5 | 4.2004 | 8.5118 | 0.8684 | 0.0729 | 3.0e-4 |
| 6 | 4.1754 | 8.0060 | 0.9474 | 0.0732 | 3.0e-4 |
| 7 | 4.1701 | 8.1099 | 0.9474 | 0.0735 | 3.0e-4 |

Best val kNN@5 = **0.9474 at epoch 2**. Temperature evolution: 0.0700 → 0.0735
(monotonic, no collapse). LR warmup (5 epochs) reaches the 3.0e-4 plateau by
epoch 5, right as early stopping engages — the cosine decay leg is never
exercised because training halts first.

**Hyperparameter mini-sweep (best val kNN@5; canonical = run_B)**

| Run | lr | batch | init τ | Best val kNN@5 | Best epoch |
|---|---|---|---|---|---|
| A | 1e-3 | 48 | 0.07 | 0.9211 | 1 |
| **B (default)** | **3e-4** | **48** | **0.07** | **0.9474** | **2** |
| C | 1e-4 | 32 | 0.10 | 0.9737 | 5 |

All three exceed 0.92; the spread (0.9211–0.9737) is ≈2 val samples wide, i.e.
within noise on a 38-sample val set. No config is meaningfully better; default
retained.

## Next Up

- Day 9 — build `notebooks/02_training_analysis.ipynb`: dual-axis loss/kNN curve,
  τ evolution, **per-lineage** val kNN (is SKCM the hardest?), per-encoder gradient
  norms, UMAP of val embeddings at epoch 1 vs. best epoch.
- Investigate the near-saturated retrieval metric: confirm the sampler assigns
  lineage labels correctly and that high kNN isn't a leakage artifact.
- `embed_dim` ablation (32 / 64 / 128, 3 epochs each); keep 64 unless 32 already
  clears 70%.
- Check for domain collapse: cosine similarity of CCLE vs. TCGA embedding
  centroids should stay <0.5.
- Prepare Gate 1 (Day 10): `evaluation/knn.py`, `silhouette.py`, `tfs.py`,
  `pctrans-evaluate` on the held-out **test** set.
