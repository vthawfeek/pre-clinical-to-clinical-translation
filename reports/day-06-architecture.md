# Day 6: Dual-Tower Architecture & Loss Module

**Date:** 2026-07-02
**Commit:** `day 6: CCLEEncoder, TCGAEncoder, DualTowerModel, SupConInfoNCELoss`

## What Was Built

- `pctrans/models/encoders.py` — `MLPBlock` (Linear → BatchNorm1d → ReLU → Dropout), a shared
  `_TowerEncoder` template, and `CCLEEncoder` / `TCGAEncoder` as separate subclasses (identical
  architecture, independent weights). Four hidden blocks `[1024, 512, 256, 128]` at dropout 0.3
  (last hidden block at 0.2) plus a bare `Linear(128, 64)` projection head with no BN/activation.
- `pctrans/models/dual_tower.py` — `DualTowerModel` wraps both encoders; `forward`, `encode_ccle`,
  and `encode_tcga` all L2-normalise the raw embeddings onto the unit hypersphere.
- `pctrans/models/losses.py` — `SupConInfoNCELoss` with a learnable `log_tau = log(1/tau)`
  parameter. Builds a same-lineage cross-domain positive mask, scales cosine similarities by
  `exp(log_tau)`, applies the symmetric SupCon-InfoNCE formula (CCLE-anchored + TCGA-anchored),
  and exposes a `tau` property for monitoring.
- `pctrans/models/__init__.py` — re-exports the four public classes.
- `tests/conftest.py` — `tiny_model` fixture upgraded from a config dict to a real
  `DualTowerModel(input_dim=2000, embed_dim=64)` in eval mode (the Day 1 report flagged this
  fixture would "land once encoders.py is implemented Day 6").
- `tests/test_models.py`, `tests/test_losses.py` — 13 new tests (shapes, unit-sphere norm,
  separate tower weights, gradient flow to both towers, parameter count, loss positivity,
  loss-decreases-on-aligned-batch, learnable-temperature gradient).
- `configs/model.yaml` — already carried the Day 6 values (`input_dim`, `hidden_dims`, `embed_dim`,
  `dropout_high`, `dropout_low`, `init_tau`); no change needed.

## What Was Learned

- **Actual parameter count is ~5.5M, not the plan's ~8.6M estimate.** Each encoder is 2,750,144
  params (dominated by the `Linear(2000, 1024)` block at 2.05M); two towers = 5,500,288. The plan's
  8.6M figure overshoots — the real network is leaner. Reported honestly rather than fudged.
- **`log_tau` stores `log(1/tau)`, not `log(tau)`.** At `init_tau=0.07`, `log_tau = 2.659` and
  `exp(log_tau) = 14.286` is the similarity multiplier (CLIP logit-scale convention). The `tau`
  property returns `exp(-log_tau) = 0.070`, matching CLAUDE.md's "learnable log(1/τ)". The plan's
  Day 8 "clamp log_tau to [-4, 2]" note is inconsistent with this parameterisation (it would clamp
  the init value and floor τ at 0.135, above the expected 0.02–0.05 convergence range), so clamping
  was deferred rather than applied here.
- **The SupCon symmetric loss is the *sum* of the two directional means**, giving
  `symmetric ≈ 2 × single-direction` — consistent with the Day 13 test expectation.
- Anchors with zero positives are masked out (`pos_counts.clamp(min=1)` + `valid` filter) so a
  degenerate batch can never produce a NaN, even though the stratified sampler guarantees every
  lineage appears in both domains.

## Key Decisions

- **`tiny_model` fixture is now a full-dimension `DualTowerModel` (2000→64), not a genuinely tiny
  one.** The plan's Day 6 tests literally call `tiny_model.encode_ccle(torch.randn(8, 2000))` and
  expect `(8, 64)`, so the fixture must use production dims. Instantiating 5.5M params and running a
  forward on 8 samples is <0.1s, so keeping it "tiny" in spirit (small batch) is enough. Fixture is
  `.eval()` so BatchNorm uses running stats and results are deterministic.
- **Shared `_TowerEncoder` base class** rather than duplicating the MLP in both encoders. Keeps the
  "same template, separate weights" design in one place; `CCLEEncoder`/`TCGAEncoder` are distinct
  `nn.Module` subclasses so their parameters never alias (verified by `data_ptr` test).
- **Encoder signature extended with `dropout_low=0.2`** (superset of the plan's `dropout=0.3`
  signature) so the architecture diagram's per-layer dropout schedule is expressible without a
  magic constant.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/test_models.py tests/test_losses.py -q
.............                                                             [100%]
13 passed in 2.90s

$ uv run pytest tests/ -q -m "not slow and not integration"
.................................                                        [100%]
33 passed, 3 deselected in 3.81s
```

Parameter breakdown (CCLEEncoder; TCGAEncoder identical):

```
  blocks.0 Linear(2000,1024)          2,049,024
  blocks.0 BatchNorm1d(1024)              2,048
  blocks.1 Linear(1024,512)             524,800
  blocks.1 BatchNorm1d(512)               1,024
  blocks.2 Linear(512,256)              131,328
  blocks.2 BatchNorm1d(256)                 512
  blocks.3 Linear(256,128)               32,896
  blocks.3 BatchNorm1d(128)                 256
  projection Linear(128,64)               8,256
  TOTAL per encoder                   2,750,144
  DualTowerModel total                5,500,288
  SupConInfoNCELoss learnable params          1  (log_tau)
```

## Numbers (if applicable)

- Total model parameters: **5,500,288** (~5.5M), 2,750,144 per tower.
- Loss init: `log_tau = 2.6593`, `exp(log_tau) = 14.286`, `tau = 0.0700`.
- Encoder output: `(B, 64)`, L2 norm = 1.0 (atol 1e-5).
- Test count: 13 new (models + losses); 33 total passing, 3 deselected (slow/integration).

## Next Up

- Day 7: `ContrastiveTrainer` (Adam, cosine LR + 5-epoch warmup, grad-clip 1.0, MLflow logging,
  checkpoint on val-kNN improvement, early stopping patience 5).
- Day 7: `KNNValidationCallback` (kNN@5 over 64-dim L2 space via scikit-learn `NearestNeighbors`).
- Day 7: `configs/training.yaml` + `pctrans-train` CLI, then **Gate 0** — run 1 epoch, confirm
  finite loss, kNN > 0, temperature logs correctly.
- Day 7: Draft `reports/blog-01-concept.md` ("The Cell Line Translation Problem") via `/blog-draft 7`.
