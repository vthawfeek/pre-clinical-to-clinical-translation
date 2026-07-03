# Pre-Clinical to Clinical Translation

**Can a neural network map cancer cell lines to their human patient counterparts?**

A drug that works on a cell line often fails in patients — partly because the cell
line was never a good stand-in for the disease. This project learns a shared
embedding space in which a **CCLE cell line** sits next to the **TCGA patients** it
most resembles, so you can ask *"which patients does this pre-clinical model
actually represent?"* and get a quantified answer.

Two encoder towers (one per domain) are trained with a supervised contrastive
(SupCon-InfoNCE) objective to align same-lineage cell lines and patients across the
cell-line ↔ tumour domain gap, over LUAD, BRCA, and SKCM.

```
   CCLE cell line                                   TCGA patient
   (2000 HVGs, z-scored)                            (2000 HVGs, z-scored)
          │                                                │
   ┌──────▼──────┐   [2000→1024→512→256→128→64]     ┌──────▼──────┐
   │ CCLEEncoder │   Linear→BatchNorm→ReLU→Dropout  │ TCGAEncoder │
   │  (~2.75M)   │   (separate weights per domain)  │  (~2.75M)   │
   └──────┬──────┘                                  └──────┬──────┘
          │  z_ccle (64-d)                                 │  z_tcga (64-d)
          └────────────────►  L2 normalise  ◄──────────────┘
                       shared unit hypersphere
          SupCon-InfoNCE: same lineage across domains = positive
                                  │
                    kNN retrieval: nearest patients
```

## Results (Test Set)

Held-out test split; the model never saw these samples during training or
checkpoint selection.

| Metric | Random | PCA + kNN | Harmony (real, harmonypy) | **This Work** |
|---|---|---|---|---|
| kNN@5 Accuracy | 33.3 % | 65.8 % | 84.2 % | **100 %** (95% CI 90.8–100 %, n=38) |
| kNN@1 Accuracy | 33.3 % | — | — | **97.4 %** |
| Silhouette Score | — | — | — | **+0.57** |
| TFS (composite) | — | — | — | **0.89** |

**Gate 1 decision: DEPLOY** (threshold: kNN@5 ≥ 70 %). Per-lineage kNN@5 is 100 %
for LUAD, BRCA, and SKCM. The single hardest cell line is `ACH-000264` (Calu-6, an
anaplastic NSCLC line), TFS 0.662. Full breakdown in
[docs/05_evaluation.md](docs/05_evaluation.md).

**This result has since been stress-tested (Phase 2, Days 15–24 — see
[reports/phase2-summary.md](reports/phase2-summary.md)).** The headline claim is:
**100 % kNN@5 (95 % CI 90.8–100 %, n=38); stable across 10 seeds (0.950 ± 0.034,
CI [0.932, 0.971]); beats the best real batch-correction baseline (Harmony 84.2 %)
by +15.8 pts; survives a label-shuffle negative control (p = 0.0099) and a
tumour-purity confounder check (100 % kNN@5 in both purity strata).**

Extending to a harder **15-lineage** task gives genuine headroom — kNN@5 **78.4 %**
(Wilson CI 69.8–85.0 %, n=111) — with errors concentrated on biologically
confusable pairs (LUAD↔LUSC, GBM↔LGG, COAD↔READ absorb 45.8 % of off-diagonal
error mass, a 12× enrichment), not scattered randomly. A first case study tying the
embedding to a real drug-response phenotype (BRAF-mutant melanoma / vemurafenib)
found a **weak positive** placement effect (p = 0.047, effect 0.649, CI
[0.465, 0.834]) but a **null** link to vemurafenib sensitivity itself (Spearman
rho = 0.209, CI [−0.109, 0.493], p = 0.19, n=41) — reported honestly rather than
rounded up.

## Quick Start

```bash
# 1. Install (uv manages the environment)
pip install uv && uv sync

# 2. Download raw data (DepMap 24Q4 CCLE + UCSC Xena TCGA Pan-Cancer)
uv run pctrans-download ccle
uv run pctrans-download tcga

# 3. Preprocess: harmonise gene IDs, select 2000 HVGs, split, fit scalers
uv run pctrans-preprocess --split

# 4. Train the dual-tower contrastive model (~3 min on a T4, ~12 min on CPU)
uv run pctrans-train

# 5. Evaluate on the held-out test set (prints the Gate 1 report)
uv run pctrans-evaluate --model models/best_model.pt --data-dir data/processed/

# 6. Render UMAP + TFS figures
uv run pctrans-visualize

# 7. Query: nearest TCGA patients for one cell line
uv run pctrans-query ACH-000264
```

Programmatic use:

```python
from pctrans.inference import TranslationEmbedder

embedder = TranslationEmbedder("models/best_model.pt", data_dir="data/processed/")
z = embedder.embed_cell_line("ACH-000264")          # -> (1, 64) unit vector
patients = embedder.query_patients("ACH-000264", k=5) # -> nearest TCGA patients
```

## Documentation

| Doc | Contents |
|---|---|
| [01 · Data Pipeline](docs/01_data_pipeline.md) | Sources, gene-ID harmonisation, union-rank HVG selection |
| [02 · Feature Engineering](docs/02_feature_engineering.md) | z-score rationale, split strategy, leakage boundary |
| [03 · Architecture](docs/03_architecture.md) | Encoder diagram, BatchNorm, L2 sphere, asymmetric towers |
| [04 · Training](docs/04_training.md) | SupCon-InfoNCE derivation, learnable τ, batch construction, LR schedule |
| [05 · Evaluation](docs/05_evaluation.md) | kNN protocol, silhouette, TFS, UMAP settings, baselines |

## Stack

Python 3.11 · PyTorch 2.2 · pandas / scikit-learn · umap-learn · MLflow · Streamlit ·
plotly · typer. Managed with **uv**; linted with **ruff**; tested with **pytest**
(85 % coverage).

```bash
uv run ruff check pctrans/ tests/
uv run pytest tests/ --cov=pctrans
```

## Live Demo

**https://pctrans.streamlit.app** — pick a cell line, see its position among the
TCGA patients on a live UMAP, its TFS gauge, and its nearest-patient table.

## Data & Method Notes

- **Lineages:** LUAD, BRCA, SKCM.
- **Features:** top 2,000 highly variable genes via a union-rank method that gives
  CCLE and TCGA equal weight (see docs/01).
- **Leakage control:** scalers are fit on pooled *training* expression only, then
  frozen for val / test / inference (see docs/02).
- **Honesty:** every reported number is on the held-out test split; UMAP is used for
  visualisation only, never for a metric.

## Limitations & Scope

This is a **validated research method, not a clinical tool**. What Phase 2 checked and what it
deliberately did not:

- **Validated:** statistical stability across 10 random splits, a real batch-correction baseline
  (Harmony) beaten by a wide margin, generalisation to 15 lineages with biologically sensible
  failure modes, robustness to a tumour-purity confounder, and a label-shuffle negative control.
  See [reports/phase2-summary.md](reports/phase2-summary.md) for the full evidence table.
- **Out of scope, named so it isn't mistaken for "done":** external cohorts, cross-platform assays
  (microarray/other RNA-seq platforms), patient-derived xenografts, and single-cell data (all
  public bulk RNA-seq here) — and any prospective, pre-registered clinical validation or
  regulatory (SaMD / biomarker) qualification, which is a multi-year, multi-institution effort,
  not an individual-repo extension.
- **The vemurafenib/BRAF case study is a first, partial tie to real biology, not proof of clinical
  utility** — a weak placement effect and a null response-correlation at n=41, reported as such.

## Licence

Research/educational project. CCLE data © Broad Institute (DepMap); TCGA data via
the UCSC Xena Pan-Cancer Atlas hub — see their respective terms of use.
