# Pre-Clinical to Clinical Translation

## Plan

Full plan: [PLAN.md](PLAN.md) (also at `C:\Users\vthawfeek.Shajitha\.claude\plans\use-the-below-plan-concurrent-kahan.md` for plan mode)

## How to trigger a day's work

Type `/day N` where N is the day number (1-14).

Each invocation executes all tasks for that day from PLAN.md, runs lint and tests, writes
`reports/day-N-<topic>.md`, commits, and pushes to GitHub. On days 7 and 12 it also drafts
blog/LinkedIn/X content via `/blog-draft N`. Run `/gate-check` any time to re-print the Gate 0/1
evaluation decision.

## Current status

- Day 1: COMPLETE — scaffold pctrans package, pyproject.toml, CI workflow, stub modules, `/day` `/blog-draft` `/gate-check` slash commands
- Day 2: COMPLETE — CCLE download client (DepMap 24Q4 via Figshare), lineage filter, Model.csv metadata parse
- Day 3: COMPLETE — TCGA download client (UCSC Xena PANCAN via S3), phenotype lineage filter, gene-ID mismatch flagged for Day 4
- Day 4-14: PENDING

## Project

- GitHub: https://github.com/vthawfeek/pre-clinical-to-clinical-translation
- App: https://pctrans.streamlit.app (live Day 14)
- Working directory: `c:\Users\vthawfeek.Shajitha\Documents\Projects\pre-clinical-to-clinical-translation`

## Quick reference

```
Stack:     Python 3.11, PyTorch 2.2, pandas, scikit-learn, umap-learn, streamlit, plotly
Data:      CCLE RNA-seq (DepMap 24Q4) + TCGA RNA-seq (UCSC Xena Pan-Cancer)
Lineages:  LUAD (~100 cell lines / ~500 patients), BRCA (~130 / ~1,000), SKCM (~70 / ~460)
Features:  Top 2,000 highly variable genes (union-rank HVG method — see Day 4)
Model:     CCLEEncoder [2000→1024→512→256→128→64] + TCGAEncoder [2000→1024→512→256→128→64]
Loss:      SupCon-style multi-positive InfoNCE, learnable log(1/τ) initialised at log(14.3)≈2.66
Batch:     B=48 pairs: 16 LUAD + 16 BRCA + 16 SKCM (SupCon: all same-lineage cross-domain = positive)
Training:  30 epochs, Adam lr=3e-4, cosine LR schedule, Colab T4 (~3 min) or CPU (~12 min)
Gate:      Day 10 kNN@5 Retrieval Accuracy ≥ 70% → DEPLOY; < 70% → DEBUG PROTOCOL
Baselines: Random 33.3%, PCA+kNN ~55%, Harmony ~63%, ours target ≥ 70%
```

## Tech stack

uv, PyTorch, pandas/numpy/pyarrow, scikit-learn, umap-learn, MLflow, Streamlit, plotly, typer, pytest, ruff
