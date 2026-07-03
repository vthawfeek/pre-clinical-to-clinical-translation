# Pre-Clinical to Clinical Translation

## Plan

- Phase 1 (Days 1–14): [PLAN.md](PLAN.md) (also at `C:\Users\vthawfeek.Shajitha\.claude\plans\use-the-below-plan-concurrent-kahan.md` for plan mode)
- Phase 2 (Days 15–25): [PLAN-phase2.md](PLAN-phase2.md) — rigorous validation (CIs, real baselines, 15-lineage task, purity confounder, label-shuffle control, vemurafenib/BRAF case study)

## How to trigger a day's work

Type `/day N` where N is the day number (1-25). The `/day` command routes Days 1–14 to `PLAN.md`
and Days 15–25 to `PLAN-phase2.md`.

Each invocation executes all tasks for that day from the active plan file, runs lint and tests,
writes `reports/day-N-<topic>.md`, commits, and pushes to GitHub. On blog-milestone days (7, 12, 25)
it also drafts blog/LinkedIn/X content via `/blog-draft N`. Run `/gate-check` any time to re-print
the Gate 0/1 (and, once Day 24 lands, Gate 2) evaluation decision.

## Current status

- Day 1: COMPLETE — scaffold pctrans package, pyproject.toml, CI workflow, stub modules, `/day` `/blog-draft` `/gate-check` slash commands
- Day 2: COMPLETE — CCLE download client (DepMap 24Q4 via Figshare), lineage filter, Model.csv metadata parse
- Day 3: COMPLETE — TCGA download client (UCSC Xena PANCAN via S3), phenotype lineage filter, gene-ID mismatch flagged for Day 4
- Day 4: COMPLETE — FeatureSynchroniser (gene-ID harmonisation, union-rank HVG selection), pctrans-preprocess CLI, EDA notebook
- Day 5: COMPLETE — DataSplitter (lineage-stratified split, train-only pooled scalers), CCLEDataset/TCGADataset, StratifiedContrastiveBatchSampler, --split CLI
- Day 6: COMPLETE — CCLEEncoder/TCGAEncoder (shared MLP template, separate weights), DualTowerModel (L2-normalised), SupConInfoNCELoss (learnable log(1/τ)); ~5.5M params
- Day 7: COMPLETE — ContrastiveTrainer (Adam, cosine+warmup LR, grad-clip, MLflow, checkpoint+early-stop), KNNValidationCallback, pctrans-train CLI, blog-01 draft; Gate 0 PASS (epoch-1 loss finite, val kNN@5 0.816, τ 0.070)
- Day 8: COMPLETE — full training run (early-stopped epoch 7/30, patience 5), best_model.pt saved (epoch 2, val kNN@5 0.9474), MLflow experiment pctrans-v1, hyperparameter mini-sweep A/B/C (all >0.92, default retained); τ stable 0.070→0.074, no collapse
- Day 9: COMPLETE — 02_training_analysis notebook (5 panels), trainer grad-norm history + on_epoch_end hook; best val kNN@5 0.9737 (epoch 2), τ stable 0.070→0.073, towers balanced (no 10× asymmetry), SKCM hardest (0.9375), domain centroid cosine 0.251 (no collapse); embed_dim ablation 32/64/128 all ≥0.89, embed_dim confirmed 64
- Day 10: COMPLETE — evaluation modules (knn/silhouette/tfs), pctrans-evaluate CLI, PCA baseline; Gate 1 PASS → DEPLOY (test kNN@5 100%, kNN@1 97.4%, silhouette +0.57, TFS 0.89; PCA+kNN baseline 65.8%, random 33.3%; per-cell-line TFS ranked, lowest ACH-000264 LUAD 0.662)
- Day 11: COMPLETE — viz.py (umap_projection, lineage/domain scatter, TFS ranking bar, static + before/after renderers), pctrans-visualize CLI, embeddings_test.npz; UMAP + before/after + TFS figures rendered; outlier ACH-000264=Calu-6 (anaplastic NSCLC, TFS 0.662), bottom BRCA all TNBC/basal; tightness BRCA +0.861 > SKCM +0.802 > LUAD +0.787; 03_evaluation notebook (5 sections)
- Day 12: COMPLETE — Streamlit app (app/streamlit_app.py: sidebar dropdown grouped by lineage, live UMAP with star-highlighted cell line + 5 nearest-patient hexagons, TFS gauge, neighbours table with tumour stage/histology), pctrans-precompute CLI → ccle_embeddings.npz (259 cell lines × 64-d) + app_meta.json (259 names, 339 TCGA annotations, deploy-safe); app runs on held-out test embeddings, AppTest render clean; lowest test TFS CALU6/ACH-000264 0.662 (no <0.4 case exists); blog-02 + LinkedIn-02 + X-thread-02 drafts
- Day 13: COMPLETE — 5 docs (data pipeline, feature eng, architecture, training, evaluation), README with test-set results table + ASCII diagram; implemented TranslationEmbedder (inference/api.py) + pctrans-query CLI; new test_inference.py + test_pipeline.py (end-to-end train→evaluate→visualize→query on synthetic session fixture), download idempotency + save_filtered tests; coverage 54%→85% (target ≥80%), 79 tests pass
- Day 14: COMPLETE — Phase 1 wrap + launch prep: requirements.txt (cross-platform runtime pins), notebooks/colab_quickstart.ipynb (download→train→UMAP), app data artefacts git-tracked (embeddings_test.npz, ccle_embeddings.npz, app_meta.json) for Streamlit Cloud, final code review (no hardcoded paths), v0.1.0 tag + GitHub release + repo topics; Phase 2 harness wired (`/day` routes 15–25 → PLAN-phase2.md). Manual/deferred: Streamlit Cloud connect + social posting (see reports/day-14-launch.md). Repo already public.

## Phase 2 status (Days 15–25 — PLAN-phase2.md)

- Day 15: COMPLETE — evaluation/stats.py (Wilson + bootstrap CIs, seed aggregation), pctrans-multiseed harness, pctrans-evaluate now prints/stores CIs; single-split kNN@5 100% (Wilson 90.8–100%, n=38); 10-seed (42–51) kNN@5 0.950±0.034 CI [0.932,0.971] (min 0.895), TFS 0.910±0.038 — stable, not a lucky split; test_stats.py (8 tests), 83 tests pass
- Day 16: COMPLETE — train-only HVG selection (`--hvg-on all|train` flag, split-then-select), gene_list_trainhvg.txt + best_model_trainhvg.pt; leakage-delta analysis: gene-list Jaccard 0.9512, test kNN@5 unchanged at 100%, silhouette −0.0036, TFS −0.0009 — leakage confirmed negligible, Phase-1 numbers stand
- Day 17: COMPLETE — pctrans/evaluation/baselines.py (harmony_knn/combat_knn/scanorama_knn/supervised_ceiling), pctrans-baselines CLI → reports/baselines.json; real numbers on test data: PCA 65.8%, Harmony 84.2% (harmonypy, real), ComBat/Scanorama n/a locally (no prebuilt wheels, need C++ toolchain — optional `baselines` extra, None-safe), supervised ceiling 97.1% (CCLE→TCGA logistic regression, no alignment); contrastive still wins at 100% (+15.8pts vs best real baseline, +2.9pts vs the fully-supervised ceiling); CI scoped to `uv sync --extra dev` to keep fragile native deps opt-in
- Day 18: PENDING — config-driven lineages, 15-lineage data + training run
- Day 19: PENDING — 15-lineage evaluation, error-structure biology analysis
- Day 20: PENDING — tumour-purity confounder analysis (stratified + residualised)
- Day 21: PENDING — label-shuffle permutation negative control, empirical p-value
- Day 22: PENDING — vemurafenib + BRAF mutation data assembly (DepMap/GDSC/MC3)
- Day 23: PENDING — BRAF/vemurafenib placement + response-link case study
- Day 24: PENDING — Gate 2 evaluation, phase2-summary, README/CLAUDE claims updated
- Day 25: PENDING — blog-03 validation story, LinkedIn/X drafts, Phase 2 complete

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
Baselines: Random 33.3%, PCA+kNN 65.8%, Harmony 84.2% (real, Day 17), ours 100% (Wilson CI 90.8–100%)
```

## Tech stack

uv, PyTorch, pandas/numpy/pyarrow, scikit-learn, umap-learn, MLflow, Streamlit, plotly, typer, pytest, ruff
