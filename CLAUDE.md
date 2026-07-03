# Pre-Clinical to Clinical Translation

## Plan

- Phase 1 (Days 1–14): [PLAN.md](PLAN.md) (also at `C:\Users\vthawfeek.Shajitha\.claude\plans\use-the-below-plan-concurrent-kahan.md` for plan mode)
- Phase 2 (Days 15–27): [PLAN-phase2.md](PLAN-phase2.md) — rigorous validation (CIs, real baselines, 15-lineage task, purity confounder, label-shuffle control, vemurafenib/BRAF case study) + Phase 2E prior-art benchmark & preprint (Days 25–27: Celligner head-to-head, CODE-AE positioning, preprint assembly)

## How to trigger a day's work

Type `/day N` where N is the day number (1-27). The `/day` command routes Days 1–14 to `PLAN.md`
and Days 15–27 to `PLAN-phase2.md`.

Each invocation executes all tasks for that day from the active plan file, runs lint and tests,
writes `reports/day-N-<topic>.md`, commits, and pushes to GitHub. On blog-milestone days (7, 12, 27)
it also drafts blog/LinkedIn/X content via `/blog-draft N`. Run `/gate-check` any time to re-print
the Gate 0, 1, and 2 evaluation decisions.

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

## Phase 2 status (Days 15–27 — PLAN-phase2.md)

- Day 15: COMPLETE — evaluation/stats.py (Wilson + bootstrap CIs, seed aggregation), pctrans-multiseed harness, pctrans-evaluate now prints/stores CIs; single-split kNN@5 100% (Wilson 90.8–100%, n=38); 10-seed (42–51) kNN@5 0.950±0.034 CI [0.932,0.971] (min 0.895), TFS 0.910±0.038 — stable, not a lucky split; test_stats.py (8 tests), 83 tests pass
- Day 16: COMPLETE — train-only HVG selection (`--hvg-on all|train` flag, split-then-select), gene_list_trainhvg.txt + best_model_trainhvg.pt; leakage-delta analysis: gene-list Jaccard 0.9512, test kNN@5 unchanged at 100%, silhouette −0.0036, TFS −0.0009 — leakage confirmed negligible, Phase-1 numbers stand
- Day 17: COMPLETE — pctrans/evaluation/baselines.py (harmony_knn/combat_knn/scanorama_knn/supervised_ceiling), pctrans-baselines CLI → reports/baselines.json; real numbers on test data: PCA 65.8%, Harmony 84.2% (harmonypy, real), ComBat/Scanorama n/a locally (no prebuilt wheels, need C++ toolchain — optional `baselines` extra, None-safe), supervised ceiling 97.1% (CCLE→TCGA logistic regression, no alignment); contrastive still wins at 100% (+15.8pts vs best real baseline, +2.9pts vs the fully-supervised ceiling); CI scoped to `uv sync --extra dev` to keep fragile native deps opt-in
- Day 18: COMPLETE — `build_lineage_maps` makes LINEAGE_TO_IDX config-driven (order-preserving, Phase-1 default byte-identical), `FeatureSynchroniser(lineages=...)`, 17 new CCLE OncotreeAlias entries (LUSC/COAD/READ/PAAD/STAD/LIHC/KIRC/HNSC/GBM/LGG/OV/BLCA), `configs/data_15.yaml` + `configs/training_15.yaml` (batch_size 120, 4/lineage/domain); discovered + fixed a real NaN gap in the Xena PANCAN matrix (`drop_incomplete_genes`, 2,946 of 16,568 common genes affected, zero-impact on the 3-lineage pipeline); real 15-lineage run: 734 CCLE + 7,136 TCGA samples, best val kNN@5 0.829 (vs 3-lineage 0.974 — genuine headroom), weakest lineages LGG/READ (0%) are exactly the smallest cohorts and the named confusable pairs, promising signal for Day 19
- Day 19: COMPLETE — real 15-lineage test evaluation (`pctrans-evaluate --data-config configs/data_15.yaml`: kNN@5 78.4%, Wilson CI 69.8–85.0%, n=111; silhouette +0.70; TFS 0.82; vs 3-lineage 100%), `knn.py`/`viz.py`/`evaluate.py`/`visualize.py` generalised to arbitrary lineage counts via optional `idx_to_lineage`/`lineage_order` overrides, new `top_confusions`/`confusable_pair_mass`/`confusion_matrix_heatmap`; error structure is biologically sensible — LGG (0%) and READ (0%) send 100% of their misses to their named partners (GBM, COAD), LUSC sends its misses to LUAD/HNSC, named pairs absorb 45.8% of off-diagonal error mass from just 3.8% of possible cells (12x enrichment); 15-lineage UMAP + confusion heatmap rendered, `03_evaluation.ipynb` Section 6 added and executed end-to-end (also fixed a Day-17 regression in Section 5's stale baseline cell)
- Day 20: COMPLETE — TCGAClient.download_purity (ABSOLUTE PanCanAtlas mastercalls, GDC-hosted), `pctrans/evaluation/confounders.py` (domain-axis/purity correlation, purity-stratified kNN, purity-residualised silhouette), `purity_confounder_panel` viz, `pctrans-confounders` CLI; real numbers on the 3-lineage test embeddings: domain-axis/purity r = −0.455 (n=333, moderate not dominant), kNN@5 100% in both high- and low-purity TCGA strata (n=153/142), silhouette +0.566 → +0.500 after purity residualisation (lineage cohesion survives) — Gate 2 criterion G2-4 PASS
- Day 21: COMPLETE — `permutation_test()` (generic empirical p-value + null distribution) in `pctrans/evaluation/stats.py`, `permutation_null_panel` viz, `pctrans-permutation-test` CLI (eval-only label-shuffle + retrain-based label-shuffle variants); real 15-lineage run (100 perms x 5-epoch short retrain): null mean 7.0–7.7% (chance 6.7%), max 17–20%, real kNN@5 78.4% never approached, p = 0.0099 on both variants (target p<0.01 met) — Gate 2 negative-control criterion PASS
- Day 22: COMPLETE — `pctrans/casestudy/braf_vemurafenib.py` (PrismClient for DepMap PRISM Repurposing 20Q2 vemurafenib AUC, CBioPortalClient for BRAF calls, `assemble_braf_table`), `pctrans-casestudy` CLI → `data/processed/braf_vemurafenib.parquet` + `reports/braf_coverage.json`; real numbers: 61 SKCM cell lines with BRAF status (47 mutant/14 WT) via cBioPortal `ccle_broad_2019`, 41 of those also have a PRISM vemurafenib AUC (33 mutant/8 WT); 65 SKCM test patients with BRAF status via `skcm_tcga_pan_can_atlas_2018` (32 mutant/33 WT); classic CCLE/GDSC panels only tested vemurafenib's precursor PLX4720, not vemurafenib itself — PRISM Repurposing was the real source
- Day 23: COMPLETE — Part A/B placement + response-link analysis in `pctrans/casestudy/braf_vemurafenib.py` (`braf_placement_test`, `braf_response_link`), `braf_casestudy_panel`/`_interactive` viz, `pctrans-casestudy-analysis` CLI → `reports/braf_casestudy.json` + `reports/braf_vemurafenib.{png,html}`, `03_evaluation.ipynb` Section 7; real result: Part A weak positive (BRAF-mutant SKCM lines closer to BRAF-mutant patient centroid, Mann-Whitney p=0.047, effect 0.649, 95% CI [0.465, 0.834]), Part B null (Spearman rho=0.209, 95% CI [-0.109, 0.493], p=0.19, n=41) — reported honestly, not buried
- Day 24: COMPLETE — Gate 2 extended into `/gate-check` (`.claude/commands/gate-check.md`), `reports/phase2-summary.md` (evidence table: rung → claim → experiment → number+CI → verdict), README/CLAUDE headline claims hardened (bare "100%" → "100% (95% CI 90.8–100%, n=38); stable across 10 seeds; beats Harmony/ComBat/Scanorama; survives label-shuffle (p<.01) and purity adjustment"), README "Limitations & Scope" section added (Rungs 3/5 out of scope, research method not clinical tool); Gate 2 decision: **PORTFOLIO-READY** — G2-1 through G2-4 all PASS (multi-seed CI lower bound 0.932 ≥ 0.90; beats Harmony by +15.8pts and exceeds supervised ceiling; permutation p=0.0099; purity strata both 100% kNN@5), G2-5 reports the expected 15-lineage headroom (78.4%) with biologically sensible confusions, G2-BONUS reports the vemurafenib case study honestly (Part A weak positive, Part B null)
- Day 25: COMPLETE — `pctrans/evaluation/celligner_compare.py` (`run_celligner`/`retrieval_on_embedding`, None-safe like Day 17's ComBat/Scanorama), `pctrans-celligner-compare` CLI → `reports/celligner_comparison.json` + Figure F7 (`celligner_comparison_panel`); ran the identical kNN@5+silhouette metric on both lineage variants against every reference method (3-lineage: random 33.3%, PCA 65.8%, Harmony 84.2%, ceiling 97.1%, contrastive 100%; 15-lineage: random 6.7%, PCA 25.2%, contrastive 78.4%); **Celligner itself n/a** — its PyPI release depends on a nonexistent `umap` package (not `umap-learn`), unresolvable by pip/uv on any platform, plus needs R (absent here) for a from-source build — reported as an honest gap in `reports/preprint-outline.md` rather than fabricated; plan file extended to Days 25–27 (Phase 2E: Celligner benchmark → CODE-AE positioning → preprint assembly), superseding the original Day-25 blog-post plan
- Day 26: PENDING — drug-response-transfer probe (`drug_signal_retained`) + CODE-AE positioning
- Day 27: PENDING — preprint draft assembly + blog-03 validation story, LinkedIn/X drafts, Phase 2 complete

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
Gate 2:    PORTFOLIO-READY (Day 24) — stable across 10 seeds (0.950±0.034, CI [0.932,0.971]),
           beats Harmony by +15.8pts, survives label-shuffle (p=0.0099) and purity adjustment,
           15-lineage 78.4% with sensible confusions, vemurafenib case study weak+null (honest)
```

## Tech stack

uv, PyTorch, pandas/numpy/pyarrow, scikit-learn, umap-learn, MLflow, Streamlit, plotly, typer, pytest, ruff
