# Preprint Outline — Auditing Contrastive CCLE–TCGA Alignment

> **Framing (read first):** This is deliberately an **audit / benchmark** paper, not a novel-method
> paper. The contribution is a *controlled evaluation* of what supervised cross-domain contrastive
> alignment of cancer cell lines and tumours actually learns — with negative controls, a confounder
> analysis, a difficulty stress-test, a head-to-head against the incumbent method (Celligner), and an
> honestly-reported drug-response case study that returns a null. The honesty *is* the contribution.
> Every claim stays at "research method," never "clinical." Target venues: bioRxiv/arXiv (preprint
> now), then a soundness-based journal (PLOS ONE, PeerJ) or a comp-bio workshop (MLCB, ML4H).

---

## Working Title (primary + alternates)

- **Primary:** "How much of cell-line-to-tumour transcriptional alignment is just lineage signal? A
  controlled audit of contrastive CCLE–TCGA embedding with negative controls and a drug-response
  case study."
- Alt 1: "Auditing cross-domain contrastive alignment of cancer cell lines and tumours: stability,
  confounders, negative controls, and an honest drug-response null."
- Alt 2: "Lineage signal is nearly trivial; drug-response transfer is not: a reproducible benchmark
  of CCLE–TCGA contrastive alignment."

**Author / affiliation:** Thawfeek Varusai, *Independent Researcher*.
*(Recommend registering a free ORCID before submission — it materially helps unaffiliated authors
with journal and preprint-server identity checks.)*

---

## Abstract (structured, ~230 words — draft)

- **Background.** Cell lines are the workhorse of pre-clinical oncology, yet ~85% of drugs that
  succeed in them fail in patients, in part because in-vitro transcriptomes diverge from tumours.
  Several methods (Celligner, PRECISE, CODE-AE) align cell-line and tumour transcriptomes to improve
  translational relevance. We ask a narrower, testable question: *when a supervised contrastive model
  aligns CCLE and TCGA into a shared space, how much of its apparent success is genuine biology
  versus trivially-separable lineage signal — and does it transfer to a functional phenotype?*
- **Methods.** We train a dual-tower encoder with a supervised multi-positive InfoNCE loss on bulk
  RNA-seq (CCLE 24Q4, TCGA PanCancer), and subject it to a battery of controls: multi-seed
  stability, train-only feature selection, real batch-correction baselines, a fully-supervised
  classifier ceiling, a label-shuffle permutation test, a tumour-purity confounder analysis, a
  15-lineage difficulty stress-test, a head-to-head comparison against Celligner, and a BRAF/
  vemurafenib drug-response case study.
- **Results.** Cross-domain lineage retrieval is stable (10-seed kNN@5 0.950 ± 0.034) and robust to
  permutation (p = 0.0099) and purity adjustment, but a plain supervised classifier already reaches
  97.1%, and the contrastive model adds only +2.9 points — coarse lineage is nearly trivially
  separable. Difficulty scales sensibly (15-lineage kNN@5 78.4%, errors 12× enriched on
  biologically adjacent pairs). Embedding proximity recovers BRAF-driver structure weakly
  (p = 0.047) but does **not** predict vemurafenib sensitivity (Spearman ρ = 0.209, p = 0.19, n = 41).
- **Conclusions.** Cross-domain alignment robustly recovers lineage but that signal is largely
  trivial, and — on this task — it does not yet translate to drug response. We release a
  reproducible controls harness for evaluating such claims honestly.

**Keywords:** cancer cell lines, TCGA, contrastive learning, domain alignment, translational
oncology, benchmarking, negative controls, reproducibility.

---

## 1. Introduction

- The translation problem: cell lines vs. tumours, the domain gap (TME absence, serum-driven
  proliferation, plastic adhesion, passage drift), the clinical cost.
- The promise of shared-latent-space methods and why they're attractive.
- **The gap this paper fills:** these methods are usually reported with a headline success metric and
  little adversarial evaluation. Supervised alignment + lineage-retrieval evaluation is especially
  prone to *looking* successful while measuring something trivial or circular.
- **Contributions (state plainly, no overclaim):**
  1. A reproducible *controls harness* (multi-seed CIs, train-only feature selection, permutation
     negative control, purity confounder analysis, supervised ceiling, difficulty stress-test).
  2. A head-to-head comparison of a supervised contrastive aligner against the incumbent Celligner on
     an identical cross-domain retrieval metric.
  3. An honest functional case study (BRAF/vemurafenib) with a reported null, situating what the
     embedding does and does not capture.
- Explicitly *not* claimed: a new clinical tool, a novel architecture, or state-of-the-art drug-
  response prediction.

## 2. Related Work (positioning — the section reviewers will scrutinise)

- **Cell-line↔tumour alignment.**
  - *Celligner* (Warren et al., Nat. Commun. 2021): unsupervised global alignment (contrastive PCA +
    mutual nearest neighbours) to match cell lines to tumour types. **The incumbent; we benchmark
    against it directly (Day 25).** Contrast: Celligner is unsupervised and pan-lineage; ours is
    supervised on lineage — we must be explicit that supervision makes our task *easier*, not harder,
    and frame our retrieval metric accordingly.
  - *PRECISE / PRECISE+* (Mourragui et al., Bioinformatics 2019/2021): domain-adaptation via principal
    vectors to transfer drug-response predictors from models to tumours.
- **Drug-response transfer (the functional endpoint).**
  - *CODE-AE* (He et al., Nat. Mach. Intell. 2022): context-aware deconfounding autoencoder for
    cross-domain drug-response prediction. **The relevant SOTA for our Part B; we position our null
    against it (Day 26)** and do not claim to compete with it — we show that a *lineage-aligned
    embedding alone* is insufficient for drug-response transfer, which motivates methods like CODE-AE.
  - Related: MOLI, AITL, Velodrome, TUGDA (transfer learning for drug response).
- **Batch integration (our generic baselines).** Harmony, ComBat, Scanorama, scGen/scANVI — designed
  to remove technical batch, used here as no-biology-supervision reference points.
- **Representation learning.** SupCon (Khosla et al. 2020), CLIP/InfoNCE (Oord et al. 2018; Radford
  et al. 2021) — the loss family; note we contribute no new learning method.
- **The honest positioning sentence:** *"Our aim is not to outperform Celligner or CODE-AE, but to
  quantify — with controls those papers largely omit — how much of a supervised contrastive aligner's
  apparent success is trivial lineage signal, and whether that signal reaches a functional phenotype."*

## 3. Methods

- **3.1 Data.** CCLE DepMap 24Q4 (log2 TPM+1); TCGA PanCancer (UCSC Xena, log2 norm-count+1). Gene-ID
  harmonisation (HUGO), union-rank HVG selection (n=2,000). 3-lineage (LUAD/BRCA/SKCM) and 15-lineage
  variants. Sample counts table.
- **3.2 Architecture.** Dual-tower MLP encoders (2000→…→64), L2-normalised embeddings, ~5.5M params.
- **3.3 Loss.** Supervised multi-positive InfoNCE (SupCon-style), learnable temperature log(1/τ);
  same-lineage cross-domain pairs = positives. Give the equations.
- **3.4 Evaluation protocol.** Lineage-stratified train/val/test split; scalers + HVG fit on train
  only (Day 16); checkpoint selected on val; all metrics on held-out test. Metrics: cross-domain
  kNN@{1,3,5,10} retrieval, cross-domain silhouette, composite TFS (defined, flagged as bespoke).
- **3.5 Controls harness (the methodological core).** Multi-seed re-split (§Day 15); Wilson +
  bootstrap CIs; real batch-correction baselines + supervised classifier ceiling (§Day 17);
  label-shuffle permutation test (§Day 21); tumour-purity confounder analysis via ESTIMATE
  (stratified kNN + residualised silhouette, §Day 20); 15-lineage difficulty stress-test with
  confusion-structure analysis (§Days 18–19).
- **3.6 Prior-art comparison.** Celligner run on identical CCLE+TCGA input; same kNN@5 retrieval
  metric computed on its aligned embedding (§Day 25). CODE-AE positioning for drug-response (§Day 26).
- **3.7 Functional case study.** BRAF status (CCLE OmicsSomaticMutations; TCGA MC3), vemurafenib
  sensitivity (DepMap PRISM); placement test (Mann-Whitney on distance-to-BRAF-mutant-patient
  centroid) and response-link test (Spearman of proximity vs. vemurafenib AUC).
- **3.8 Reproducibility.** Public data, pinned deps, seeds, CLI + configs released; per-day reports.

## 4. Results (the Phase 2 evidence table is the backbone)

Each subsection = one row of the table below; lead each with the number and its CI, then interpret.

- **4.1 Cross-domain lineage retrieval is stable** (multi-seed).
- **4.2 The result is not a leakage artefact** (train-only HVG).
- **4.3 It beats generic batch correction — but a plain classifier nearly matches it** (baselines +
  supervised ceiling). *This is the pivotal, sobering result; give it prominence, not a footnote.*
- **4.4 Difficulty scales sensibly and errors are biological** (15-lineage + confusion structure).
- **4.5 The signal survives a purity confounder** (stratified + residualised).
- **4.6 The signal is destroyed by label-shuffle** (permutation, p = 0.0099).
- **4.7 Head-to-head vs Celligner** — attempted, not obtained: the published `celligner` PyPI
  release (1.1.0) declares a dependency on a package literally named `umap` (not `umap-learn`),
  which has no installable release on PyPI, so `pip`/`uv` cannot resolve it on any platform without
  a hand-built install from GitHub source; it additionally requires R plus a bundled `mnnpy` build.
  Neither R nor a manual Celligner build is available in this project's environment. Reported as a
  gap, not fabricated — see the Limitations addition below.
- **4.8 Functional case study: weak placement, null drug-response** (BRAF/vemurafenib).

**Evidence table (verbatim results backbone):**

| # | Claim | Experiment | Number (with CI) | Verdict |
|---|---|---|---|---|
| 4.1 | 100% kNN@5 is stable, not a lucky split | 10-seed re-split→train→eval | kNN@5 **0.950 ± 0.034**, CI **[0.932, 0.971]** (min 0.895) | Stable |
| 4.2 | Removing last leakage path doesn't change it | Train-only HVG | Gene-list Jaccard **0.951**; kNN@5 **100.0%** unchanged; silhouette Δ −0.0036 | Negligible leakage |
| 4.3 | Beats real batch correction; ~ties supervised ceiling | Harmony/ceiling on identical test | Harmony **84.2%**; supervised ceiling **97.1%**; contrastive **100%** (**+15.8** vs Harmony, **+2.9** vs ceiling) | Beats batch corr.; **near-trivial margin over ceiling** |
| 4.4 | Generalises beyond 3 easy lineages | 15-lineage retrain+eval | kNN@5 **78.4%**, Wilson CI **[69.8, 85.0]**, n=111 | Genuine headroom |
| 4.4 | 15-lineage errors are biological | Confusion analysis | Named pairs absorb **45.8%** of off-diag mass from **3.8%** of cells (**12×**); LGG/READ→GBM/COAD 100% | Errors track biology |
| 4.5 | Not secretly a purity axis | Purity-stratified + residualised | domain-purity r **−0.455**; kNN@5 **100%** high & low strata; silhouette **+0.566→+0.500** | Survives adjustment |
| 4.6 | Not a label/batch artefact | Label-shuffle, 100 perms | real **78.4%** vs null **7.0–7.7%** (max 17–20%); **p = 0.0099** | Survives control |
| 4.8 | Captures BRAF-driver structure | Part A placement | Mann-Whitney **p = 0.047**; effect **0.649**, CI **[0.465, 0.834]** | Weak positive |
| 4.8 | Proximity predicts vemurafenib response | Part B response-link | Spearman **ρ = 0.209**, CI **[−0.109, 0.493]**, p = 0.19, n=41 | **Null (honest)** |
| 4.7 | vs incumbent method | Celligner head-to-head (identical kNN@5 metric) | Celligner **n/a** (dep unresolvable by pip/uv, needs R+mnnpy from source); PCA 65.8%/25.2%, Harmony 84.2%, ceiling 97.1%, contrastive **100%**/**78.4%** (3-/15-lineage) | *Attempted, dependency unavailable — reported honestly (§4.7, Limitations)* |

## 5. Discussion

- **The headline is a sober one, stated plainly:** cross-domain contrastive alignment robustly
  recovers cancer lineage, but that recovery is *largely trivial* (a plain classifier reaches 97.1%),
  and on this task it does *not* reach a functional drug-response phenotype.
- **Why this matters:** headline retrieval accuracies in this literature can overstate biological
  insight; supervised-alignment-plus-lineage-retrieval is especially vulnerable. The controls harness
  separates "the model learned biology" from "the task was easy."
- **What is genuinely encouraging:** biologically-sensible 15-lineage confusions and survival of
  permutation + purity controls indicate the embedding is not a pure artefact — it encodes real
  lineage structure, and a weak BRAF-driver signal beyond lineage.
- **Where it falls short:** the null Part B and the near-trivial supervised margin show that lineage
  alignment alone is not a translational predictor — motivating deconfounding / drug-supervised
  approaches (CODE-AE) rather than geometry-of-lineage approaches for the functional task.
- **Relation to Celligner (Day 25 update):** a direct numeric head-to-head was attempted but not
  obtained — Celligner's own PyPI packaging is unresolvable by standard tooling (see §4.7) and its
  from-source install requires R, unavailable in this environment. We report this as an open gap
  rather than compare against a literature-cited number for a different evaluation protocol. The
  qualitative point stands regardless: Celligner is unsupervised and would need to *discover* lineage
  structure that our model is handed as a training signal, so a comparable retrieval number from
  Celligner would reinforce "lineage is easy," not threaten our result.

## 6. Limitations (honest, thorough — a strength, not a disclaimer)

- **Task easiness / near-saturation:** kNN@5 ≈ 0.87 at epoch 1; supervised ceiling 97.1%. Coarse
  lineage is nearly linearly separable; the contrastive machinery adds little for this endpoint.
- **Circularity:** supervised on lineage, evaluated on lineage retrieval. The 15-lineage and case
  study partly mitigate, but the primary metric is close to the training objective.
- **Bespoke metric:** TFS is defined here, not externally validated against a clinical outcome; its
  thresholds are chosen. Reported for continuity, not as a validated instrument.
- **Small samples:** 38-line 3-lineage test; n=41 for the vemurafenib response-link. CIs given, but
  power is limited — the Part B null is "no evidence of effect," not "evidence of no effect."
- **Bulk RNA only:** no single-cell resolution; TME/stromal composition partially confounds (purity
  analysed, not eliminated).
- **No external cohort / cross-platform (Rung 3):** all TCGA; generalisation to other cohorts and
  platforms untested.
- **No clinical validation (Rung 5):** nothing here is prospective, pre-registered, or regulatory-
  grade; no diagnostic or treatment claim is made or supported.
- **Single functional case study:** one drug, one driver, one lineage; not a systematic drug-response
  benchmark.
- **No numeric Celligner head-to-head:** the published `celligner` package cannot be installed via
  pip/uv on any platform (broken PyPI metadata: it depends on a nonexistent `umap` release, not
  `umap-learn`) and its from-source path additionally requires R plus a bundled `mnnpy` build, neither
  available in this project's environment. We report reference numbers (PCA, Harmony, supervised
  ceiling, contrastive) on the identical metric so a reader with a working Celligner install can slot
  its number directly into Table T3 rather than trusting a cross-paper comparison.

## 7. Conclusion

- A reproducible, honest audit: lineage alignment is stable and control-robust but largely trivial,
  and does not (here) transfer to drug response. The value is the *evaluation discipline* and the
  released harness, offered as a template for adversarially evaluating cross-domain alignment claims.

## Back matter

- **Data availability:** all public (CCLE DepMap 24Q4; TCGA Xena PanCancer; DepMap PRISM; TCGA MC3;
  ESTIMATE purity). URLs + versions.
- **Code availability:** GitHub repo + tagged release + `PLAN.md`/`PLAN-phase2.md`; per-day reports;
  `reports/*.json` machine-readable results.
- **Funding:** none. **Competing interests:** none. **Ethics:** public de-identified data; no IRB
  required.
- **Author contributions:** sole author.

## Figures & Tables (planned)

- **F1** Before/after: PCA domain gap vs aligned UMAP (3-lineage).
- **F2** Multi-seed kNN@5 distribution + CI; baselines + supervised ceiling bar (the pivotal panel).
- **F3** 15-lineage confusion heatmap with biologically-confusable pairs annotated.
- **F4** Permutation null distribution with real value marked.
- **F5** Purity-stratified retrieval + residualised silhouette.
- **F6** BRAF/vemurafenib: placement scatter + proximity-vs-AUC scatter (with the null fit).
- **F7 (Day 25)** Reference-method bar chart (random/PCA/Harmony/ceiling/contrastive) on the identical
  kNN@5 metric, both lineage variants; Celligner bar absent with an in-figure "dep not installed"
  note (`reports/celligner_comparison.png`, from `pctrans-celligner-compare`).
- **T1** Data/sample counts. **T2** The evidence table above. **T3** Baseline comparison; Celligner
  row marked n/a with the packaging/R blocker, all other methods populated
  (`reports/celligner_comparison.json`).

## Pre-submission checklist

- [x] Day 25 Celligner head-to-head *attempted*; F7/T3 filled with reference numbers + honest n/a for
      Celligner itself (dependency unresolvable in this environment — see Limitations). Revisit with a
      working Celligner install (Colab/Linux + R) before final submission if a numeric comparison is
      required by a target venue.
- [ ] Day 26 CODE-AE positioning written (Part B situated against drug-transfer SOTA).
- [ ] ORCID registered; "Independent Researcher" affiliation set.
- [ ] Preprint on bioRxiv (biology framing) or arXiv q-bio (check endorsement).
- [ ] Verify all citations (Celligner, PRECISE, CODE-AE, SupCon, CLIP) — exact venue/year.
- [ ] Target-journal APC / waiver checked; venue verified against DOAJ/COPE (avoid predatory).
