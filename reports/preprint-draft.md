# How much of cell-line-to-tumour transcriptional alignment is just lineage signal? A controlled audit of contrastive CCLE–TCGA embedding with negative controls and a drug-response case study

**Thawfeek Varusai**, Independent Researcher

*Manuscript type: audit / benchmark. Target venues: bioRxiv (q-bio) preprint now; PLOS ONE, PeerJ,
or a comp-bio workshop (MLCB, ML4H) for peer review.*

---

## Abstract

**Background.** Cell lines are the workhorse of pre-clinical oncology, yet roughly 85% of drugs that
succeed in them fail in patients, in part because in-vitro transcriptomes diverge from tumour
transcriptomes. Several methods (Celligner, PRECISE, CODE-AE) align cell-line and tumour
transcriptomes to improve translational relevance. We ask a narrower, testable question: when a
supervised contrastive model aligns CCLE and TCGA into a shared embedding space, how much of its
apparent success is genuine biology versus trivially-separable lineage signal, and does that
alignment transfer to a functional drug-response phenotype?

**Methods.** We train a dual-tower encoder (CCLE tower, TCGA tower, ~5.5M parameters, 64-d
L2-normalised output) with a supervised multi-positive InfoNCE (SupCon-style) loss on bulk RNA-seq
(CCLE DepMap 24Q4; TCGA PanCancer via UCSC Xena), where same-lineage cell-line/patient pairs are
positives. We subject the result to seven controls: multi-seed stability, train-only feature
selection, real batch-correction baselines, a fully-supervised classifier ceiling, a label-shuffle
permutation test, a tumour-purity confounder analysis, and a 15-lineage difficulty stress-test,
followed by a head-to-head comparison against the incumbent alignment method (Celligner) and a
BRAF/vemurafenib melanoma drug-response case study.

**Results.** Cross-domain lineage retrieval (kNN@5) is 100% on the original 3-lineage, 38-cell-line
test set (Wilson 95% CI 90.8–100%) and stable across ten independent train/test re-splits (mean
95.0% ± 3.4%, 95% CI [93.2, 97.1]). It survives train-only feature selection (gene-list Jaccard
0.951 vs. all-sample HVG selection; Δ kNN@5 = 0), beats real batch-correction (Harmony 84.2%, +15.8
points), and survives a label-shuffle permutation test (real 78.4% on the 15-lineage task vs. null
mean 7.0–7.7%, empirical p = 0.0099) and a tumour-purity confounder check (kNN@5 100% in both high-
and low-purity TCGA strata; silhouette +0.566 → +0.500 after purity residualisation). But a plain
fully-supervised classifier, trained on CCLE alone with no cross-domain alignment step at all,
already reaches 97.1% on the same TCGA test set — the contrastive model's advantage over that
ceiling is 2.9 points. Scaling to 15 lineages drops kNN@5 to 78.4% (Wilson CI [69.8, 85.0], n=111),
and the errors concentrate 12× more than chance on biologically adjacent pairs (LUAD↔LUSC,
GBM↔LGG, COAD↔READ). A numeric head-to-head against Celligner was attempted but not obtained: its
published PyPI release has an unresolvable dependency and its from-source path requires R, neither
available in this project's environment; reference numbers for all other methods are reported on
the identical metric so a reader with a working Celligner install can complete the row. Within the
BRAF/vemurafenib melanoma case study, embedding proximity to a BRAF-mutant patient centroid is a
weak positive marker of BRAF-mutant cell-line identity (Mann-Whitney p = 0.047, effect size 0.649,
95% CI [0.465, 0.834]) but does not predict vemurafenib sensitivity (Spearman ρ = 0.209, 95% CI
[−0.109, 0.493], p = 0.19, n = 41). A within-CCLE cross-validated probe shows the 64-d embedding
does not clearly underperform raw 2,000-gene expression at recovering vemurafenib AUC (both weak,
negative out-of-fold R² at this n), arguing the null is closer to an underpowered-task problem than
a proof that alignment selectively destroyed drug-response signal.

**Conclusions.** Cross-domain contrastive alignment robustly recovers cancer lineage, but that
recovery is largely trivial — a supervised classifier with no alignment step nearly matches it — and
on this task it does not yet transfer to a drug-response phenotype. We release a reproducible
controls harness (multi-seed CIs, permutation test, purity-confounder analysis, real
batch-correction baselines, supervised ceiling, difficulty stress-test) as a template for
adversarially evaluating this class of claim.

**Keywords:** cancer cell lines, TCGA, contrastive learning, domain alignment, translational
oncology, benchmarking, negative controls, reproducibility.

---

## 1. Introduction

Cancer cell lines remain the dominant pre-clinical model for oncology drug discovery: cheap,
scalable, and genetically tractable. But cell lines grown in plastic dishes with serum-driven
proliferation and no tumour microenvironment diverge transcriptionally from the tumours they are
meant to model, and that divergence is implicated in the high attrition rate of oncology drugs
between pre-clinical success and clinical failure. A natural response is to learn a shared latent
space in which cell lines and tumours are directly comparable, so that a cell line's position
relative to real patient tumours becomes a signal about how trustworthy it is as a pre-clinical
model. Celligner (Warren et al., 2021), PRECISE/PRECISE+ (Mourragui et al., 2019, 2021), and CODE-AE
(He et al., 2022) are three published approaches to this general problem, each with a different
mechanism and a different downstream goal (visualisation/matching, drug-response-predictor transfer,
and deconfounded drug-response prediction respectively).

This paper is not a fourth method. It trains a straightforward supervised dual-tower contrastive
model on the same class of data these methods use, and then asks the question a reviewer should ask
of any such result before believing it: *how much of the apparent success is real biology, and how
much is an easy task dressed up as a hard one?* Supervised alignment evaluated by lineage retrieval
is a particularly easy trap, because the evaluation metric is close to the training objective:
reporting kNN retrieval accuracy on the same axis the model was supervised on can look like evidence
of learned biology when it is closer to evidence that the model memorised its own training signal.

**Contributions.**

1. A reproducible *controls harness* — multi-seed confidence intervals, train-only feature selection
   (removing an unsupervised leakage path), a label-shuffle permutation negative control, a
   tumour-purity confounder analysis, a fully-supervised classifier ceiling, and a difficulty
   stress-test that scales the task from 3 to 15 lineages including deliberately confusable pairs.
2. A head-to-head comparison of a supervised contrastive aligner against the incumbent unsupervised
   method (Celligner) on an identical cross-domain retrieval metric, reported honestly including the
   case where the comparison could not be completed for environment reasons.
3. An honest functional case study (BRAF-mutant melanoma / vemurafenib) that returns a weak positive
   on driver-structure placement and a null on the response-magnitude question, situated against the
   drug-response-transfer literature (CODE-AE) rather than buried or reframed as success.

We explicitly do **not** claim a new clinical tool, a novel architecture, or state-of-the-art
drug-response prediction. Every number in this paper is a retrieval or association statistic on
public bulk RNA-seq data.

## 2. Related Work

**Cell-line↔tumour alignment.** Celligner (Warren et al., *Nat. Commun.* 2021) performs unsupervised,
pan-lineage alignment of cell lines to tumour types using contrastive PCA and mutual nearest
neighbours; it is the closest published incumbent and the method we attempt to benchmark against
directly (§4.7). The comparison is asymmetric by design: Celligner receives no lineage labels and
must discover tumour-type structure unsupervised, while our model is handed lineage as a training
signal, which should make our retrieval task strictly easier. A comparable retrieval number from
Celligner therefore reinforces this paper's central claim ("lineage is nearly trivial"), not
threatens it. PRECISE/PRECISE+ (Mourragui et al., *Bioinformatics* 2019, 2021) instead learns
domain-adaptation principal vectors specifically to transfer drug-response predictors from
cell-line models to tumours, a narrower and more directly translational goal than ours.

**Drug-response transfer.** CODE-AE (He et al., *Nat. Mach. Intell.* 2022) is the relevant
state-of-the-art for the functional half of this paper. It is a context-aware deconfounding
autoencoder: cell-line and tumour representations are factorised into a shared domain-invariant code
plus domain-specific confounder codes, trained end-to-end under drug-response supervision so the
shared code is explicitly optimised to be predictive of response. Related transfer-learning
approaches for drug response include MOLI, AITL, Velodrome, and TUGDA. We do not attempt to compete
with CODE-AE; §5 uses it to explain why our lineage-supervised, drug-agnostic embedding is not
expected to transfer to a continuous drug-response phenotype, and to situate our Part-B null as
consistent with, rather than contradicting, that literature.

**Batch integration.** Harmony, ComBat, and Scanorama are generic batch-correction methods designed
to remove technical (not necessarily domain) variation; we use them here purely as no-biology,
no-supervision reference points, not as competing translational methods.

**Representation learning.** The loss family used here (SupCon, Khosla et al. 2020; InfoNCE,
Oord et al. 2018; CLIP, Radford et al. 2021) is well established. We contribute no new learning
method — the contribution is entirely in the evaluation.

**The honest positioning sentence.** Our aim is not to outperform Celligner or CODE-AE, but to
quantify, with controls those papers largely omit, how much of a supervised contrastive aligner's
apparent success is trivial lineage signal, and whether that signal reaches a functional phenotype.

## 3. Methods

**3.1 Data.** CCLE expression from DepMap 24Q4 (log2(TPM+1)); TCGA PanCancer expression from UCSC
Xena (log2(norm_count+1)). Gene identifiers harmonised via HUGO symbols; the top 2,000 genes by a
union-rank highly-variable-gene (HVG) score are retained as features. Two lineage configurations are
used: a 3-lineage variant (LUAD, BRCA, SKCM; 259 CCLE lines, ~2,000 TCGA patients total across
splits) and a 15-lineage variant (LUAD, LUSC, BRCA, SKCM, COAD, READ, PAAD, STAD, LIHC, KIRC, HNSC,
GBM, LGG, OV, BLCA; 734 CCLE lines, 7,136 TCGA patients), the latter deliberately including three
lung/colorectal/glioma pairs known to be transcriptionally similar.

**3.2 Architecture.** Two independent MLP encoder towers (CCLE tower, TCGA tower; 2,000 → 1,024 →
512 → 256 → 128 → 64), separate weights, L2-normalised 64-d output embeddings; ~5.5M parameters
total.

**3.3 Loss.** Supervised multi-positive InfoNCE (SupCon-style): within a batch, all cross-domain pairs
sharing a lineage label are treated as positives against all other in-batch samples as negatives,
with a learnable inverse-temperature log(1/τ) initialised at log(14.3) ≈ 2.66.

**3.4 Evaluation protocol.** A lineage-stratified train/val/test split is drawn once per seed; feature
scalers and (from Day 16 onward) HVG selection are fit on the training slice only, eliminating an
earlier mild unsupervised leakage path. Checkpoint selection uses validation kNN@5; all reported
metrics are computed once, on the held-out test split. Primary metrics: cross-domain kNN@{1,3,5,10}
retrieval accuracy (does a cell line's *k* nearest TCGA neighbours share its lineage majority),
cross-domain silhouette score by lineage, and a composite Translational Fidelity Score (TFS) blending
per-sample neighbour match fraction with silhouette contribution — TFS is defined for this project
and is explicitly *not* an externally validated clinical instrument.

**3.5 Controls harness.** (a) *Multi-seed re-split*: ten independent seeds (42–51), each re-running
split → HVG(train-only) → train → test-eval end-to-end, testing whether the headline number depends
on one lucky 38-line test set. (b) *Wilson and bootstrap confidence intervals* on every point metric.
(c) *Real batch-correction baselines* (Harmony via `harmonypy`; ComBat and Scanorama attempted but
unavailable as prebuilt wheels in this environment) run on the identical test embeddings, plus a
*fully-supervised classifier ceiling* — logistic regression trained on CCLE expression only, with no
cross-domain alignment step, evaluated on the TCGA test set — establishing how much lineage signal is
recoverable without any alignment machinery at all. (d) *Label-shuffle permutation test*: 100
permutations that destroy the CCLE-lineage↔TCGA-lineage correspondence the loss relies on (both an
eval-only variant and a short-retrain variant), collecting a null kNN@5 distribution and an empirical
p-value. (e) *Tumour-purity confounder analysis* using ABSOLUTE/ESTIMATE-derived purity scores: the
correlation between each sample's projection on the CCLE→TCGA domain axis and its purity; a
purity-stratified retrieval check (high- vs. low-purity TCGA halves); and a purity-residualised
silhouette. (f) *15-lineage difficulty stress-test* with a confusion-matrix analysis of whether
errors concentrate on biologically adjacent lineage pairs.

**3.6 Prior-art comparison.** Celligner is run, where possible, on the identical HVG-filtered CCLE and
TCGA matrices, and the resulting joint embedding is scored with the exact same cross-domain kNN@5 and
silhouette functions used for our own model, so the alignment method is the only variable that
differs (§4.7). CODE-AE is treated as a literature comparison, not re-implemented (§5).

**3.7 Functional case study.** BRAF V600 mutation status for CCLE melanoma lines (DepMap
OmicsSomaticMutations) and TCGA SKCM patients (cBioPortal `skcm_tcga_pan_can_atlas_2018`); vemurafenib
sensitivity (AUC) for CCLE lines from the DepMap PRISM Repurposing 20Q2 dataset (the only
vemurafenib-proper readout available; classic CCLE/GDSC panels tested only its precursor, PLX4720).
*Part A (placement)*: Mann-Whitney U test on distance-to-BRAF-mutant-patient-centroid between
BRAF-mutant and BRAF-wild-type SKCM cell lines. *Part B (response link)*: Spearman correlation
between that same centroid-proximity and PRISM vemurafenib AUC. *Day-26 drug-signal-retained probe*:
within-CCLE, 5-fold cross-validated out-of-fold R²/Spearman ρ predicting vemurafenib AUC from three
feature blocks — BRAF status alone, raw 2,000-gene HVG expression, and the 64-d embedding — using a
per-fold RidgeCV to keep the high-dimensional raw-expression block from overfitting at n≈33
training rows per fold.

**3.8 Reproducibility.** All data are public. Code, configs, seeds, and a `pctrans-*` CLI for every
analysis are released on GitHub; every experimental day has a corresponding dated report with
verbatim command output.

## 4. Results

### 4.1 Cross-domain lineage retrieval is stable across independent splits

On the original 3-lineage, 38-cell-line test split, kNN@5 is 100.0% (Wilson 95% CI 90.8–100.0%). Ten
independent re-splits (seeds 42–51), each re-running the full split→HVG→train→eval pipeline, give a
mean kNN@5 of 95.0% ± 3.4% (95% CI [93.2%, 97.1%], minimum 89.5% across all ten seeds) — see Figure
F2, left panel. The CI lower bound of 93.2% comfortably clears the pre-registered Gate-2 stability
threshold of 90%: the headline number is not an artefact of one favourable 38-line test set.

### 4.2 The result is not a leakage artefact

Moving HVG gene selection from "fit on all samples, then split" to "split first, fit on the training
slice only" changes the retained gene list only modestly (Jaccard overlap 0.951 against the
all-sample list) and leaves test kNN@5 exactly unchanged at 100.0% (silhouette Δ = −0.0036, TFS
Δ = −0.0009). The one unsupervised leakage path this project had is confirmed negligible; the
Phase-1 numbers stand under the stricter protocol.

### 4.3 It beats generic batch correction — but a plain classifier nearly matches it

This is the pivotal, sobering result of the paper and is given prominence rather than a footnote
(Figure F2, right panel). On the identical 3-lineage test embeddings: random guessing scores 33.3%;
PCA followed by kNN scores 65.8%; Harmony (real batch integration via `harmonypy`) scores 84.2%; and
a plain logistic-regression classifier trained on CCLE expression alone, with **no cross-domain
alignment step whatsoever**, evaluated directly on TCGA test patients, scores 97.1%. The contrastive
model reaches 100.0%. It beats the best real batch-correction baseline by 15.8 points, which is a
genuine and meaningful margin, but it beats the *supervised, no-alignment ceiling* by only 2.9
points. Coarse cancer lineage is nearly linearly separable in bulk RNA-seq without any domain
adaptation at all; the contrastive machinery is doing comparatively little additional work for this
specific endpoint. ComBat and Scanorama could not be run in this environment (no prebuilt wheels for
their native dependencies) and are reported as not-available rather than estimated.

### 4.4 Difficulty scales sensibly and the errors are biological

Scaling from 3 to 15 lineages (734 CCLE lines, 7,136 TCGA patients, including deliberately confusable
pairs: LUAD/LUSC, COAD/READ, GBM/LGG, plus HNSC as a squamous confound for LUSC) drops test kNN@5 to
78.4% (Wilson 95% CI [69.8%, 85.0%], n=111) — genuine headroom on a genuinely harder task. Critically,
the errors are not random: named biologically-confusable pairs absorb 45.8% of all off-diagonal
confusion mass while accounting for only 3.8% of possible confusion-matrix cells, a 12× enrichment.
The two smallest and hardest cohorts, LGG and READ (0% per-lineage accuracy), send 100% of their
misses to their named biological partners, GBM and COAD respectively (Figure F3). This is the
strongest evidence in the paper that the model encodes real cancer-lineage biology rather than an
arbitrary shortcut: when it fails, it fails toward tumours that are genuinely similar.

### 4.5 The signal survives a tumour-purity confounder check

A plausible alternative explanation for "alignment" is that the model has learned *pure cultured cell
vs. stroma-contaminated tumour*, not cancer lineage identity. The correlation between each sample's
projection onto the CCLE→TCGA domain axis and its ABSOLUTE/ESTIMATE-derived tumour purity is r = −0.455
(n=333) — a moderate association, expected and not concerning on its own, since purity is confounded
with domain by construction (cell lines are ≈100% pure). What matters is whether lineage retrieval
holds up when purity is controlled: kNN@5 is 100% in both the high-purity (n=153) and low-purity
(n=142) TCGA test halves, and cross-domain silhouette survives purity residualisation at +0.500 (down
from +0.566 unresidualised, still strongly positive) — Figure F5. Lineage cohesion is not primarily a
purity axis in disguise.

### 4.6 The signal collapses under label-shuffle

The final negative control asks whether the model is exploiting some artefact of label or batch
structure that requires no real biological learning. Shuffling the CCLE-lineage↔TCGA-lineage
correspondence across 100 permutations (both an eval-only variant on the trained model and a
short-retrain variant on the shuffled labels) produces a null kNN@5 distribution with mean 7.0–7.7%
(chance for 15 lineages is 6.7%) and a maximum across all 100 permutations of 17–20% (Figure F4). The
real 15-lineage kNN@5 of 78.4% is never approached by chance in either variant: empirical p = 0.0099
for both, meeting the pre-registered p < 0.01 target. The correspondence the loss relies on is real,
not an artefact a shuffled model could reconstruct.

### 4.7 Head-to-head vs. Celligner: attempted, not obtained

A direct numeric comparison against Celligner, run on the identical HVG-filtered CCLE+TCGA matrices
and scored with the identical kNN@5/silhouette metric, was attempted (`pctrans-celligner-compare`)
but could not be completed in this project's environment. The published `celligner` PyPI release
(1.1.0) declares a dependency on a package literally named `umap` — not `umap-learn` — which has no
installable release on PyPI at all; `pip`/`uv` cannot resolve this dependency on **any** platform, a
stronger blocker than a missing compiler toolchain. Its documented from-source install additionally
requires R plus a bundled `mnnpy` build, and neither R nor `Rscript` is present in this environment.
This is reported as an honest gap rather than a fabricated or literature-borrowed number. Table T3
reports the reference numbers that *are* available on the identical metric — random (33.3%/6.7% for
3-/15-lineage), PCA+kNN (65.8%/25.2%), Harmony (84.2%, 3-lineage only), the supervised ceiling (97.1%,
3-lineage only), and the contrastive model (100.0%/78.4%) — so a reader with a working Celligner
build can drop its number directly into the same table without a cross-paper comparison. The
qualitative point survives regardless of the missing number: Celligner is unsupervised and must
discover lineage structure our model is handed as a training signal, so a comparable Celligner number
would reinforce, not threaten, §4.3's "lineage is nearly trivial" reading.

### 4.8 Functional case study: weak placement, null drug-response link

*Part A.* BRAF-mutant SKCM cell lines (n=47) sit closer to the centroid of BRAF-mutant SKCM patients
in embedding space than BRAF-wild-type lines (n=14) do (Mann-Whitney p = 0.047; effect size 0.649, 95%
CI [0.465, 0.834]). The effect is real but soft — the CI lower bound sits close to the no-effect line
of 0.5.

*Part B.* Among SKCM cell lines with both a BRAF call and a PRISM vemurafenib readout (n=41: 33
mutant, 8 wild-type), the correlation between proximity to the BRAF-mutant-patient centroid and
vemurafenib AUC is Spearman ρ = 0.209 (95% CI [−0.109, 0.493], p = 0.19) — not significant. Embedding
proximity to a driver-defined patient subgroup does not, in this dataset, predict the corresponding
drug's sensitivity. This null is reported plainly rather than reframed or omitted (Figure F6).

### 4.9 The Part-B null is a sample-size problem, not clear evidence of selective information loss

To distinguish "alignment discarded drug-response signal" from "the proximity metric was simply the
wrong probe, but signal survives elsewhere," a within-CCLE, 5-fold cross-validated probe
(`drug_signal_retained`) predicts vemurafenib AUC out-of-fold from three feature blocks: BRAF status
alone (R² = +0.226, ρ = +0.130, p = 0.42), raw 2,000-gene HVG expression (R² = −0.041, ρ = −0.167,
p = 0.30), and the 64-d contrastive embedding (R² = −0.333, ρ = −0.187, p = 0.24). None reach
significance at n=41, and — the key comparison — the embedding does not clearly underperform raw
expression: both return weak/negative out-of-fold R² here, so raw expression does not "rescue" the
signal the embedding supposedly threw away. A single categorical driver call (BRAF status alone) is
nominally the best of the three, which at n=41 with ~8 held-out samples per fold is the regime where
a 1-parameter model can out-generalise a 2,000- or 64-dimensional regressor even under ridge
shrinkage. The honest read is *inconclusive on information loss*: both continuous representations are
underpowered for a continuous phenotype at this n, which argues the binding constraint is sample size,
not that alignment selectively destroyed something raw expression uniquely retained. A descriptive,
unvalidated ElasticNet (CCLE raw expression → AUC) applied to all 65 TCGA-SKCM patients (no ground
truth exists to score it against) predicts AUC 0.780 ± 0.107, inside the CCLE training range
[0.560, 1.502] — offered only as a proximity-free reference point, not a validated prediction.

**Table T2 — Evidence table.**

| # | Claim | Experiment | Number (with CI) | Verdict |
|---|---|---|---|---|
| 4.1 | 100% kNN@5 is stable, not a lucky split | 10-seed re-split → train → eval | kNN@5 **0.950 ± 0.034**, CI **[0.932, 0.971]** (min 0.895) | Stable |
| 4.2 | Removing last leakage path doesn't change it | Train-only HVG | Gene-list Jaccard **0.951**; kNN@5 **100.0%** unchanged; silhouette Δ −0.0036 | Negligible leakage |
| 4.3 | Beats real batch correction; nearly ties supervised ceiling | Harmony/ceiling on identical test | Harmony **84.2%**; supervised ceiling **97.1%**; contrastive **100%** (+15.8 vs Harmony, +2.9 vs ceiling) | Beats batch corr.; near-trivial margin over ceiling |
| 4.4 | Generalises beyond 3 easy lineages | 15-lineage retrain+eval | kNN@5 **78.4%**, Wilson CI **[69.8, 85.0]**, n=111 | Genuine headroom |
| 4.4 | 15-lineage errors are biological | Confusion analysis | Named pairs absorb **45.8%** of off-diag mass from **3.8%** of cells (12×); LGG/READ→GBM/COAD 100% | Errors track biology |
| 4.5 | Not secretly a purity axis | Purity-stratified + residualised | domain-purity r **−0.455**; kNN@5 **100%** high & low strata; silhouette **+0.566→+0.500** | Survives adjustment |
| 4.6 | Not a label/batch artefact | Label-shuffle, 100 perms | real **78.4%** vs null **7.0–7.7%** (max 17–20%); **p = 0.0099** | Survives control |
| 4.7 | vs. incumbent method | Celligner head-to-head (identical kNN@5) | Celligner **n/a** (dep unresolvable, needs R+mnnpy); reference row populated for all other methods | Attempted, reported honestly |
| 4.8 | Captures BRAF-driver structure | Part A placement | Mann-Whitney **p = 0.047**; effect **0.649**, CI **[0.465, 0.834]** | Weak positive |
| 4.8 | Proximity predicts vemurafenib response | Part B response-link | Spearman **ρ = 0.209**, CI **[−0.109, 0.493]**, p = 0.19, n=41 | Null (honest) |
| 4.9 | Is the null selective information loss or underpowered task? | Drug-signal-retained probe (5-fold CV, n=41) | BRAF-alone R²=+0.226; raw expr R²=−0.041; embedding R²=−0.333 (all p>0.2) | Inconclusive — embedding doesn't clearly underperform raw expression |

**Table T3 — Reference-method comparison (identical kNN@5 metric).**

| Method | 3-lineage kNN@5 | 15-lineage kNN@5 |
|---|---|---|
| Random | 33.3% | 6.7% |
| PCA+kNN | 65.8% | 25.2% |
| Harmony+kNN | 84.2% | n/a (Day-17 sweep was 3-lineage only) |
| Celligner+kNN | n/a — dependency unresolvable in this environment | n/a |
| Supervised ceiling (CCLE→TCGA classifier, no alignment) | 97.1% | n/a |
| Contrastive (ours) | 100.0% (Wilson CI 90.8–100.0%, n=38) | 78.4% (Wilson CI 69.8–85.0%, n=111) |

## 5. Discussion

The headline of this paper is deliberately a sober one: cross-domain contrastive alignment of cell
lines and tumours robustly recovers cancer lineage, but that recovery is largely trivial — a plain
supervised classifier with no alignment step at all reaches 97.1% on the identical task — and on this
task the alignment does not reach a functional drug-response phenotype. Headline retrieval accuracies
in this class of literature can overstate biological insight, and supervised-alignment-evaluated-by-
lineage-retrieval is an especially easy way to look successful while demonstrating something close to
the training objective. The controls harness in this paper is built to separate "the model learned
biology" from "the task was easy," and applying it changes the interpretation of an otherwise clean
100% result substantially, from "a solved problem" to "a nearly-trivial problem that the model solves
robustly and for structurally sensible reasons."

What is genuinely encouraging: the 15-lineage confusions are biologically sensible rather than random
(§4.4), and the result survives both the permutation negative control (§4.6) and the purity confounder
check (§4.5) — evidence the embedding is not a pure artefact of label or batch structure, and encodes
real, if largely lineage-level, cancer identity, plus a weak signal beyond lineage (BRAF-driver
placement, §4.8 Part A).

Where it falls short: the near-trivial margin over the supervised ceiling (§4.3) and the null
drug-response link (§4.8 Part B) together argue that lineage-level alignment alone is not, on this
evidence, a translational predictor of drug response — it motivates deconfounding and
drug-supervised approaches (CODE-AE) rather than geometry-of-lineage approaches for the functional
endpoint that actually matters clinically.

**Relation to Celligner.** A direct numeric head-to-head was attempted but not obtained (§4.7):
Celligner's own PyPI packaging is unresolvable by standard tooling, and its from-source install
requires R, unavailable in this environment. We report this as an open gap rather than substitute a
literature-cited number computed under a different evaluation protocol. The qualitative point stands
regardless: Celligner is unsupervised and would need to *discover* lineage structure that our model
is handed as a training signal, so a comparable retrieval number from Celligner would reinforce
"lineage is easy," not threaten this paper's central claim.

**Relation to CODE-AE.** CODE-AE (He et al., 2022) factorises cell-line and tumour representations
into a shared domain-invariant code plus domain-specific confounder codes, trained end-to-end under
drug-response supervision so the shared code is explicitly optimised to be predictive of response.
Our encoder is the opposite on both axes that matter: it is supervised on lineage, not drug response,
and it performs no explicit deconfounding of domain-specific, response-irrelevant variation beyond
whatever the InfoNCE objective induces incidentally. A lineage-supervised, drug-agnostic embedding is
therefore not *expected* to transfer to a continuous drug-response phenotype — nothing in its
training objective rewards preserving that signal, and nothing penalises discarding it relative to
the lineage-separating axes the loss pushes it to amplify. Read this way, the Part-B null is close to
the a-priori expectation for this architecture, not a surprising failure. The Day-26 drug-signal-
retained probe (§4.9) adds one clarification: the embedding does not clearly underperform raw
expression at recovering AUC out-of-fold (both weak/negative R² at n=41), so the null cannot be
cleanly attributed to the embedding specifically destroying signal that raw expression uniquely
retained. The more defensible reading is that both representations are underpowered for a continuous
phenotype at this sample size, and that CODE-AE's drug supervision — not just its deconfounding
architecture — is doing real work that a lineage-only signal has no mechanism to substitute for.

## 6. Limitations

- **Task easiness / near-saturation.** kNN@5 was already ≈87% at epoch 1 of training; the
  fully-supervised ceiling reaches 97.1%. Coarse lineage is nearly linearly separable in bulk
  RNA-seq, and the contrastive machinery adds comparatively little for this specific endpoint.
- **Circularity.** The model is supervised on lineage and evaluated on lineage retrieval. The
  15-lineage stress-test and the BRAF case study partly mitigate this, but the primary metric remains
  close to the training objective.
- **Bespoke metric.** The Translational Fidelity Score (TFS) is defined for this project, not
  externally validated against a clinical outcome, and its thresholds are chosen rather than derived.
  It is reported for continuity across the project's reports, not as a validated instrument.
- **Small samples.** The 3-lineage test set is 38 cell lines; the vemurafenib response-link analysis
  is n=41. Confidence intervals are reported throughout, but statistical power is limited — the Part-B
  null is "no evidence of an effect," not "evidence of no effect."
- **The drug-signal-retained probe (§4.9) is itself underpowered.** Five-fold cross-validation on
  n=41 with a 2,000-dimensional raw-expression feature block leaves wide out-of-fold error; all three
  feature blocks return non-significant R²/ρ, so §4.9 cannot fully distinguish "the embedding retains
  as much signal as raw expression" from "neither representation has enough data to show a real effect
  at this n." The ElasticNet CCLE→patient reference has no ground truth to validate against and is
  reported descriptively only.
- **Bulk RNA-seq only.** No single-cell resolution; tumour stromal/immune composition partially
  confounds the TCGA side (analysed via purity in §4.5, not eliminated).
- **No external cohort or cross-platform validation (Rung 3, out of scope).** All tumour data is
  TCGA; generalisation to other cohorts, sequencing platforms, or patient-derived xenografts is
  untested.
- **No clinical validation (Rung 5, out of scope).** Nothing in this paper is prospective,
  pre-registered, or regulatory-grade; no diagnostic or treatment claim is made or supported anywhere
  in this work.
- **Single functional case study.** One drug (vemurafenib), one driver (BRAF), one lineage (SKCM);
  this is a proof-of-concept case study, not a systematic drug-response benchmark.
- **No numeric Celligner head-to-head (§4.7).** The published `celligner` package cannot be installed
  via `pip`/`uv` on any platform (broken PyPI metadata declaring a dependency on a nonexistent `umap`
  release rather than `umap-learn`), and its from-source path additionally requires R plus a bundled
  `mnnpy` build, neither available in this project's environment. Reference numbers on the identical
  metric are reported for every other method so a reader with a working Celligner install can complete
  Table T3 directly.

## 7. Conclusion

This paper is a reproducible, honest audit of a plausible-looking result: cross-domain contrastive
alignment of cancer cell lines and tumours is stable and survives a battery of negative controls, but
the signal it recovers is largely trivial lineage information that a plain supervised classifier
nearly matches without any alignment step, and it does not, on the evidence gathered here, transfer to
a real drug-response phenotype. The contribution is the evaluation discipline itself, and the released
controls harness is offered as a template for adversarially evaluating similar cross-domain alignment
claims before treating a clean headline metric as biological insight.

## Back matter

**Data availability.** All data used are public: CCLE expression (DepMap 24Q4, Figshare); TCGA
PanCancer expression (UCSC Xena); DepMap PRISM Repurposing 20Q2 (vemurafenib AUC); DepMap
OmicsSomaticMutations and TCGA MC3/cBioPortal (BRAF status); ABSOLUTE/ESTIMATE tumour purity
(GDC-hosted PanCanAtlas mastercalls). Exact URLs and versions are documented in the project's data
pipeline docs and per-day reports.

**Code availability.** GitHub: https://github.com/vthawfeek/pre-clinical-to-clinical-translation
(tagged release v0.1.0 for Phase 1; Phase 2 tracked in `PLAN-phase2.md` with one dated report per
day). All figures and tables in this manuscript are reproducible from the corresponding
`reports/*.json` machine-readable outputs and the `pctrans-*` CLI commands documented in each day's
report.

**Funding.** None. **Competing interests.** None. **Ethics.** All data are public and de-identified;
no IRB approval was required.

**Author contributions.** Sole author: conception, implementation, analysis, writing.

## Figures

- **F1.** Before/after: PCA of raw expression showing the CCLE/TCGA domain gap, alongside the aligned
  UMAP projection of the 3-lineage test set. `umap_before_after.png`.
- **F2.** Left: per-seed kNN@5 across 10 independent re-splits with mean and 95% CI marked against
  the Gate-2 stability threshold. Right: random / PCA+kNN / Harmony / supervised-ceiling /
  contrastive bar comparison on the identical 3-lineage test set — the pivotal panel for §4.3.
  `multiseed_baselines_panel.png`.
- **F3.** 15-lineage confusion-matrix heatmap with biologically-confusable pairs annotated.
  `confusion_matrix_15.png`.
- **F4.** Label-shuffle permutation null distribution (both variants) with the real value marked.
  `permutation_null.png`.
- **F5.** Purity-stratified retrieval and residualised-silhouette panel.
  `confounder_purity.png`.
- **F6.** BRAF/vemurafenib case study: placement scatter (Part A), proximity-vs-AUC scatter with the
  null fit (Part B), and the Day-26 drug-signal-retained R²/Spearman-ρ bar panel (BRAF-status-alone
  vs. raw expression vs. embedding). `braf_vemurafenib.png` (interactive version:
  `braf_vemurafenib.html`).
- **F7.** Reference-method bar chart (random/PCA/Harmony/ceiling/contrastive) on the identical kNN@5
  metric for both lineage variants, with the Celligner bar absent and annotated as
  dependency-unavailable rather than plotted as zero. `celligner_comparison.png`.

## Tables

- **T1.** Data and sample counts by lineage variant (3-lineage: 259 CCLE / ~2,000 TCGA across
  train/val/test; 15-lineage: 734 CCLE / 7,136 TCGA). See `reports/day-18-fifteen-lineages-setup.md`
  for the full per-lineage breakdown including lineages considered and dropped (e.g. prostate PRAD,
  excluded for insufficient CCLE line count).
- **T2.** The evidence table (§4, above).
- **T3.** The reference-method comparison table (§4.7, above); machine-readable version in
  `reports/celligner_comparison.json`.

## Pre-submission checklist

- [x] Day 25 Celligner head-to-head *attempted*; Table T3/Figure F7 populated with reference numbers
      and an honest n/a for Celligner itself (dependency unresolvable in this environment).
- [x] Day 26 CODE-AE positioning written (§5); drug-signal-retained probe (§4.9/F6) run on real data
      (n=41), reported honestly as inconclusive at this sample size.
- [x] Day 27: full manuscript draft assembled (`reports/preprint-draft.md`), all seven figures present
      and referenced, evidence table and reference-method table finalised with real numbers, abstract
      numbers cross-checked against `reports/phase2-summary.md` and the day-15/17/19/20/21/23/25/26
      reports.
- [ ] ORCID registered; "Independent Researcher" affiliation set on the submission record.
- [ ] Preprint posted to bioRxiv (biology framing) or arXiv q-bio (endorsement permitting).
- [ ] Final citation check (exact venue/year) for Celligner, PRECISE, CODE-AE, SupCon, CLIP/InfoNCE
      before submission — this draft cites them from the outline's prior research pass, not a fresh
      bibliographic search.
- [ ] Target-journal APC/waiver checked; venue cross-referenced against DOAJ/COPE before submission.
