# PLAN-phase2.md — Rigorous Validation of the CCLE–TCGA Contrastive Manifold

> **Purpose:** Phase 1 (Days 1–14) built a working dual-tower contrastive model and showed it aligns
> three cancer lineages across the cell-line/patient domain gap (test kNN@5 = 100%, TFS 0.89). That
> result is *internally valid but easy to attack*: the task is a 3-class problem the metric already
> half-solved at epoch 1, the evaluation is close to circular with the training objective, the test
> set is 38 cell lines, and "Translational Fidelity" has only ever been checked against lineage
> labels — never against real biology. Phase 2 **stress-tests** the system so the claims survive a
> knowledgeable reviewer, and takes the first concrete step toward a biological endpoint.
>
> This is the **portfolio package**: enough rigour to be credible and demonstrate scientific maturity,
> scoped to public data and the architecture already built. It is deliberately *not* a clinical study.

---

## How to Use This Plan

1. This file continues `PLAN.md`. Phase 1 runs Days 1–14; Phase 2 runs **Days 15–25**.
2. `/day N` executes all tasks for day N (N ≥ 15 reads this file), runs ruff + pytest, writes the
   daily report, commits, and pushes — same contract as Phase 1.
3. `/gate-check` gains a **Gate 2** decision (see *Phase 2 Gate*) alongside the existing Gate 0/1.
4. Milestone content days: `/blog-draft 25` drafts Blog Post 3.
5. Every day ends with the mandatory quality gate (ruff + pytest) before committing.

---

## What Phase 2 Is — and Is NOT

**In scope (this plan):**
- **Rung 1 — Statistical hardening (full):** confidence intervals, multi-seed reproducibility,
  train-only feature selection (removes the last leakage path), real batch-correction baselines run
  on identical data, and a supervised cross-domain classifier ceiling.
- **Rung 2 — Harder task (focused slice):** scale from 3 to ~15 lineages (incl. deliberately
  confusable pairs), one confounder analysis (tumour purity), and the label-shuffle permutation
  negative control.
- **Rung 4 — Functional proof-of-concept (single case study):** the vemurafenib / BRAF-mutant
  melanoma story — the first time the embedding is tied to a real drug-response phenotype rather
  than a lineage label.

**Explicitly OUT of scope (future work, named so reviewers know we know):**
- **Rung 3 — External cohorts / cross-platform / PDX / single-cell.** Real work; a separate phase.
- **Rung 5 — Prospective, pre-registered clinical validation, regulatory (SaMD / biomarker
  qualification).** A multi-year, funded, multi-institution effort — not an individual repo extension.
- **Any clinical claim.** Phase 2 outputs are a *research method with early biological support*,
  nothing more. Language in reports and blog must stay at that altitude.

---

## Quick Reference — New Modules & Data (Phase 2)

```
New source modules
  pctrans/evaluation/stats.py         bootstrap + Wilson CIs, multi-seed aggregation, permutation test
  pctrans/evaluation/baselines.py     Harmony / ComBat / Scanorama + supervised cross-domain classifier
  pctrans/evaluation/confounders.py   tumour-purity confounder analysis (ESTIMATE)
  pctrans/casestudy/braf_vemurafenib.py   Rung-4 placement + response-link analysis

New / changed configs
  configs/data.yaml                   lineages -> ~15; LINEAGE_TO_IDX becomes config-driven (dynamic)
  configs/data_15.yaml                15-lineage variant (keeps the 3-lineage run reproducible)

New public data (all unrestricted)
  DepMap PRISM Repurposing / GDSC     vemurafenib (PLX4032) sensitivity for CCLE lines
  DepMap OmicsSomaticMutations        BRAF V600 status for CCLE melanoma lines
  TCGA MC3 / SKCM BRAF status         BRAF V600 status for SKCM patients
  TCGA ESTIMATE / ABSOLUTE purity     per-patient tumour-purity scores (confounder analysis)

New reports
  reports/day-15..25-*.md             daily reports
  reports/phase2-summary.md           consolidated Gate-2 evidence table
  reports/blog-03-validation.md       "What survives rigorous validation"

Compute: unchanged — CPU / free Colab. Bulk-RNA MLP; ~1 min/train even at 15 lineages. Cost ~$0.
```

---

## Phase 2 Gate (Gate 2) — the honesty checkpoint

Run after Day 24. Gate 2 asks: *does the Phase-1 result survive being attacked?* All five are
reported; the decision is a narrative, not a single number.

```
══════════════════════════════════════════════════════════
                 GATE 2 — VALIDATION REPORT
══════════════════════════════════════════════════════════
G2-1  Multi-seed 3-lineage kNN@5: mean ± sd over 10 seeds, 95% CI
      PASS if CI lower bound ≥ 0.90 (stable, not a lucky split)
G2-2  Beats real baselines on identical data
      PASS if kNN@5 > best of {Harmony, ComBat, Scanorama} by ≥ 5 pts
      AND is not merely matching the supervised classifier ceiling
G2-3  Survives the label-shuffle permutation test
      PASS if real kNN@5 sits outside the shuffled null (empirical p < 0.01)
G2-4  Lineage signal survives purity adjustment
      PASS if kNN@5 holds within high- and low-purity patient strata
      AND embedding is not primarily a purity axis
G2-5  15-lineage task has genuine headroom + sensible errors
      REPORT kNN@5 (expected < 100%); confusions are biologically plausible
      (lung LUAD↔LUSC, glioma GBM↔LGG, colorectal COAD↔READ), not random
G2-BONUS  Vemurafenib case study is directionally positive
      REPORT whether proximity-to-BRAF-mutant-patient-space tracks vemurafenib
      sensitivity; a null result is reported honestly, not hidden
──────────────────────────────────────────────────────────
DECISION:  PORTFOLIO-READY  /  NEEDS-WORK  (with the specific failing rung named)
══════════════════════════════════════════════════════════
```

---

## Phase 2A — Rung 1: Statistical Hardening (Days 15–17)

---

### Day 15: Confidence Intervals & Multi-Seed Reproducibility

**Goal:** Every headline metric carries an interval, and the Phase-1 result is shown to be stable
across data splits — not an artefact of one lucky 38-cell-line test set.

**Tasks:**

1. Implement `pctrans/evaluation/stats.py`:
   - `wilson_ci(successes, n, alpha=0.05) -> tuple[float, float]` — analytic CI for the kNN
     accuracy proportion (correct for small n; the 38-anchor case).
   - `bootstrap_ci(values, statistic, n_boot=2000, seed=0) -> dict` — percentile CI; resample CCLE
     test anchors (for kNN) or pooled samples (for silhouette) with replacement.
   - `bootstrap_metric_ci(match_fraction, n_boot=2000, seed=0)` — convenience wrapper that turns the
     per-CCLE `match_fraction` (already produced by `knn.py`) into a kNN@5 point estimate + 95% CI.
2. Implement a multi-seed runner `pctrans/scripts/multiseed.py` (`pctrans-multiseed` entry point):
   - For `--seeds 42..51` (10 seeds): re-run **split → HVG(train-only, see Day 16 preview) →
     train → test-eval** end-to-end, collect test kNN@{1,5}, silhouette, TFS per seed.
   - Note: varying the split seed changes *which* cell lines land in test — this is the honest test
     of small-test-set stability. Reuse the default training config; no tuning per seed.
   - Write `reports/multiseed_results.json` (per-seed rows + aggregate mean/sd/CI).
3. Extend `pctrans-evaluate` to print Wilson + bootstrap CIs next to every point metric and add them
   to `reports/eval_summary.json`.

**Expected outputs:**
```
Test kNN@5:  100.0%  (Wilson 95% CI 90.8–100.0%, n=38)   ← honest interval, not a bare "100%"
Multi-seed (10 seeds): kNN@5 mean X.XXX ± X.XXX, CI [lo, hi]; silhouette mean ± sd; TFS mean ± sd
```

**Tests to add:**
```python
def test_wilson_ci_known_value():          # 38/38 -> lower bound ~0.908, upper 1.0
def test_bootstrap_ci_contains_point():     # point estimate inside its own 95% CI
def test_bootstrap_ci_shrinks_with_n():     # wider CI at n=38 than at n=380 (synthetic)
```

**Verification:** `uv run ruff check pctrans/ tests/` · `uv run pytest tests/test_stats.py -q`
**Daily report:** `reports/day-15-confidence-intervals.md`
**Commit:** `day 15: bootstrap + Wilson CIs, multi-seed reproducibility harness`
**Next up:** Day 16 — remove the last leakage path (train-only HVG).

---

### Day 16: Train-Only Feature Selection (Close the Last Leakage Path)

**Goal:** HVG selection stops seeing test samples. Quantify how much (if any) the Phase-1 numbers
depended on the mild unsupervised leakage of selecting genes on all data.

**Tasks:**

1. Refactor the pipeline order in `pctrans/data/preprocessor.py`:
   - Currently: HVG on all samples → split. Change to: **stratified split on sample IDs first →
     compute union-rank HVG variance on the TRAIN slice only → apply that gene list to val/test.**
   - Add `select_hvgs(..., train_ids)` param; keep the old all-sample path behind a
     `--hvg-on all|train` flag so the Phase-1 result stays reproducible for the before/after.
2. Re-run preprocessing + split + train + Gate-1 eval under `--hvg-on train`. Save the new gene list
   as `data/processed/gene_list_trainhvg.txt` (do not overwrite the Phase-1 artefact).
3. **Leakage-delta analysis:** report Δ in test kNN@5, silhouette, TFS, and gene-list overlap
   (Jaccard) between all-sample vs train-only HVG. Interpretation: small Δ → leakage was negligible
   (the honest, expected outcome); large Δ → Phase-1 numbers were partly leaked and the train-only
   numbers become canonical going forward.

**Expected outputs:**
```
HVG gene-list Jaccard (all vs train-only): 0.XX
Test kNN@5:  all-sample 100.0%  ->  train-only XX.X%   (Δ = ...)
Silhouette:  +0.57 -> +0.XX     TFS: 0.89 -> 0.XX
```

**Tests to add:**
```python
def test_hvg_train_only_ignores_test(splits, exprs):
    # gene list is identical whether test rows are present or shuffled — proves no test influence
def test_hvg_flag_reproduces_phase1():
    # --hvg-on all reproduces the Day-4 gene_list.txt exactly (backward-compat guard)
```

**Verification:** full suite green; `pctrans-preprocess --hvg-on train ...` runs clean.
**Daily report:** `reports/day-16-train-only-hvg.md` — include the leakage-delta table.
**Commit:** `day 16: train-only HVG selection, leakage-delta analysis`
**Next up:** Day 17 — real baselines, run for real, on identical data.

---

### Day 17: Real Baselines + Supervised Ceiling

**Goal:** Replace "~63% (literature)" with numbers actually computed on our test data, and establish
the supervised ceiling — how much lineage signal is trivially recoverable without any alignment.

**Tasks:**

1. Implement `pctrans/evaluation/baselines.py`:
   - `harmony_knn(ccle_test, tcga_test, k=5)` — `harmonypy` batch-integrate pooled features (domain =
     batch), then cross-domain kNN@5 by lineage.
   - `combat_knn(...)` — ComBat (`inmoose.pycombat` or equivalent) batch-correct on domain, then kNN.
   - `scanorama_knn(...)` — Scanorama integrate, then kNN.
   - `supervised_ceiling(ccle_train, tcga_test)` — train a plain logistic-regression / small MLP
     lineage classifier on **CCLE** expression, evaluate cross-domain on **TCGA** test. This is the
     "how easy is the problem, really" ceiling; if the contrastive model only matches this, the
     alignment isn't the thing doing the work.
2. Add these deps as an optional extra `[project.optional-dependencies] baselines` so the core install
   stays light (`harmonypy`, `scanorama`, `inmoose`). Gate them behind an import guard + `pytest.mark.skipif`.
3. Extend `pctrans-evaluate` (or a `pctrans-baselines` subcommand) to run all baselines and write
   `reports/baselines.json`. Update the Gate-1 report's baseline block with real numbers.

**Expected outputs (illustrative — record actual):**
```
Random            33.3%
PCA+kNN (Day 10)  65.8%
ComBat+kNN        ~XX%
Harmony+kNN       ~XX%
Scanorama+kNN     ~XX%
Supervised ceiling (CCLE->TCGA classifier)  ~XX%   ← interpret vs. contrastive
Contrastive (ours) 100.0% (CI 90.8–100)
```

**Tests to add:**
```python
def test_supervised_ceiling_beats_random():         # sanity: > 0.33 on real signal
def test_baseline_knn_shapes_and_range():           # each baseline returns a [0,1] accuracy
```

**Verification:** `uv run pytest tests/test_baselines.py -q` (baseline libs skipped if absent).
**Daily report:** `reports/day-17-real-baselines.md` — full baseline table + interpretation.
**Commit:** `day 17: Harmony/ComBat/Scanorama baselines, supervised cross-domain ceiling`
**Next up:** Day 18 — make the task hard: scale to ~15 lineages.

---

## Phase 2B — Rung 2 (slice): Harder Task, Confounder, Negative Control (Days 18–21)

---

### Day 18: Scale to ~15 Lineages — Data & Model

**Goal:** A harder classification problem with real headroom, including confusable lineage pairs, so
the retrieval metric can actually *fail* and thereby *mean* something.

**Tasks:**

1. Make lineages config-driven (they are currently hard-coded):
   - `pctrans/data/dataset.py`: build `LINEAGE_TO_IDX` from the config lineage list at load time
     instead of the fixed `{"LUAD":0,"BRCA":1,"SKCM":2}`.
   - Add `configs/data_15.yaml` with the 15-lineage set; keep `configs/data.yaml` (3-lineage) intact.
2. Choose ~15 lineages present with adequate N in **both** CCLE and TCGA, deliberately including
   confusable pairs (this is the point):
   ```
   LUAD, LUSC (lung pair) · BRCA · SKCM · COAD, READ (colorectal pair) · PAAD · STAD ·
   LIHC · KIRC · HNSC (squamous, confusable w/ LUSC) · GBM, LGG (glioma pair) · OV · BLCA
   ```
   - Add a CCLE `OncotreePrimaryDisease` → label alias table for the new lineages.
   - **Guard:** verify each retained lineage has ≥ ~15 CCLE lines and ≥ ~40 TCGA patients; drop or
     swap any that are too small (e.g. prostate PRAD has too few CCLE lines — excluded by design).
     Record the final kept set and per-lineage counts.
3. Re-run preprocessing (`--hvg-on train`, full matrices already downloaded) and training with the
   15-lineage config. Update the sampler to stratify across the larger lineage set (per-lineage batch
   slot may shrink; keep ≥ 4 CCLE + 4 TCGA per lineage per batch, raise batch size if needed).
4. Save the 15-lineage artefacts under a distinct prefix (`*_15.parquet`, `best_model_15.pt`) so the
   3-lineage deliverable stays intact and reproducible.

**Expected outputs:** processed 15-lineage parquets + gene list + splits; `models/best_model_15.pt`;
per-lineage train/val/test count table (dropped lineages noted).

**Tests to add:**
```python
def test_lineage_map_is_config_driven():       # 15-lineage config -> 15 label ids, contiguous
def test_sampler_covers_all_lineages(cfg15):   # every batch draws each lineage from each domain
```

**Verification:** training completes, val kNN logged; suite green.
**Daily report:** `reports/day-18-fifteen-lineages-setup.md` — kept/dropped lineages + counts.
**Commit:** `day 18: config-driven lineages, 15-lineage data + training run`
**Next up:** Day 19 — evaluate the 15-lineage model; are the mistakes biological?

---

### Day 19: 15-Lineage Evaluation — Headroom & Error Structure

**Goal:** Report the harder metric honestly and show the model's *mistakes are biologically sensible*
— the strongest evidence that it learned real lineage biology, not a shortcut.

**Tasks:**

1. Run `pctrans-evaluate` on the 15-lineage test set: overall + per-lineage kNN@{1,3,5,10} with
   Wilson/bootstrap CIs, the 15×15 confusion matrix, silhouette, TFS.
2. **Error-structure analysis:** identify the top off-diagonal confusions and check them against
   known biology. Expected sensible confusions: LUAD↔LUSC (lung), GBM↔LGG (glioma), COAD↔READ
   (colorectal), LUSC↔HNSC (squamous). Random/implausible confusions (e.g. GBM↔BRCA) would be a red
   flag — report either way.
3. Render a 15-lineage UMAP (reuse `viz.py`) and update `notebooks/03_evaluation.ipynb` with a
   15-lineage section (metrics table + confusion heatmap + UMAP).
4. Compare 3-lineage vs 15-lineage: state plainly that the metric dropped (expected) and that the
   remaining errors concentrate on genuinely related cancers — which is the point of the exercise.

**Expected outputs (illustrative):**
```
15-lineage overall kNN@5: XX.X% (CI ..)   [3-lineage was 100%]
Top confusions: LUAD->LUSC, LGG->GBM, READ->COAD  (all biologically adjacent)
Silhouette (15-way): +0.XX
```

**Tests to add:**
```python
def test_confusion_matrix_is_15x15(eval15):
def test_offdiagonal_mass_on_related_pairs(eval15):  # majority of errors within curated pairs
```

**Verification:** notebook executes end-to-end (`nbconvert --execute`, exit 0); suite green.
**Daily report:** `reports/day-19-fifteen-lineage-eval.md` — confusion heatmap + biology read.
**Commit:** `day 19: 15-lineage evaluation, error-structure biology analysis`
**Next up:** Day 20 — is the alignment lineage, or is it just tumour purity?

---

### Day 20: Confounder Analysis — Tumour Purity

**Goal:** Rule out the biggest alternative explanation: that "alignment" is really the model learning
*pure cell line vs. stroma-contaminated tumour*, not cancer identity.

**Tasks:**

1. Assemble purity labels in `pctrans/evaluation/confounders.py`:
   - TCGA per-patient tumour purity from ESTIMATE (or ABSOLUTE) — downloadable table keyed by
     TCGA barcode. Cell lines assigned purity ≈ 1.0 (pure culture).
   - `load_purity(...)` joins purity onto the test embeddings by sample ID.
2. Three analyses:
   - **(a) Domain-axis vs purity:** define the CCLE→TCGA centroid direction; correlate each sample's
     projection on it with purity. High correlation would mean the residual domain axis *is* a purity
     axis (informative, not fatal — it is the axis we want alignment to ignore).
   - **(b) Purity-stratified retrieval:** split TCGA test patients into high- vs low-purity halves;
     recompute cross-domain kNN@5 within each stratum. Lineage retrieval must hold in *both* — if it
     only works for high-purity patients, the model leans on purity.
   - **(c) Purity-regressed embeddings:** regress purity out of the pooled embeddings (residualise);
     recompute silhouette by lineage. Lineage cohesion should survive.
3. Write `reports/confounder_purity.json` + a scatter/box figure.

**Expected outputs (illustrative):**
```
corr(domain-axis projection, purity): r = 0.XX      (context, not pass/fail)
kNN@5 high-purity stratum: XX%   low-purity stratum: XX%   (both should hold)
Silhouette after purity residualisation: +0.XX       (should stay > 0)
```

**Tests to add:**
```python
def test_purity_stratified_knn_runs(both_strata):
def test_residualisation_preserves_shape():          # residual embeddings same shape, finite
```

**Verification:** `uv run pytest tests/test_confounders.py -q`; suite green.
**Daily report:** `reports/day-20-purity-confounder.md` — the three analyses + verdict.
**Commit:** `day 20: tumour-purity confounder analysis (stratified + residualised)`
**Next up:** Day 21 — the permutation control: does it collapse on shuffled labels?

---

### Day 21: Label-Shuffle Negative Control (Permutation Test)

**Goal:** Prove the model exploits *real* CCLE↔TCGA lineage correspondence, not batch structure or a
leak — by showing performance collapses to chance when that correspondence is destroyed.

**Tasks:**

1. Implement `permutation_test(...)` in `pctrans/evaluation/stats.py`:
   - For `--n-perm 20` permutations: randomly shuffle the mapping between CCLE lineage labels and
     TCGA lineage labels (break the correspondence the loss relies on), retrain a short schedule (few
     epochs is enough — a working model must fail here), evaluate test kNN@5. Collect the null
     distribution.
   - Also implement the cheaper **eval-only** variant: keep the trained model, shuffle test labels,
     recompute kNN@5 — this isolates metric-level chance.
2. Compute the empirical p-value: fraction of shuffled runs whose kNN@5 ≥ the real kNN@5. Target
   `p < 0.01`; the real value should sit far above the shuffled null (which should hover near
   1/n_lineages).
3. Plot the null distribution with the real value marked (`reports/permutation_null.png`).

**Expected outputs (illustrative):**
```
Shuffled-label kNN@5 null: mean ~0.07 (≈1/15), max 0.14 over 20 perms
Real kNN@5: XX%   -> empirical p < 0.01   (real result is not reachable by chance)
```

**Tests to add:**
```python
def test_permutation_null_near_chance():        # shuffled model ~ 1/n_lineages on synthetic
def test_permutation_pvalue_range():            # p in [0,1], real-signal case -> small p
```

**Verification:** `uv run pytest tests/test_stats.py -q -k permutation`; suite green.
**Daily report:** `reports/day-21-label-shuffle-control.md` — null plot + p-value.
**Commit:** `day 21: label-shuffle permutation negative control, empirical p-value`
**Next up:** Day 22 — the first tie to real biology: vemurafenib + BRAF melanoma.

---

## Phase 2C — Rung 4 (case study): Vemurafenib / BRAF-Mutant Melanoma (Days 22–23)

---

### Day 22: Drug & Mutation Data Assembly

**Goal:** Bring in a real drug-response phenotype and the driver mutation that governs it, so the
next day can ask whether the embedding *predicts response*, not just lineage.

**Tasks:**

1. Implement data loaders in `pctrans/casestudy/braf_vemurafenib.py`:
   - **Cell-line vemurafenib sensitivity:** DepMap PRISM Repurposing (or GDSC `PLX4032/Vemurafenib`)
     — per-CCLE-line potency (log-fold-change / IC50 / AUC). Keep the raw metric + a z-scored version.
   - **CCLE BRAF status:** DepMap `OmicsSomaticMutations` — flag BRAF V600E/V600K per melanoma line.
   - **TCGA SKCM BRAF status:** MC3 / TCGA SKCM MAF — flag BRAF V600 per patient.
   - `assemble_braf_table(...)` → tidy frame: sample_id, domain, lineage, BRAF_status, vemurafenib
     sensitivity (cell lines only), embedding vector.
2. Restrict to the SKCM slice of the 3-lineage model's test/all embeddings (melanoma is where the
   biology is clean and cell-line N is adequate). Document join coverage (how many SKCM lines have
   both BRAF status and a vemurafenib readout).
3. Save `data/processed/braf_vemurafenib.parquet` and a coverage summary.

**Expected outputs:**
```
SKCM cell lines with BRAF status + vemurafenib readout: N = ~..
BRAF-mutant vs WT split (cell lines / patients): ../..   ../..
```

**Tests to add:**
```python
def test_braf_status_parsing():                 # V600E flagged mutant, silent BRAF -> WT
def test_assemble_table_has_required_cols():
```

**Verification:** loaders run on real files; `uv run pytest tests/test_casestudy.py -q`.
**Daily report:** `reports/day-22-braf-data.md` — sources, join coverage, BRAF splits.
**Commit:** `day 22: vemurafenib + BRAF mutation data assembly (DepMap/GDSC/MC3)`
**Next up:** Day 23 — does embedding proximity predict vemurafenib response?

---

### Day 23: Placement + Response-Link Analysis

**Goal:** Test — honestly, small-N, retrospective — the two-part hypothesis that makes the embedding
biologically meaningful, using a textbook translational fact as a positive control.

**Tasks:**

1. **Part A — Placement (does the space recover a driver-defined subgroup?):**
   - Compute the centroid of BRAF-mutant SKCM *patients* in embedding space.
   - Show BRAF-mutant SKCM *cell lines* embed nearer that centroid than BRAF-WT lines do
     (Mann–Whitney on distance-to-centroid; report effect size + CI). This asks whether the model
     captured more than coarse lineage — a within-melanoma, driver-defined structure.
2. **Part B — Response link (the actual translational question):**
   - For SKCM cell lines, correlate **proximity to the BRAF-mutant-patient centroid** with
     **vemurafenib sensitivity** (Spearman ρ + bootstrap CI). Hypothesis: lines the model places
     closer to BRAF-mutant patient space are the vemurafenib-sensitive BRAF-mutant lines.
   - Positive control framing: vemurafenib's BRAF-V600 dependence is established, so a *positive*
     result validates the embedding; a *null* result is a legitimate, reportable finding about the
     model's current resolution — write it as such, do not bury it.
3. Figures: embedding scatter coloured by BRAF status (cell lines ✕ / patients ●); scatter of
   proximity vs vemurafenib sensitivity with fit + CI. Save to `reports/braf_vemurafenib.{png,html}`.
4. Add a Section 6 to `notebooks/03_evaluation.ipynb` and write the honest caveats: N is small, it is
   retrospective, response is multi-factorial, and this is one case study — a proof of concept, not
   proof of clinical utility.

**Expected outputs (illustrative — record actual, incl. nulls):**
```
Part A: BRAF-mut lines closer to BRAF-mut patient centroid than WT (p = .., effect ..)
Part B: Spearman ρ(proximity, vemurafenib sensitivity) = 0.XX (95% CI ..); n = ..
```

**Tests to add:**
```python
def test_centroid_distance_math():              # known synthetic points -> expected ordering
def test_response_correlation_runs_and_bounds():# rho in [-1,1], CI returned
```

**Verification:** notebook executes end-to-end; figures render; suite green.
**Daily report:** `reports/day-23-vemurafenib-casestudy.md` — both parts, figures, caveats.
**Commit:** `day 23: BRAF/vemurafenib placement + response-link case study`
**Next up:** Day 24 — assemble the Gate-2 evidence and decide.

---

## Phase 2D — Consolidation (Days 24–25)

---

### Day 24: Gate 2 Evaluation & Phase-2 Summary

**Goal:** Assemble all Phase-2 evidence into one place, run the Gate-2 decision, and update the
project's headline claims to match what actually survived.

**Tasks:**

1. Extend `/gate-check` with the **Gate 2** report (the five criteria + bonus above), reading from
   `reports/multiseed_results.json`, `baselines.json`, `permutation_null` outputs,
   `confounder_purity.json`, the 15-lineage eval, and the case-study result.
2. Write `reports/phase2-summary.md` — a single evidence table: for each rung, the claim, the
   experiment, the number (with CI), and the verdict. This is the artefact a reviewer reads first.
3. Update `README.md` and `CLAUDE.md`:
   - Replace bare "100%" with "100% (95% CI 90.8–100, n=38); stable across 10 seeds; beats
     Harmony/ComBat/Scanorama; survives label-shuffle (p<.01) and purity adjustment".
   - Add the 15-lineage result and the vemurafenib case-study outcome.
   - Add the explicit *limitations & scope* section (Rungs 3/5 out of scope; research method, not a
     clinical tool).
4. Add a `PHASE2` status block to `CLAUDE.md` mirroring the Phase-1 day tracker.

**Verification:** `/gate-check` prints Gate 0, 1, and 2; suite green; all reports present.
**Daily report:** `reports/day-24-gate2-summary.md`.
**Commit:** `day 24: Gate 2 evaluation, phase2-summary, README/CLAUDE claims updated`
**Next up:** Day 25 — tell the story: Blog Post 3.

---

### Day 25: Blog Post 3 — "What Survives Rigorous Validation"

**Goal:** The content deliverable — the credibility story that turns a clean demo into a mature piece.

**Tasks:**

1. `/blog-draft 25` → `reports/blog-03-validation.md`:
   - Title: "I Got 100% Accuracy. Then I Tried to Break It." (working title)
   - Hook: the honest confession — the Phase-1 metric was already 87% at epoch 1 and the test set was
     38 cell lines, so a good result *demanded* stress-testing before it could be believed.
   - Body: the five stress tests and what each showed (CIs & multi-seed; real baselines & supervised
     ceiling; 15 lineages with biologically sensible confusions; purity confounder ruled out;
     label-shuffle collapse); then the vemurafenib case study as the first tie to real drug response.
   - Honesty as the theme: report any null/negative outcomes plainly; name Rungs 3 & 5 as the road
     not yet travelled; keep the claim at "validated research method," never "clinical".
   - 1,200–1,500 words.
2. Draft LinkedIn Post 3 (the "how I tried to falsify my own model" angle) and an X thread outline,
   each anchored on the before/after credibility arc and one figure (permutation null or the
   biologically sensible 15-lineage confusion heatmap).
3. Final full-suite run + push.

**Verification:** `uv run ruff check` + `uv run pytest tests/ -q` green; content drafts present.
**Daily report:** `reports/day-25-blog3.md`.
**Commit:** `day 25: blog-03 validation story, LinkedIn/X drafts, Phase 2 complete`
**Next up:** (Optional future) Phase 3 — Rung 3 external cohorts / cross-platform / PDX; or a scoped
Rung-4 expansion beyond the single vemurafenib case study.

---

## Effort / Cost Snapshot (recap)

| Item | Phase 2 (Days 15–25) |
|---|---|
| Calendar | ~11 working days, solo, part-time (same `/day N` cadence as Phase 1) |
| Compute | CPU / free Colab; bulk-RNA MLP, ~1 min/train even at 15 lineages |
| Cost | ~$0 — all data public/unrestricted (CCLE, TCGA, DepMap PRISM/GDSC, MC3, ESTIMATE) |
| New deps | `harmonypy`, `scanorama`, `inmoose` (optional extra), `statsmodels` (Wilson/bootstrap) |
| Risk | Rung-4 case study may return a null — that is a legitimate, reportable outcome |
| Out of scope | Rung 3 (external/cross-platform/PDX/single-cell), Rung 5 (prospective/clinical/regulatory) |
```
