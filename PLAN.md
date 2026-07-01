# PLAN.md — Pre-Clinical to Clinical Translation: CCLE–TCGA Contrastive Manifold

> **Elevator pitch:** 85% of oncology drugs that cure cancer in cell lines fail in human clinical trials.
> The root cause is not chemistry — it is *translation fidelity*: a cell line grown in a petri dish is not
> the same biological object as a human tumour living inside an immune system and a blood supply.
> This project trains a dual-tower contrastive neural network to learn **what IS the same**:
> the shared transcriptional program that a lung adenocarcinoma cell line and a lung adenocarcinoma
> patient both express, stripped of the in-vitro culture artefacts. The result is a quantitative
> *Translational Fidelity Score* (TFS) for any CCLE cell line and an interactive map of exactly
> where that cell line lands in human patient space.

---

## How to Use This Plan

1. This file lives in the repo root as `PLAN.md` and is the source of truth for daily work.
2. `/day N` slash command executes all tasks for day N, runs lint + tests, writes the daily report, commits, and pushes.
3. `/blog-draft N` drafts blog post N with the paired LinkedIn post.
4. `/gate-check` runs the full Gate evaluation protocol and prints the decision.
5. Gate thresholds are numeric and non-negotiable — see **Gate Decision Architecture**.
6. Each day ends with a mandatory quality gate (ruff + pytest) before committing.

---

## Context: The Problem This Project Solves

### The Translation Crisis in Oncology

The average cost to bring a single cancer drug to market is $2.7 billion. The #1 reason drugs fail
in Phase II and III trials — after succeeding in preclinical models — is that **cell lines do not
represent the disease state of real patients**. In-vitro culture imposes artefacts:

| Artefact | Biological Origin | Effect on Gene Expression |
|---|---|---|
| Absence of tumour microenvironment | No immune, stromal, or endothelial cells | ~800 TME-related genes silenced |
| Serum-driven proliferation | Media composition forces mitotic programs | High MKI67, PCNA, CDK expression |
| Plastic-surface adhesion | Integrin signalling from 2D substrate | Aberrant ECM gene expression |
| Passage-induced drift | Cumulative mutation across culture | CNV-driven expression changes |

Contrastive representation learning offers a principled solution: treat CCLE cell lines and TCGA
human tumours as **two views of the same underlying cancer lineage identity**, and learn a shared
latent space that maximises mutual information between the two domains while discarding
domain-specific artefacts.

### Why CCLE + TCGA (Not Something Else)

- **Both are public, unrestricted.** No IRB approval or data access application required.
- **Both are bulk RNA-seq.** Same measurement modality eliminates multi-modal harmonisation.
- **Same gene space.** CCLE DepMap and TCGA both use GENCODE-annotated human transcripts; mapping is HUGO symbol matching, not cross-species orthology.
- **Known ground truth.** Cancer lineage labels (LUAD, BRCA, SKCM) serve as validation — if the model works, lineage-matched cell lines and patients cluster together in latent space.
- **Clinically relevant lineages.** LUAD (lung), BRCA (breast), and SKCM (melanoma) together represent ~40% of new cancer diagnoses annually in the US.

### The "Magic Math Loop"

```
CCLE RNA-seq vector (2,000 HVGs)       TCGA RNA-seq vector (2,000 HVGs)
            │                                         │
     CCLEEncoder (MLP)                        TCGAEncoder (MLP)
            │                                         │
      L2-normalised z_c ─────── cos-sim ──────── L2-normalised z_t
                                    │
                    SupCon-InfoNCE Loss (learnable τ)
                    "Pull same-lineage cross-domain pairs together.
                     Push different-lineage pairs apart."
```

The information-theoretic intuition: InfoNCE is a lower bound on mutual information between the
two domains. Maximising it forces the encoder to extract only the shared signal — cancer lineage
identity — and discard anything that differs between petri dish and patient (culture artefacts,
TME, stromal contamination).

---

## Quick Reference (CLAUDE.md to be extracted on Day 1)

```
Project:   Pre-Clinical to Clinical Translation (pctrans)
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
Repo:      https://github.com/vthawfeek/pre-clinical-to-clinical-translation
App:       https://pctrans.streamlit.app (Day 14)
Days done: [track here]
```

---

## Project Structure

```
pre-clinical-to-clinical-translation/
├── pctrans/                         # Main Python package
│   ├── data/
│   │   ├── ccle_client.py           # DepMap portal downloader
│   │   ├── tcga_client.py           # UCSC Xena downloader
│   │   ├── preprocessor.py          # Normalise, HVG selection, split
│   │   ├── dataset.py               # CCLEDataset, TCGADataset (PyTorch)
│   │   └── sampler.py               # StratifiedContrastiveBatchSampler
│   ├── models/
│   │   ├── encoders.py              # CCLEEncoder, TCGAEncoder (MLP + BN + ReLU + Dropout)
│   │   ├── dual_tower.py            # DualTowerModel (wraps both, L2 normalise)
│   │   └── losses.py                # SupConInfoNCELoss (learnable temperature)
│   ├── training/
│   │   ├── trainer.py               # ContrastiveTrainer (MLflow, early stop, checkpoint)
│   │   └── callbacks.py             # KNNValidationCallback (runs kNN@5 every epoch)
│   ├── evaluation/
│   │   ├── knn.py                   # knn_retrieval_accuracy (per lineage + overall)
│   │   ├── silhouette.py            # cross_domain_silhouette_score
│   │   ├── tfs.py                   # translational_fidelity_score (composite metric)
│   │   └── viz.py                   # umap_plot, training_curves, confusion_matrix
│   ├── inference/
│   │   └── api.py                   # TranslationEmbedder (embed_cell_line, query_patients)
│   └── scripts/
│       ├── download.py              # pctrans-download CLI
│       ├── preprocess.py            # pctrans-preprocess CLI
│       ├── train.py                 # pctrans-train CLI
│       ├── evaluate.py              # pctrans-evaluate CLI
│       └── query.py                 # pctrans-query CLI (single cell line → TFS + neighbours)
├── app/
│   └── streamlit_app.py             # Streamlit demo
├── notebooks/
│   ├── 01_eda.ipynb                 # Gene distributions, PCA before training, lineage counts
│   ├── 02_training_analysis.ipynb   # Loss curves, temperature evolution, gradient norms
│   ├── 03_evaluation.ipynb          # kNN table, UMAP, TFS per cell line
│   └── 04_showcase.ipynb            # Case studies, outlier cell lines, biology interpretation
├── data/
│   ├── raw/
│   │   ├── ccle/                    # Downloaded CCLE CSVs
│   │   └── tcga/                    # Downloaded TCGA TSVs
│   └── processed/
│       ├── ccle_2k.parquet          # [~300, 2000] filtered + normalised
│       ├── tcga_2k.parquet          # [~1960, 2000] filtered + normalised
│       ├── gene_list.txt            # 2,000 HVG HUGO symbols in order
│       ├── scalers.pkl              # Per-gene z-score params (fit on train only)
│       └── splits.json              # Train/val/test sample IDs by domain
├── models/
│   └── best_model.pt                # Best checkpoint (saved by trainer on val kNN improvement)
├── reports/
│   ├── day-01-scaffold.md
│   ├── ...
│   ├── blog-01-concept.md           # Draft: "The Cell Line Translation Problem"
│   └── blog-02-results.md           # Draft: "Teaching AI to Translate"
├── docs/
│   ├── 01_data_pipeline.md
│   ├── 02_feature_engineering.md
│   ├── 03_architecture.md
│   ├── 04_training.md
│   └── 05_evaluation.md
├── configs/
│   ├── model.yaml                   # Encoder layer dims, dropout, embed_dim
│   ├── training.yaml                # Epochs, batch_size, lr, warmup_steps, patience
│   └── data.yaml                    # HVG count, lineages, min_samples_per_lineage
├── tests/
│   ├── conftest.py                  # tiny_ccle, tiny_tcga, tiny_model fixtures
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   ├── test_inference.py
│   └── test_scripts.py
├── .claude/
│   └── commands/                    # /day, /blog-draft, /gate-check slash commands
├── PLAN.md                          # This file
├── CLAUDE.md                        # Quick reference (extracted from above)
├── README.md                        # Results table + architecture diagram + quick start
├── pyproject.toml                   # UV/Hatchling build, dependencies, CLI entry points
└── .github/
    └── workflows/ci.yml             # ruff + pytest on every push
```

---

## Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| Package manager | `uv` | Same as mtdna-foundation-model; fast, lockfile-based |
| ML framework | PyTorch 2.2 | Industry standard; InfoNCE loss in pure Python |
| Data processing | pandas, numpy, pyarrow | Parquet IO for processed data |
| HVG selection | scanpy-style variance calculation (pure numpy) | Avoid scanpy install overhead |
| Dimensionality reduction | umap-learn 0.5 | UMAP for latent space visualisation |
| Experiment tracking | MLflow 2.12 | Loss curves, checkpoint tagging |
| Visualisation | plotly 5, matplotlib | Interactive UMAP in Streamlit |
| App | Streamlit 1.34 | Fast demo deployment |
| Testing | pytest 8, pytest-cov | ≥80% coverage target |
| Linting | ruff 0.4 | Fast, zero-config |
| CLI | typer 0.12 | Same as mtdna-foundation-model |
| Config | PyYAML | YAML configs for model/training/data |
| Compute | CPU (local) or Colab T4 | Zero cost; ~12 min CPU / ~3 min GPU |

---

## Dataset Specifications

### CCLE (Cancer Cell Line Encyclopedia)

- **Source:** DepMap Portal 24Q4 release — https://depmap.org/portal/
- **Expression file:** `OmicsExpressionProteinCodingGenesTPMLogp1.csv`
  - Shape: ~1,900 cell lines × ~19,177 genes
  - Units: log₂(TPM + 1), protein-coding genes only
  - Gene IDs: "HUGO_SYMBOL (ENTREZ_ID)" format, e.g. `EGFR (1956)`
- **Metadata file:** `Model.csv`
  - Key columns: `ModelID`, `OncotreePrimaryDisease`, `OncotreeLineage`, `CancerType`
- **Lineage filter (OncotreePrimaryDisease):**
  - LUAD: `"Lung Adenocarcinoma"` → ~100 cell lines
  - BRCA: `"Breast Cancer"` → ~130 cell lines (all subtypes included)
  - SKCM: `"Melanoma"` → ~70 cell lines (cutaneous primary preferred)
- **Download method:** Python `requests` library, direct Figshare URL from DepMap API

### TCGA (The Cancer Genome Atlas)

- **Source:** UCSC Xena Pan-Cancer Hub — https://xenabrowser.net/datapages/
- **Expression file:** `EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena`
  - Shape: ~10,535 samples × 20,530 genes
  - Units: log₂(normalized_count + 1)
  - Gene IDs: HUGO symbols (no Entrez ID suffix)
- **Phenotype file:** `Survival_SupplementalTable_S1_20171025_xena_sp.tsv`
  - Key column: `cancer type abbreviation` → LUAD, BRCA, SKCM
- **Lineage filter:**
  - LUAD: `cancer type abbreviation == "LUAD"` → ~500 patients
  - BRCA: `cancer type abbreviation == "BRCA"` → ~1,000 patients
  - SKCM: `cancer type abbreviation == "SKCM"` → ~460 patients
- **Download method:** UCSC Xena HTTP file server — direct wget-style download

### Gene ID Harmonisation Protocol (Day 4)

1. Parse CCLE columns: split on ` (` to extract HUGO symbol → `EGFR (1956)` → `EGFR`
2. TCGA columns already in HUGO symbol format
3. Find common genes: `common_genes = set(ccle_genes) ∩ set(tcga_genes)`
4. Expected common genes: ~18,000 (minor differences from transcript version)
5. Proceed to HVG selection on the common gene set only

### HVG Selection — Union-Rank Method (Day 4)

Standard top-variance selection is biased: highly expressed genes (TP53, ACTB) have higher
absolute variance purely due to scale. The **union-rank method** corrects for this:

```
1. Compute per-gene variance WITHIN CCLE: var_ccle[g] for each common gene g
2. Compute per-gene variance WITHIN TCGA: var_tcga[g] for each common gene g
3. Rank genes by var_ccle → rank_ccle[g]
4. Rank genes by var_tcga → rank_tcga[g]
5. mean_rank[g] = (rank_ccle[g] + rank_tcga[g]) / 2
6. Select top 2,000 genes by mean_rank (highest mean rank = highest average variability)
```

This gives equal weight to variability in each domain, preventing TCGA (10× more samples)
from dominating the HVG list.

---

## Architecture Specifications

### Encoder Architecture

Both CCLEEncoder and TCGAEncoder use the same MLP template **by design** — even though features
are already harmonised. This signals to reviewers that the architecture is built to handle
true asymmetry (different feature spaces, different modalities) in production.

```
Input: x ∈ ℝ²⁰⁰⁰  (2,000 HVGs, per-gene z-scored)
│
├─ Linear(2000, 1024) → BatchNorm1d(1024) → ReLU → Dropout(0.3)
├─ Linear(1024, 512)  → BatchNorm1d(512)  → ReLU → Dropout(0.3)
├─ Linear(512, 256)   → BatchNorm1d(256)  → ReLU → Dropout(0.3)
├─ Linear(256, 128)   → BatchNorm1d(128)  → ReLU → Dropout(0.2)
└─ Linear(128, 64)    # No BN/activation here — projection head
│
Output: z ∈ ℝ⁶⁴  (raw embedding, before L2 normalisation)
```

**DualTowerModel** wraps both encoders and applies L2 normalisation:
```python
z_c = F.normalize(self.ccle_encoder(x_ccle), dim=-1)   # unit hypersphere
z_t = F.normalize(self.tcga_encoder(x_tcga), dim=-1)    # unit hypersphere
```

**Parameter count:** ~4.3M per encoder × 2 = ~8.6M total parameters.

**Why 64 dimensions?** The latent space must separate 3 lineage clusters with clean margins.
64 dimensions is over-parameterised for 3 classes (UMAP will show this clearly) but gives
the model flexibility during training. Ablation: also test 32 and 128 (Day 9).

**Why BatchNorm before each activation?** BatchNorm stabilises the scale of inter-layer
activations, which is critical when CCLE (small N ~300) and TCGA (large N ~1960) have
different gene expression distributions entering each mini-batch.

### Learnable Temperature

The temperature τ controls sharpness of the similarity distribution. Rather than hand-tune,
we learn it:

```python
self.log_tau = nn.Parameter(torch.tensor(math.log(1 / 0.07)))  # init τ ≈ 0.07
tau = self.log_tau.exp()  # always positive
```

During training, τ typically converges to 0.02–0.05 as the model sharpens its boundary.
If τ collapses to ~0 or diverges to >1, this signals training instability — see Risk table.

---

## Loss Function: SupCon-Style Multi-Positive InfoNCE

### Mathematical Derivation

Standard CLIP InfoNCE assumes exactly one positive per anchor. Our setting has **multiple valid
positives per anchor**: all TCGA samples of the same lineage as a given CCLE cell line.

Let batch B contain B_c CCLE samples and B_t TCGA samples (B_c = B_t = B/2 = 24).
Let `Pos(i)` = set of TCGA indices j in the batch where lineage(j) == lineage(i).

**SupCon-InfoNCE loss for CCLE anchor i:**

```
L_c(i) = − (1/|Pos(i)|) × Σ_{p ∈ Pos(i)} log [
    exp(z_c(i) · z_t(p) / τ)
    ─────────────────────────────────────────────
    Σ_{j=1}^{B_t} exp(z_c(i) · z_t(j) / τ)
]
```

**Symmetric total loss:**
```
L = (1/B_c) Σ_i L_c(i) + (1/B_t) Σ_j L_t(j)
```

Where L_t(j) is the symmetric loss with TCGA as anchors and CCLE as keys.

### Batch Construction (StratifiedContrastiveBatchSampler)

```
Batch size B = 48 = 16 LUAD + 16 BRCA + 16 SKCM
Each "slot" of 16 = 8 CCLE samples + 8 TCGA samples from that lineage

For CCLE: oversample (repeat) since N_CCLE < N_TCGA
For TCGA: undersample (random sample without replacement per epoch)

Within each B:
  Pos(CCLE_LUAD_i) = {all 8 TCGA_LUAD_j} — 8 positives
  Pos(CCLE_BRCA_i) = {all 8 TCGA_BRCA_j} — 8 positives
  Pos(CCLE_SKCM_i) = {all 8 TCGA_SKCM_j} — 8 positives
  
  Neg(CCLE_LUAD_i) = {all 8 TCGA_BRCA + all 8 TCGA_SKCM} — 16 true negatives
```

This construction eliminates **false negatives** (within-lineage cross-domain pairs treated
as negatives). Every negative in the batch is a true between-lineage negative.

### Why Not Standard InfoNCE?

Standard CLIP pairs (cell_line_i, patient_i) randomly. If batch has 16 LUAD cell lines and
16 LUAD patients, the 15 TCGA LUAD samples that are NOT paired with cell_line_i are treated
as negatives — this punishes correct cross-domain alignment. SupCon fixes this with |Pos(i)| > 1.

---

## Week 1 (Days 1–7): Data Infrastructure & Architecture

---

### Day 1: Project Scaffold + Environment

**Goal:** Working `pctrans` package that imports cleanly, CI passing, GitHub remote set up.

**Tasks:**

1. Create directory tree (`data/raw/ccle`, `data/raw/tcga`, `data/processed`, `models`, `reports`, `docs`, `configs`, `tests`, `notebooks`, `app`, `.claude/commands`)
2. Write `pyproject.toml`:
   - `[build-system]` uv/hatchling
   - `[project.dependencies]`: torch≥2.2.0, pandas, numpy, pyarrow, scikit-learn, umap-learn, streamlit, plotly, mlflow, typer, pyyaml, requests, tqdm
   - `[project.optional-dependencies]` dev: pytest, pytest-cov, ruff, ipykernel
   - `[project.scripts]`: 5 CLI entry points (`pctrans-download`, `pctrans-preprocess`, `pctrans-train`, `pctrans-evaluate`, `pctrans-query`)
3. Write `pctrans/__init__.py` (version = "0.1.0", import sentinel)
4. Write stub modules (empty classes/functions with `raise NotImplementedError`) for all 13 source files
5. Write `configs/model.yaml`, `configs/training.yaml`, `configs/data.yaml` with initial values
6. Write `CLAUDE.md` (extracted from Quick Reference above)
7. Write `tests/conftest.py` with fixtures: `tiny_ccle` (10 samples × 50 genes), `tiny_tcga` (20 samples × 50 genes), `tiny_model` (5 HVGs, embed_dim=8)
8. Write `tests/test_data.py` skeleton (1 test: assert tiny_ccle has correct shape)
9. Write `.github/workflows/ci.yml`: on push, run `uv run ruff check pctrans/ tests/` and `uv run pytest tests/`
10. `uv sync` — verify environment installs cleanly
11. Write `/day`, `/blog-draft`, `/gate-check` stub commands in `.claude/commands/`

**Verification:**
```
uv run ruff check pctrans/ tests/     # must pass (0 errors)
uv run pytest tests/ -q               # must pass (1 test)
python -c "import pctrans; print(pctrans.__version__)"  # prints 0.1.0
```

**Daily report:** `reports/day-01-scaffold.md`
**Commit:** `day 1: scaffold pctrans package, pyproject.toml, CI workflow, stub modules`
**Next up:** Day 2 — CCLE download script

---

### Day 2: CCLE Data Download

**Goal:** `data/raw/ccle/` populated with expression matrix and metadata, verified intact.

**Tasks:**

1. Implement `pctrans/data/ccle_client.py` — `CCLEClient` class:
   - `download_expression(out_dir, force=False)`: downloads `OmicsExpressionProteinCodingGenesTPMLogp1.csv` from DepMap 24Q4 Figshare
   - `download_metadata(out_dir, force=False)`: downloads `Model.csv`
   - Both methods: check if file exists (idempotent), stream download with tqdm progress bar, verify file size > 1MB
   - Log download URLs to `reports/day-02-ccle-download.md` so they can be audited
2. Implement `pctrans/scripts/download.py` — `download-ccle` sub-command via Typer
3. Write lineage filter function `filter_lineages(df_meta, lineages: list[str]) -> pd.Series[bool]`
   - Maps OncotreePrimaryDisease to {"LUAD", "BRCA", "SKCM"} labels
   - Include explicit alias table (e.g. "Melanoma" → "SKCM", "Lung Adenocarcinoma" → "LUAD")
4. Run `pctrans-download ccle --out-dir data/raw/ccle/`
5. Verify: print shape of loaded expression matrix, count samples per lineage, check for NaN values

**Expected outputs:**
```
data/raw/ccle/OmicsExpressionProteinCodingGenesTPMLogp1.csv  (~250 MB)
data/raw/ccle/Model.csv                                       (~5 MB)
```

**Expected lineage counts (approximate):**
```
LUAD: 97–112 cell lines
BRCA: 125–145 cell lines
SKCM: 62–78 cell lines
```

**Tests to add:**
```python
def test_ccle_client_filter_lineages(tiny_ccle_meta):
    filtered = filter_lineages(tiny_ccle_meta, ["LUAD", "BRCA", "SKCM"])
    assert filtered.sum() > 0

def test_ccle_expression_no_nan():
    # smoke test: assert loaded expression (first 10 rows) has no NaN
```

**Verification:**
```
uv run pytest tests/test_data.py -q -k "ccle"
```

**Daily report:** `reports/day-02-ccle-download.md`
**Commit:** `day 2: CCLE download client, lineage filter, Model.csv metadata parse`
**Next up:** Day 3 — TCGA download

---

### Day 3: TCGA Data Download

**Goal:** `data/raw/tcga/` populated with TCGA Pan-Cancer RNA-seq and phenotype, verified intact.

**Tasks:**

1. Implement `pctrans/data/tcga_client.py` — `TCGAClient` class:
   - `download_expression(out_dir, force=False)`: downloads TCGA Pan-Cancer expression matrix from UCSC Xena
     - URL: `https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz`
     - This is ~200 MB gzipped → ~1.5 GB uncompressed
     - Stream download with tqdm; gunzip in place
   - `download_phenotype(out_dir, force=False)`: downloads TCGA Pan-Cancer survival/phenotype
     - URL: `https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/Survival_SupplementalTable_S1_20171025_xena_sp.gz`
2. Add `download-tcga` sub-command to `pctrans/scripts/download.py`
3. Write lineage filter for TCGA: `filter_tcga_lineages(df_pheno, lineages) -> pd.Series[bool]`
   - Maps `cancer type abbreviation` column: "LUAD", "BRCA", "SKCM" (already correct)
4. Run `pctrans-download tcga --out-dir data/raw/tcga/`
5. Verify: print sample counts per lineage, check gene count (~20,530), first 3 genes

**Expected outputs:**
```
data/raw/tcga/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena    (~1.5 GB)
data/raw/tcga/Survival_SupplementalTable_S1_20171025_xena_sp.tsv       (~2 MB)
```

**Expected lineage counts (approximate):**
```
LUAD: 515 patients
BRCA: 1,093 patients  
SKCM: 469 patients
Total: ~2,077 patients
```

**Key Note — TCGA expression file format:** Rows are genes (HUGO symbols), columns are TCGA
sample IDs (e.g. "TCGA-3C-AAAU-01"). Load with:
```python
df = pd.read_csv(path, sep='\t', index_col=0).T  # transpose: samples × genes
```

**Tests to add:**
```python
def test_tcga_client_filter_lineages(tiny_tcga_meta):
    filtered = filter_tcga_lineages(tiny_tcga_meta, ["LUAD", "BRCA", "SKCM"])
    assert set(filtered.unique()).issubset({"LUAD", "BRCA", "SKCM"})
```

**Daily report:** `reports/day-03-tcga-download.md`
**Commit:** `day 3: TCGA download client, Xena phenotype parse, lineage filter`
**Next up:** Day 4 — feature synchronisation

---

### Day 4: Feature Synchronisation & HVG Selection

**Goal:** `data/processed/ccle_2k.parquet`, `data/processed/tcga_2k.parquet`, `data/processed/gene_list.txt` — all three lineages, top 2,000 HVGs.

**Tasks:**

1. Implement `pctrans/data/preprocessor.py` — `FeatureSynchroniser` class:
   - `load_ccle(raw_dir) -> tuple[pd.DataFrame, pd.DataFrame]`: loads expression + metadata, filters lineages, strips Entrez IDs from column names (`"EGFR (1956)"` → `"EGFR"`)
   - `load_tcga(raw_dir) -> tuple[pd.DataFrame, pd.DataFrame]`: loads expression + phenotype, filters lineages
   - `find_common_genes(ccle_genes, tcga_genes) -> list[str]`: set intersection, sort for reproducibility
   - `select_hvgs(ccle_expr, tcga_expr, common_genes, n_hvgs=2000) -> list[str]`: union-rank HVG method (see Dataset Specifications above)
   - `save_filtered(ccle_expr, tcga_expr, hvg_list, out_dir)`: saves parquets + gene_list.txt

2. Add `pctrans-preprocess` CLI with `--raw-dir`, `--out-dir`, `--n-hvgs` arguments

3. Run `pctrans-preprocess --raw-dir data/raw/ --out-dir data/processed/ --n-hvgs 2000`

4. Build EDA notebook `notebooks/01_eda.ipynb`:
   - Section 1: Sample count table (domain × lineage)
   - Section 2: Gene expression distribution (log-scale histogram per lineage + domain)
   - Section 3: PCA before training — colour by lineage, shape by domain (do cell lines and patients separate on PCA? They should — the domain gap is clearly visible)
   - Section 4: Top 20 HVGs by mean rank — are they biologically meaningful? (e.g. tumour markers like EGFR, CDH1, SOX10 should appear in top HVGs for respective lineages)
   - Section 5: Variance spectrum — plot gene rank vs. variance for CCLE and TCGA separately

**Expected outputs:**
```
data/processed/ccle_2k.parquet     shape: (~300, 2001)  [2000 genes + lineage column]
data/processed/tcga_2k.parquet     shape: (~2077, 2001)
data/processed/gene_list.txt       2000 HUGO symbols, one per line
```

**Key scientific check (in EDA notebook):** PCA of pooled CCLE + TCGA (pre-training) should show a
clear **domain gap** — PC1 or PC2 should separate cell lines from patients. This is the
"before" picture that your "after" UMAP (Day 11) will answer.

**Tests to add:**
```python
def test_hvg_selection_count(tiny_ccle_expr, tiny_tcga_expr):
    hvgs = select_hvgs(tiny_ccle_expr, tiny_tcga_expr, common_genes, n_hvgs=10)
    assert len(hvgs) == 10

def test_no_data_leakage_gene_selection():
    # HVG variance is computed on ALL samples (no split yet) — this is correct
    # Verify: gene_list is deterministic (sorted tiebreaking)
```

**Daily report:** `reports/day-04-hvg-selection.md` — include the EDA PCA "before" figure as ASCII summary and the top 10 HVGs for each lineage
**Commit:** `day 4: feature synchronisation, union-rank HVG selection, EDA notebook`
**Next up:** Day 5 — normalisation + train/val/test split

---

### Day 5: Preprocessing Pipeline & Train/Val/Test Split

**Goal:** Per-gene z-scored train/val/test splits with no data leakage. `data/processed/scalers.pkl` and `data/processed/splits.json`.

**Tasks:**

1. Extend `pctrans/data/preprocessor.py` — `DataSplitter` class:
   - `stratified_split(ccle_df, tcga_df, val_frac=0.15, test_frac=0.15, seed=42) -> dict`
     - Stratify by lineage within each domain separately
     - Returns `{"ccle": {"train": [...ids], "val": [...ids], "test": [...ids]}, "tcga": {...}}`
   - `fit_scalers(ccle_train_expr, tcga_train_expr) -> dict[str, StandardScaler]`
     - Fit ONE scaler per gene across pooled CCLE_train + TCGA_train
     - Returns `{"scaler": fitted_StandardScaler}`
     - CRITICAL: scaler is fit on TRAIN only, applied to val and test
   - `apply_scalers(expr_df, scaler) -> pd.DataFrame`
   - `save_splits(splits, scalers, out_dir)` — JSON + pickle

2. Add `--split` sub-command to `pctrans-preprocess` CLI

3. Run: `pctrans-preprocess --raw-dir data/raw/ --out-dir data/processed/ --n-hvgs 2000 --split`

4. Implement `pctrans/data/dataset.py`:
   - `CCLEDataset(expr_df, lineage_col) -> Dataset` — `__getitem__` returns `(tensor_2000, lineage_label_int)`
   - `TCGADataset(expr_df, lineage_col) -> Dataset` — same interface
   - Lineage encoding: `{"LUAD": 0, "BRCA": 1, "SKCM": 2}` — stored in `LINEAGE_TO_IDX` module constant

5. Implement `pctrans/data/sampler.py` — `StratifiedContrastiveBatchSampler`:
   - Input: CCLEDataset + TCGADataset
   - Output: iterator of `(ccle_batch, tcga_batch)` where each batch = 8 CCLE + 8 TCGA per lineage (48 total)
   - CCLE oversampling: uniform random with replacement (since N_CCLE << N_TCGA)
   - TCGA undersampling: shuffle per epoch, no replacement
   - Guarantee: each batch contains ≥1 sample of each lineage from each domain

**Split size check:**
```
CCLE train: ~216  |  CCLE val: ~45  |  CCLE test: ~45
TCGA train: ~1495 |  TCGA val: ~290 |  TCGA test: ~292
```

**Tests to add:**
```python
def test_no_data_leakage(splits):
    # train IDs ∩ val IDs == empty set, train IDs ∩ test IDs == empty set
    
def test_scaler_fit_on_train_only(scalers, ccle_train, tcga_val):
    # scaler.mean_ should not change if tcga_val is added

def test_stratified_sampler_lineage_balance(sampler):
    batch = next(iter(sampler))
    ccle_labels = [ccle_dataset[i][1] for i in batch["ccle_indices"]]
    assert len(set(ccle_labels)) == 3  # all 3 lineages present
```

**Daily report:** `reports/day-05-splits.md` — include split size table, verify per-lineage balance
**Commit:** `day 5: normalisation pipeline, stratified split, Dataset classes, StratifiedContrastiveBatchSampler`
**Next up:** Day 6 — dual-tower architecture

---

### Day 6: Dual-Tower Architecture & Loss Module

**Goal:** `CCLEEncoder`, `TCGAEncoder`, `DualTowerModel`, `SupConInfoNCELoss` — all tested with random batches.

**Tasks:**

1. Implement `pctrans/models/encoders.py`:
   - `MLPBlock(in_features, out_features, dropout, use_bn) -> nn.Module`: Linear + BN + ReLU + Dropout
   - `CCLEEncoder(input_dim=2000, hidden_dims=[1024, 512, 256, 128], embed_dim=64, dropout=0.3)`
   - `TCGAEncoder(input_dim=2000, hidden_dims=[1024, 512, 256, 128], embed_dim=64, dropout=0.3)`
   - Both share the same `MLPBlock` but are **separate** `nn.Module` instances with separate weights
   - `forward(x) -> torch.Tensor` (raw embedding, NOT L2 normalised)

2. Implement `pctrans/models/dual_tower.py`:
   - `DualTowerModel(ccle_encoder, tcga_encoder)`
   - `forward(x_ccle, x_tcga) -> tuple[Tensor, Tensor]`: returns L2-normalised (z_ccle, z_tcga)
   - `encode_ccle(x) -> Tensor`: L2-normalised embedding for a CCLE batch (inference)
   - `encode_tcga(x) -> Tensor`: L2-normalised embedding for a TCGA batch (inference)

3. Implement `pctrans/models/losses.py`:
   - `SupConInfoNCELoss(init_tau=0.07)`:
     - `self.log_tau = nn.Parameter(torch.tensor(math.log(1.0 / init_tau)))`
     - `forward(z_ccle, z_tcga, lineage_labels_ccle, lineage_labels_tcga) -> torch.Tensor`
     - Builds `pos_mask[i, j] = (lineage_labels_ccle[i] == lineage_labels_tcga[j])`
     - Computes similarity matrix S = z_ccle @ z_tcga.T / τ
     - Applies SupCon formula (see Loss Function section)
     - Returns scalar loss + logs `self.tau` to allow monitoring

4. Write `configs/model.yaml`:
   ```yaml
   input_dim: 2000
   hidden_dims: [1024, 512, 256, 128]
   embed_dim: 64
   dropout_high: 0.3
   dropout_low: 0.2
   init_tau: 0.07
   ```

**Tests to add:**
```python
def test_encoder_output_shape(tiny_model):
    x = torch.randn(8, 2000)
    z = tiny_model.encode_ccle(x)
    assert z.shape == (8, 64)

def test_l2_norm_unit_sphere(tiny_model):
    x = torch.randn(8, 2000)
    z = tiny_model.encode_ccle(x)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)

def test_loss_decreases_on_correct_batch():
    # manually construct a batch where all pairs are from same lineage
    # loss should be negative-log of softmax over 7 negatives
    # verify loss is a positive scalar

def test_temperature_is_positive():
    loss_fn = SupConInfoNCELoss(init_tau=0.07)
    assert loss_fn.log_tau.exp().item() > 0
```

**Verification:**
```
uv run pytest tests/test_models.py tests/test_losses.py -q  # all pass
```

**Daily report:** `reports/day-06-architecture.md` — include parameter count table per layer, verify total params ~8.6M
**Commit:** `day 6: CCLEEncoder, TCGAEncoder, DualTowerModel, SupConInfoNCELoss`
**Next up:** Day 7 — training loop + Blog Post 1 + Gate 0

---

### Day 7: Training Loop + Gate 0 + Blog Post 1

**Goal:** Full training pipeline connected end-to-end. Gate 0: data pipeline produces correct shapes, model forward pass works, at least 1 epoch completes without NaN loss.

**Tasks:**

1. Implement `pctrans/training/trainer.py` — `ContrastiveTrainer`:
   - `__init__(model, loss_fn, train_sampler, val_ccle, val_tcga, config, mlflow_run_name)`
   - `train(n_epochs) -> dict`: main training loop
     - Adam optimiser (lr from config), cosine LR schedule with linear warmup (5 epochs)
     - Per-epoch: forward pass on all train batches, compute SupCon loss, backward, clip grad norm to 1.0
     - Per-epoch validation: call `KNNValidationCallback`
     - Log to MLflow: train_loss, val_loss, val_knn_accuracy, temperature
     - Save checkpoint when val_knn_accuracy improves (early stopping patience=5)
   - `load_checkpoint(path) -> None`

2. Implement `pctrans/training/callbacks.py` — `KNNValidationCallback`:
   - `__call__(model, val_ccle_loader, val_tcga_loader, k=5) -> dict`
   - Freezes model (no_grad), embeds all val samples
   - For each CCLE val sample: find k nearest TCGA val samples in 64-dim L2 space (using scikit-learn `NearestNeighbors`)
   - Computes: fraction where majority neighbour lineage == true lineage
   - Returns `{"val_knn_accuracy": float, "per_lineage": {"LUAD": float, "BRCA": float, "SKCM": float}}`

3. Write `configs/training.yaml`:
   ```yaml
   n_epochs: 30
   batch_size: 48
   lr: 3.0e-4
   warmup_epochs: 5
   grad_clip_norm: 1.0
   knn_k: 5
   early_stop_patience: 5
   checkpoint_path: models/best_model.pt
   mlflow_experiment: pctrans-v1
   ```

4. Add `pctrans-train` CLI: `pctrans-train --config configs/training.yaml --data-dir data/processed/`

5. **Gate 0 check (manual, not automated):**
   - Run 1 epoch: `pctrans-train --epochs 1 ...`
   - Verify: loss is a positive finite scalar, no NaN/Inf
   - Verify: val kNN accuracy > 0 (even random = 33%)
   - Verify: temperature logs correctly

6. Draft `reports/blog-01-concept.md`:
   - Title: "The Cell Line Translation Problem: Why 85% of Cancer Drugs Fail Between Lab and Clinic"
   - Hook: The Phase II failure statistic + one concrete drug (Vemurafenib worked in SKCM cell lines AND patients — why? vs. hundreds that didn't)
   - Body: The domain gap (TME absence, serum artefacts), how InfoNCE solves it, the CCLE–TCGA setup, dual-tower architecture
   - "Next week" teaser: training run + UMAP reveal
   - 1,000–1,200 words, no hype, honest about what's being validated

**Gate 0 Decision:**
```
[Gate 0: Day 7]
├── ✓ 1 epoch completes, loss finite, kNN > 0 → PROCEED to Week 2
└── ✗ NaN loss or shape errors → DEBUG before Day 8
       Check: gene scaler producing inf values? Tau collapsing?
       Check: batch sampler producing duplicate indices?
```

**Tests to add:**
```python
def test_one_training_epoch(tiny_model, tiny_sampler):
    trainer = ContrastiveTrainer(...)
    result = trainer.train(n_epochs=1)
    assert math.isfinite(result["train_loss"])
    assert 0.0 <= result["val_knn_accuracy"] <= 1.0
```

**Verification:**
```
uv run pytest tests/ -q   # full suite passes
```

**Daily report:** `reports/day-07-training-loop.md` — include Gate 0 outcome, loss from epoch 1
**Commit:** `day 7: ContrastiveTrainer, KNNValidationCallback, pctrans-train CLI, blog-01 draft`
**Next up:** Day 8 — full 30-epoch training run

---

## Week 2 (Days 8–14): Training, Evaluation & Deployment

---

### Day 8: Training Run Execution

**Goal:** 30-epoch training run completed. Best model checkpoint saved. Val kNN accuracy tracked per epoch.

**Tasks:**

1. Launch training: `pctrans-train --config configs/training.yaml --data-dir data/processed/`
   - Expected runtime: ~12 min on CPU, ~3 min on Colab T4
   - Monitor in separate terminal: `mlflow ui` → observe training curves live

2. During training, watch for early warning signs:
   - **Loss goes to 0 immediately (epoch 1):** overfit; batch sampler may be leaking identities
   - **Loss stays at log(B_t) = log(24) ≈ 3.18:** model not learning; check learning rate
   - **Temperature collapses to <0.01:** numerical instability; clamp log_tau to [-4, 2]
   - **Val kNN accuracy < 33%:** worse than random — check label assignment in sampler

3. After training completes:
   - Print learning curve summary: loss at epoch 1, 10, 20, 30
   - Print best val kNN accuracy + which epoch it occurred
   - Verify `models/best_model.pt` saved

4. Run hyperparameter mini-sweep (optional if time allows on Colab — 3 extra runs):
   ```yaml
   # Sweep configs:
   run_A: lr=1e-3,  batch_size=48, init_tau=0.07
   run_B: lr=3e-4,  batch_size=48, init_tau=0.07  ← default
   run_C: lr=1e-4,  batch_size=32, init_tau=0.1
   ```
   - Compare val kNN accuracy across runs at epoch 20

**Expected training trajectory:**
```
Epoch  1: loss ~2.8,  val_knn ~0.33 (random baseline)
Epoch  5: loss ~2.2,  val_knn ~0.45
Epoch 10: loss ~1.8,  val_knn ~0.58
Epoch 15: loss ~1.5,  val_knn ~0.63
Epoch 20: loss ~1.3,  val_knn ~0.67
Epoch 25: loss ~1.2,  val_knn ~0.70
Epoch 30: loss ~1.1,  val_knn ~0.70–0.75
```
Note: These are estimates. If actual trajectory is steeper/slower, document and investigate.

**Daily report:** `reports/day-08-training-run.md` — paste actual learning curve numbers, note temperature evolution
**Commit:** `day 8: training run complete, best_model.pt saved, MLflow experiment pctrans-v1`
**Next up:** Day 9 — training analysis + debugging

---

### Day 9: Training Analysis & Debugging

**Goal:** Understand *why* the model is (or isn't) learning. Identify the top failure modes before Gate 1.

**Tasks:**

1. Build `notebooks/02_training_analysis.ipynb`:
   - **Panel 1:** Train loss vs. epoch (log scale) + val kNN@5 vs. epoch → dual-axis plot
   - **Panel 2:** Temperature τ evolution vs. epoch → does it converge? What final value?
   - **Panel 3:** Per-lineage val kNN accuracy vs. epoch (LUAD, BRCA, SKCM separately)
     - Diagnostic: if SKCM underperforms (<40%), check class balance in sampler
   - **Panel 4:** Gradient norm per encoder (CCLEEncoder, TCGAEncoder) vs. epoch
     - Diagnostic: if one encoder's grad norm is 10× the other, the towers are learning asymmetrically
   - **Panel 5:** t-SNE / UMAP of validation embeddings at epoch 1 vs. best epoch
     - Quick visual check: are clusters forming?

2. Run ablation on embed_dim: train 3 epochs each with embed_dim ∈ [32, 64, 128]
   - Report val kNN for each — include in Day 9 report
   - Stick with 64 unless 32 already achieves ≥70% (simpler model preferred)

3. Check for domain collapse: compute centroid of all CCLE embeddings and centroid of all TCGA embeddings in 64-dim space. Cosine similarity between centroids should be <0.5 (distinct domains still distinguishable post-alignment).

4. If val kNN at end of Day 8 is <60%: implement one fix from the Debug Protocol (see Gate Decision Architecture) and re-run 10 epochs to confirm improvement before Gate Day.

**Expected learnings to document:**
- Does τ converge to a specific value? (Literature: usually 0.02–0.05 for good models)
- Which lineage is hardest to translate? (Hypothesis: SKCM, because melanoma cell lines have strong melanocyte marker expression from culture that dominates over TME signals)
- Is the CCLE tower or TCGA tower providing stronger gradient signal?

**Daily report:** `reports/day-09-training-analysis.md` — include actual gradient norm plot, temperature trajectory, per-lineage kNN breakdown
**Commit:** `day 9: training analysis notebook, ablation results, embed_dim confirmed 64`
**Next up:** Day 10 — Gate 1 kNN evaluation

---

### Day 10: kNN Retrieval Evaluation [GATE 1]

**Goal:** Compute final kNN@5 retrieval accuracy on the held-out test set. Make deploy/pivot decision.

**Tasks:**

1. Implement `pctrans/evaluation/knn.py`:
   - `knn_retrieval_accuracy(model, ccle_test_loader, tcga_test_loader, k=5) -> dict`
   - Freeze model (no_grad), embed all test CCLE and TCGA samples
   - For each CCLE test sample: find k nearest TCGA test samples by L2 distance in 64-dim space
   - Compute: fraction where `mode(lineage of k neighbours) == true lineage(ccle)`
   - Report: overall accuracy + per-lineage accuracy + confusion matrix
   - Also compute kNN at k=1, 3, 5, 10 for supplementary table

2. Implement `pctrans/evaluation/silhouette.py`:
   - `cross_domain_silhouette(embeddings, lineage_labels, domain_labels) -> float`
   - Using scikit-learn's `silhouette_score` with `lineage_labels` (not domain_labels)
   - A positive silhouette score = within-lineage samples (across domains) are more similar to each other than to other lineages → alignment is working

3. Add `pctrans-evaluate` CLI: runs kNN + silhouette + saves results to `reports/eval_summary.json`

4. Implement `pctrans/evaluation/tfs.py`:
   - `translational_fidelity_score(knn_accuracy, silhouette_score) -> float`
   - `TFS = 0.5 * knn_accuracy + 0.5 * (silhouette_score + 1) / 2`
   - Range: [0, 1]. Interpretation: TFS > 0.6 = confident translation, TFS < 0.4 = poor translation
   - Per-cell-line TFS: use per-sample kNN fraction + per-sample silhouette contribution

5. Run: `pctrans-evaluate --model models/best_model.pt --data-dir data/processed/`

6. **Print Gate 1 Report:**
```
══════════════════════════════════════════════
           GATE 1 EVALUATION REPORT
══════════════════════════════════════════════
Overall kNN@5 Accuracy:  XX.X%   (threshold: 70%)
Per-lineage kNN@5:
  LUAD:  XX.X%
  BRCA:  XX.X%
  SKCM:  XX.X%
Silhouette Score:  +X.XX   (> 0 = good alignment)
TFS (composite):   X.XX    (> 0.6 = deploy)
Random baseline:   33.3%
PCA+kNN baseline:  ~55%
Harmony baseline:  ~63%
══════════════════════════════════════════════
DECISION: [DEPLOY / DEBUG]
══════════════════════════════════════════════
```

**Tests:**
```python
def test_knn_accuracy_random_model():
    # untrained model → kNN ≈ 33% (random)
    # Verify: knn_retrieval_accuracy returns ~0.33 ± 0.10

def test_knn_accuracy_perfect_embeddings():
    # synthetically perfect embeddings: all LUAD at [1,0,0], BRCA at [0,1,0], SKCM at [0,0,1]
    # kNN@5 should return 1.0
```

**Daily report:** `reports/day-10-gate-evaluation.md` — paste Gate 1 report, final decision, justification
**Commit:** `day 10: kNN evaluation, silhouette, TFS computation, Gate 1 outcome: [DEPLOY/DEBUG]`
**Next up:** Day 11 — UMAP visualisation (if DEPLOY) or debug protocol (if DEBUG)

---

### Day 11: UMAP Visualisation & TFS Analysis

**Goal:** Publication-quality UMAP of all test set embeddings. Per-cell-line TFS ranking. Biological interpretation of outliers.

**Tasks (DEPLOY path — assumes Gate 1 passed):**

1. Implement `pctrans/evaluation/viz.py`:
   - `umap_projection(embeddings, n_neighbors=15, min_dist=0.1, n_components=2, seed=42) -> np.ndarray`
     - Fit UMAP on ALL test embeddings (CCLE + TCGA pooled)
     - Return 2D coordinates
   - `lineage_domain_scatter(coords, lineage_labels, domain_labels, title) -> plotly.Figure`
     - Colour by lineage: LUAD=blue, BRCA=pink, SKCM=brown
     - Shape by domain: TCGA patients = circle (●), CCLE cell lines = cross (✕)
     - Interactive (hover shows sample ID + TFS)
   - `tfs_ranking_bar(cell_line_ids, tfs_scores) -> plotly.Figure`
     - Horizontal bar chart: top 10 and bottom 10 cell lines by TFS

2. Run UMAP on test embeddings, save figure as `reports/umap_test_set.html` (interactive) and `reports/umap_test_set.png` (static for blog/LinkedIn)

3. Investigate outlier cell lines (lowest TFS):
   - Which cell lines fail to cluster with their lineage?
   - Hypotheses: hypermutated lines, mis-classified primary disease, extreme passage number
   - Document in report — this is the *honest science* moment that makes the blog post credible

4. Compute the **"Before vs. After"** UMAP panel (for Blog Post 2):
   - Left: PCA of raw (unaligned) CCLE + TCGA test features → show domain gap
   - Right: UMAP of trained model embeddings → show domain alignment

5. Build `notebooks/03_evaluation.ipynb`:
   - Section 1: Gate 1 metrics table + confusion matrix
   - Section 2: UMAP (interactive Plotly)
   - Section 3: TFS ranking table (top/bottom 10 cell lines)
   - Section 4: Biological outlier analysis
   - Section 5: Comparison to baselines (random, PCA+kNN, Harmony)

**Biological interpretation questions to answer:**
1. Do LUAD cell lines cluster away from BRCA and SKCM patients? (Expected: yes — this validates the lineage alignment)
2. Do any BRCA TNBC cell lines land near LUAD patients? (Possible — TNBC shares proliferative signature with LUAD in some studies)
3. Is the SKCM cluster tighter than LUAD? (Hypothesis: yes — melanoma has the strongest melanocyte identity markers)

**Daily report:** `reports/day-11-umap-tfs.md` — include static UMAP image (described in text), per-lineage TFS mean, top/bottom 3 cell lines by TFS with biological hypothesis
**Commit:** `day 11: UMAP visualisation, TFS per cell line, outlier analysis, evaluation notebook`
**Next up:** Day 12 — Streamlit app + Blog Post 2

---

### Day 12: Streamlit App + Blog Post 2

**Goal:** Interactive demo live locally. Blog Post 2 drafted with UMAP figure ready.

**Tasks:**

1. Build `app/streamlit_app.py`:
   - **Sidebar:** "Select a CCLE cell line" dropdown (grouped by lineage, showing only LUAD/BRCA/SKCM)
   - **Main panel — 3 sections:**
     - **Section 1: Live UMAP**
       - Pre-computed embeddings loaded from `data/processed/embeddings_test.npz` (computed Day 11)
       - Selected cell line highlighted with ★ symbol, all others dimmed
       - Nearest 5 TCGA patients highlighted with ⬡ symbol
       - Plotly scatter (interactive hover shows sample ID, TFS, lineage)
     - **Section 2: TFS Gauge**
       - Display TFS for selected cell line (0.0 – 1.0 dial)
       - Colour: green (>0.7), yellow (0.5–0.7), red (<0.5)
       - Caption: "TFS measures how faithfully this cell line maps to its human counterpart in 64-dimensional latent space"
     - **Section 3: Nearest Patient Neighbours Table**
       - Table: rank, TCGA sample ID, lineage, cosine similarity, TCGA clinical annotation (tumour stage if available)
   - **Footer:** GitHub link, model architecture summary, methods one-liner

2. Pre-compute all CCLE cell line embeddings and save as `data/processed/ccle_embeddings.npz` (for fast app loading without model inference on CPU each time)

3. Test app locally: `streamlit run app/streamlit_app.py`
   - Test 3 cell lines (1 per lineage), verify UMAP highlights correctly
   - Test edge case: cell line with TFS < 0.4

4. Draft `reports/blog-02-results.md`:
   - Title: "Teaching AI to Translate Cell Lines: A UMAP Reveal After Two Weeks"
   - Hook: Show the "Before" PCA (domain gap) vs "After" UMAP (aligned clusters) — two images
   - Body: Gate 1 metrics, what the UMAP shows, the TFS metric, an outlier case study (honest about what failed)
   - Streamlit demo link (will go live Day 14)
   - "What I'd do next" section: molecular subtype labels, drug response integration, patient-matched iPSC data
   - 1,200–1,400 words, technical but accessible, honest about limitations

5. Draft LinkedIn Post 2:
   - Opens with the "before vs. after" comparison (static image)
   - 3-line hook: "Two weeks ago: cell lines and patients were mathematically irreconcilable. Today: they cluster by disease."
   - Tags 3 computational biology researchers in network

**Daily report:** `reports/day-12-app-blog.md`
**Commit:** `day 12: Streamlit app, pre-computed embeddings, blog-02 draft, LinkedIn-02 draft`
**Next up:** Day 13 — documentation + tests + README

---

### Day 13: Documentation, Tests & README

**Goal:** Test coverage ≥80%, all 5 docs complete, README with results table ready.

**Tasks:**

1. Complete 5 documentation files in `docs/`:
   - `01_data_pipeline.md`: CCLE DepMap URL, TCGA Xena URL, gene ID harmonisation step-by-step, HVG selection algorithm with worked example (top 5 genes + their ranks)
   - `02_feature_engineering.md`: Normalisation justification (why z-score after log1p?), data leakage prevention protocol, split strategy, scaler fitting details
   - `03_architecture.md`: Encoder layer diagram with dimensions, BatchNorm justification (small vs. large N imbalance), L2 normalisation on unit hypersphere, parameter count, why asymmetric design even with harmonised features
   - `04_training.md`: SupCon-InfoNCE derivation (the 5 equations from Loss section), learnable temperature rationale, batch construction algorithm, cosine LR schedule, early stopping
   - `05_evaluation.md`: kNN protocol (k=1,3,5,10 table), silhouette score interpretation, TFS composite formula, UMAP hyperparameter choices (n_neighbors=15, min_dist=0.1), comparison to baselines

2. Complete test suite to ≥80% coverage:
   - `tests/test_data.py`: download client idempotency, HVG selection count, gene ID parsing, split leakage check, sampler lineage balance
   - `tests/test_models.py`: encoder output shape, L2 norm unit sphere, forward pass dtype, gradient flow (both towers receive grad)
   - `tests/test_losses.py`: loss is positive scalar, loss decreases on perfect batch, temperature clipping, symmetric loss ≈ asymmetric × 2
   - `tests/test_training.py`: 1 epoch completes, checkpoint saves when kNN improves, early stopping fires
   - `tests/test_evaluation.py`: kNN on random embeddings ≈ 0.33, kNN on perfect embeddings = 1.0, silhouette on perfect clusters = 1.0, TFS range [0,1]
   - `tests/test_inference.py`: TranslationEmbedder loads checkpoint, embed_cell_line returns shape (1, 64)
   - `tests/test_scripts.py`: `pctrans-evaluate --help` exits 0

3. Write `README.md`:
   ```markdown
   # Pre-Clinical to Clinical Translation

   **Can a neural network map cancer cell lines to their human patient counterparts?**

   [Architecture diagram — ASCII art]

   ## Results (Test Set)

   | Metric | Random | PCA+kNN | Harmony | This Work |
   |---|---|---|---|---|
   | kNN@5 Accuracy | 33.3% | ~55% | ~63% | **XX.X%** |
   | Silhouette Score | — | — | — | **+X.XX** |
   | TFS (composite) | — | — | — | **X.XX** |

   ## Quick Start
   ...

   ## Live Demo
   https://pctrans.streamlit.app
   ```

4. Run coverage report: `uv run pytest tests/ --cov=pctrans --cov-report=term-missing`

**Verification:**
```
uv run ruff check pctrans/ tests/      # 0 errors
uv run pytest tests/ --cov=pctrans     # ≥80% coverage
```

**Daily report:** `reports/day-13-docs-tests.md` — include coverage % per module
**Commit:** `day 13: all 5 docs complete, test coverage ≥80%, README with results table`
**Next up:** Day 14 — GitHub release + Streamlit deploy + publish

---

### Day 14: GitHub Release & Public Launch

**Goal:** Repo public, Streamlit app live, Blog Post 1 + 2 published, LinkedIn posts scheduled, Twitter/X thread posted.

**Tasks:**

1. **Final code review:**
   - Remove any hardcoded paths or absolute directory references
   - Add `requirements.txt` (exported from `uv pip freeze > requirements.txt`)
   - Add `notebooks/colab_quickstart.ipynb`: single-notebook version that downloads data, trains, and generates UMAP — runnable on free Colab T4 in <10 minutes
   - Verify: `uv run pytest tests/` passes from fresh clone

2. **GitHub:**
   - Make repository public
   - Add GitHub release v0.1.0 with release notes:
     - kNN@5 test accuracy: XX.X%
     - Streamlit app link
     - Colab quickstart link
   - Add repository topics: `contrastive-learning`, `cancer-genomics`, `ccle`, `tcga`, `translational-biology`, `pytorch`, `streamlit`

3. **Streamlit Community Cloud deployment:**
   - Connect repo to Streamlit Cloud
   - Set Python version to 3.11
   - Main file: `app/streamlit_app.py`
   - Add `data/processed/ccle_embeddings.npz` and `data/processed/tcga_embeddings_test.npz` as Git-tracked (they're small — CCLE: ~300 samples × 64 dims × 4 bytes = ~77KB)
   - Verify deployment: test 3 cell lines, confirm UMAP renders, TFS gauge shows correctly

4. **Content publishing:**
   - Publish Blog Post 1 (if platform available, else save as Markdown in `/reports`)
   - Publish Blog Post 2 with UMAP figure
   - LinkedIn Post 1: schedule to post at 9am Tuesday (Day 7 topic — now going live)
   - LinkedIn Post 2: post immediately with UMAP figure and Streamlit link
   - Twitter/X thread (10 tweets — see Content Calendar)

5. **Final commit:**
   - Tag: `git tag -a v0.1.0 -m "Initial release: XX.X% kNN@5 accuracy on CCLE-TCGA alignment"`

**Verification:**
```
# From a clean directory:
git clone https://github.com/vthawfeek/pre-clinical-to-clinical-translation
cd pre-clinical-to-clinical-translation
pip install uv && uv sync
uv run pctrans-evaluate --model models/best_model.pt --data-dir data/processed/
# Expected: prints Gate 1 report with ≥70% kNN accuracy
```

**Daily report:** `reports/day-14-launch.md` — include Streamlit URL, GitHub release URL, LinkedIn post URLs
**Commit:** `day 14: v0.1.0 release, Streamlit deployed, Colab quickstart, public launch`

---

## Gate Decision Architecture

### Gate 0 (Day 7): Data + Architecture Sanity

```
[Gate 0: Does 1 training epoch complete without NaN and with kNN > 0?]
├── PASS → Proceed to Week 2
└── FAIL → Debug (max 4 hours before Day 8):
    ├── NaN loss:     check scaler for Inf values (columns with zero variance → z-score blows up)
    ├── Shape errors: check sampler CCLEDataset/TCGADataset __len__ and __getitem__
    └── kNN = 0:      check lineage label integer encoding (LINEAGE_TO_IDX consistent across datasets?)
```

### Gate 1 (Day 10): kNN Retrieval Accuracy

```
[Gate 1: kNN@5 Test Set Retrieval Accuracy]
│
├── ≥ 70%  ──► DEPLOY PATH (Days 11–14)
│              The model reliably clusters cell lines with their human lineage counterparts.
│              Proceed to UMAP, Streamlit, and publish.
│
├── 60–70% ──► SOFT FAIL: Debug 1 fix, re-run 10 epochs
│              Most likely cause: suboptimal hyperparameters or tau not converging
│              Fix: lower lr to 1e-4, increase warmup to 10 epochs, re-check tau clamp
│              Re-evaluate on Day 11 with extended training
│
├── 50–60% ──► HARD FAIL: Batch construction issue
│              Cell lines are aligning partially but inter-lineage boundaries are blurry
│              Fix: tighten positive mask — ensure zero within-lineage negatives (re-check sampler)
│              Consider: add L2 regularisation on embeddings (λ=1e-4 weight decay)
│              Re-evaluate on Day 11
│
└── < 50%  ──► ARCHITECTURE FAILURE: PIVOT
               kNN worse than PCA baseline: encoder is discarding lineage signal
               Options:
               (a) Use Harmony batch correction as baseline + kNN (3 hours to implement)
               (b) Reframe project: document failure case + correct analysis, publish as
                   "Why InfoNCE Alignment Fails on Low-Sample Cross-Domain RNA-seq (< 300 cell lines)"
                   This is a VALID and publishable finding — failure analysis with correct
                   statistical reasoning is more impressive than a hidden or overclaimed result.
               Do NOT proceed to Streamlit deployment on a broken signal.
```

**Baselines to compare against (required for honest reporting):**

| Method | Expected kNN@5 | Notes |
|---|---|---|
| Random assignment | 33.3% | Theoretical minimum (3 equal classes) |
| PCA (50 PCs) + L2 kNN, no alignment | ~50–55% | Run with `sklearn.decomposition.PCA` |
| Harmony batch correction + kNN | ~60–65% | `harmonypy` library, treat domain as batch |
| **Our InfoNCE dual-tower** | **≥70%** | **Target** |

Run all baselines on Day 10 so the Gate 1 report includes a proper comparison table.

---

## Evaluation Framework

### Primary Metric: kNN@5 Retrieval Accuracy

**Protocol:**
- Use HELD-OUT test set only (never val set — val was used for early stopping)
- Embed all test CCLE cell lines using CCLEEncoder (frozen)
- Embed all test TCGA patients using TCGAEncoder (frozen)
- For each CCLE cell line, find 5 nearest TCGA patients in L2 space (64-dim)
- Accuracy = fraction of cell lines where `mode(lineage of 5 TCGA neighbours) == true lineage`
- Report at k = 1, 3, 5, 10 for supplementary table

**Interpretation:** kNN accuracy captures what matters clinically: given a cell line representing
a drug experiment, which human patients are its closest analogues? If a BRCA cell line finds BRCA
patients as its 5 nearest neighbours, a drug tested on that cell line is being compared to the
right patient population.

### Secondary Metric: Cross-Domain Silhouette Score

- Compute on POOLED CCLE + TCGA test embeddings
- Use lineage as the cluster label (NOT domain)
- A positive silhouette = within-lineage cross-domain cohesion > between-lineage separation
- A negative silhouette = domain artefacts dominate over lineage signal (failure mode)

### Composite Metric: TFS (Translational Fidelity Score)

```
TFS(cell_line_i) = 0.5 × kNN_match_fraction_i + 0.5 × (silhouette_contribution_i + 1) / 2
Range: [0, 1]
Interpretation:
  TFS > 0.70: High fidelity — this cell line is a good model for its lineage
  TFS 0.50–0.70: Moderate fidelity — use with caution in drug studies
  TFS < 0.50: Poor fidelity — domain artefacts dominate; deprioritise for translational work
```

**Per-cell-line TFS** enables a ranked list: "which CCLE cell lines are most reliable for
translational research?" This is the key portfolio-differentiating output.

### UMAP Visualisation

- Fit UMAP on ALL test set embeddings (CCLE + TCGA pooled, 64-dim → 2-dim)
- Hyperparameters: n_neighbors=15, min_dist=0.1, metric="cosine"
- Colour by lineage (LUAD=blue, BRCA=pink, SKCM=brown)
- Shape by domain (● = TCGA patient, ✕ = CCLE cell line)
- Expected: 3 distinct clusters, each cluster containing mixed CCLE+TCGA

**Key visual to produce:** Side-by-side PCA (raw, pre-training) vs. UMAP (post-training) to demonstrate the domain gap before and after alignment. This is the single most shareable figure in the project.

---

## Content Calendar

### Blog Post 1 (Day 7, publish Day 14)
**Title:** "The Cell Line Translation Problem: Why 85% of Cancer Drugs Fail Between Lab and Clinic"
**Platform:** Personal blog or Substack / Medium
**Word count:** 1,000–1,200
**Sections:**
1. The failure statistic (hook)
2. What causes the domain gap (TME, serum, passage — with one figure: cell line vs. tumour histology)
3. How contrastive learning addresses it (InfoNCE intuition, no labels needed for alignment)
4. The CCLE–TCGA experimental setup
5. Week 1 progress: architecture live, data ingested, first epoch running
6. "Come back next week for the UMAP reveal"
**Key image:** Dual-tower architecture diagram

### Blog Post 2 (Day 12, publish Day 14)
**Title:** "Teaching AI to Translate Cell Lines: A UMAP Reveal and What the Numbers Actually Say"
**Platform:** Same as Blog Post 1
**Word count:** 1,200–1,400
**Sections:**
1. The "Before vs. After" UMAP (hook — two figures, maximum visual impact)
2. Gate 1 metrics table (honest numbers, baselines included)
3. The TFS metric — top 5 and bottom 5 cell lines with biology hypothesis
4. One outlier case study: "Why does [CELL LINE X] land near the wrong patient cluster?"
5. Limitations: soft positives, no molecular subtype labels, bulk RNA only
6. What's next: drug response integration, scRNA-seq, patient-matched iPSC
**Key images:** Before/after UMAP, TFS bar chart

### LinkedIn Post 1 (Day 14, 9am)
```
The Phase II oncology failure rate is 67%. Most drugs work beautifully in cell lines.
They fail in patients.

I spent this week building a neural network that tries to bridge that gap:
training a dual-tower InfoNCE contrastive model to map CCLE cancer cell lines
directly into TCGA patient space.

The architecture: two MLP towers sharing no weights, each processing 2,000 highly
variable genes, pushing representations toward a 64-dimensional shared manifold.

Week 2 brings the UMAP reveal — do the cell lines actually land near their human
counterparts, or does petri-dish noise win?

Blog post (with architecture deep-dive): [LINK]

#ComputationalBiology #MachineLearning #CancerResearch #TranslationalMedicine
```

### LinkedIn Post 2 (Day 14, immediately after deployment)
```
Two weeks ago: cancer cell lines and human tumours formed completely separate clouds
in gene expression space.

Today: [UMAP FIGURE]

Left: raw PCA — CCLE and TCGA are irreconcilably separate (this is the clinical
translation problem in mathematical form).

Right: trained model UMAP — LUAD cell lines cluster with LUAD patients.
BRCA with BRCA. SKCM with SKCM.

kNN@5 retrieval accuracy: XX.X%  (random baseline: 33.3%, Harmony baseline: ~63%)

Live demo — pick any CCLE cell line and see exactly where it lands in patient space:
👉 https://pctrans.streamlit.app

Full code: https://github.com/vthawfeek/pre-clinical-to-clinical-translation

#CancerGenomics #ContrastiveLearning #CCLE #TCGA #Bioinformatics #OpenScience
```

### Twitter/X Thread (Day 14)
```
1/ I spent 2 weeks building a neural network to translate cancer cell lines into 
   human patient space. Here's what I found. 🧵

2/ The problem: 85% of cancer drugs that work in cell lines fail in patients.
   Root cause: cell lines grown in a petri dish ≠ a human tumour living in an
   immune system. The gene expression patterns are genuinely different.

3/ The idea: CLIP-style contrastive learning. Train two encoders — one for CCLE
   cell lines, one for TCGA patients — to share a latent space where the same
   cancer lineage clusters together, regardless of which domain it came from.

4/ The data: 300 CCLE cell lines + 2,000 TCGA patients across 3 lineages
   (LUAD, BRCA, SKCM). Top 2,000 highly variable genes. Freely available.
   Zero institutional permissions needed.

5/ The key math: SupCon-InfoNCE loss. For each cell line anchor, ALL same-lineage
   patients are treated as positives. Cross-lineage patients are negatives.
   Learnable temperature τ ≈ 0.03 by epoch 30.

6/ The result: [UMAP IMAGE]
   kNN@5 accuracy = XX.X% (random = 33.3%, Harmony batch correction = ~63%)
   The LUAD cell lines land inside the LUAD patient cluster. It works.

7/ The honesty part: SKCM had the highest TFS (XX). BRCA had the lowest (XX).
   Hypothesis: BRCA has 4 molecular subtypes (LumA, LumB, HER2+, TNBC) that
   our lineage-level labels collapse — within-lineage heterogeneity is high.

8/ Every cell line now has a Translational Fidelity Score (TFS 0–1).
   The bottom 3: [CELL LINE A] (TFS=0.31), [B] (0.35), [C] (0.39).
   These should not be used for translational drug studies. The math says so.

9/ Live demo: select a CCLE cell line, watch it land in the UMAP of human patients.
   👉 https://pctrans.streamlit.app

10/ Full code, docs, Colab notebook:
    👉 https://github.com/vthawfeek/pre-clinical-to-clinical-translation
    
    Next: integrate drug response data (PRISM) to see if TFS predicts drug
    concordance between cell line and patient. 
```

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| DepMap download URL changes (24Q4 → 25Q1) | Medium | Low | Use DepMap API endpoint (stable) + fallback to manual Figshare link in README |
| TCGA Xena file >1.5 GB causes OOM on load | Medium | Medium | Load in chunks: `pd.read_csv(path, sep='\t', chunksize=1000, index_col=0)`, filter lineages per chunk |
| Gene ID mismatch (CCLE HUGO vs TCGA HUGO version difference) | Low | High | Cross-reference with HGNC gene symbol lookup table as fallback; require ≥17,000 common genes |
| τ collapse to ~0 (numerical) | Low | Medium | Clamp `log_tau` to [-4, 2] in forward pass: `tau = self.log_tau.clamp(-4, 2).exp()` |
| kNN accuracy stuck at random (~33%) | Medium | High | See Gate 1 debug protocol. Diagnose with gradient norms (Day 9) before Gate Day |
| Domain collapse (both towers learn same representation) | Low | High | Monitor cosine sim between CCLE centroid and TCGA centroid per epoch; should stay < 0.5 until late training |
| Streamlit Cloud free tier memory limit (1 GB) | Medium | Low | Pre-compute all embeddings; app loads `.npz` files (not the model) — total app memory < 200 MB |
| SKCM cell line class imbalance (only ~70 cells) | High | Medium | Confirm per-epoch stratified sampling enforces SKCM ≥ 8 samples/batch; add SKCM-specific kNN monitoring |
| Over-claiming performance vs. baselines | — | Career risk | Always report Harmony baseline; report kNN on TEST set (not val); include confidence interval via bootstrap |

---

## Daily Report Template

```markdown
# Day N: [Short Title]

**Date:** YYYY-MM-DD
**Commit:** `day N: <description>`

## What Was Built

[Bullet list of files created/modified with what they contain]

## What Was Learned

[2–4 bullet points: surprises, unexpected behaviour, scientific observations]

## Key Decisions

[1–3 decisions made today with brief justification. Only include non-obvious decisions.]

## Verification

[Paste actual output of: ruff check, pytest, and any key print statements]

## Numbers (if applicable)

[Any metrics, shapes, counts, or timings from today's work]

## Next Up

[Day N+1 tasks — 3–5 bullets]
```

---

## Commit Message Convention

- Format: `day N: <imperative description>`
- Examples:
  - `day 4: union-rank HVG selection, ccle_2k.parquet and tcga_2k.parquet saved`
  - `day 6: CCLEEncoder, TCGAEncoder, SupConInfoNCELoss with learnable temperature`
  - `day 10: Gate 1 evaluation: kNN@5=72.3%, silhouette=+0.41, DEPLOY decision`
- Always co-author: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`
- One commit per day (not per file — squash if needed before pushing)

---

## Quality Gates (Mandatory Every Day)

Before committing any day's work, BOTH must pass:

```bash
uv run ruff check pctrans/ tests/    # must exit 0
uv run pytest tests/ -q              # must exit 0 (all tests pass)
```

If either fails: fix before committing. Never commit red tests or lint errors.

---

## pyproject.toml Template

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pctrans"
version = "0.1.0"
description = "Contrastive alignment of CCLE cell lines to TCGA patients via dual-tower InfoNCE"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2.0",
    "pandas>=2.0",
    "numpy>=1.26",
    "pyarrow>=15.0",
    "scikit-learn>=1.4",
    "umap-learn>=0.5",
    "streamlit>=1.34",
    "plotly>=5.20",
    "mlflow>=2.12",
    "typer>=0.12",
    "pyyaml>=6.0",
    "requests>=2.31",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
    "ipykernel>=6.0",
    "harmonypy>=0.0.9",
]

[project.scripts]
pctrans-download   = "pctrans.scripts.download:app"
pctrans-preprocess = "pctrans.scripts.preprocess:app"
pctrans-train      = "pctrans.scripts.train:app"
pctrans-evaluate   = "pctrans.scripts.evaluate:app"
pctrans-query      = "pctrans.scripts.query:app"

[tool.hatch.build.targets.wheel]
packages = ["pctrans"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.pytest.ini_options]
addopts = "--tb=short"
markers = [
    "slow: marks tests that require real data (deselect with -m 'not slow')",
    "integration: marks tests that require downloaded files",
]
```

---

## Anticipated Final README Results Table

Fill in XX.X% values on Day 10 after Gate 1 evaluation.

| Metric | Random | PCA+kNN | Harmony | **pctrans (ours)** |
|---|---|---|---|---|
| kNN@5 Accuracy (Test) | 33.3% | ~54% | ~63% | **XX.X%** |
| kNN@1 Accuracy (Test) | 33.3% | ~49% | ~59% | **XX.X%** |
| Silhouette Score | — | ~0.05 | ~0.18 | **+X.XX** |
| TFS (composite) | — | — | — | **X.XX** |
| LUAD kNN@5 | 33.3% | ~58% | ~67% | **XX.X%** |
| BRCA kNN@5 | 33.3% | ~51% | ~60% | **XX.X%** |
| SKCM kNN@5 | 33.3% | ~52% | ~62% | **XX.X%** |

*Harmony baseline computed with `harmonypy` treating "domain" (CCLE/TCGA) as the batch covariate.
PCA baseline: top 50 PCs of concatenated CCLE+TCGA, kNN in PC space.*

---

## Appendix: Scientific Rationale for Key Choices

### Why Not Use Harmony Directly?

Harmony is a linear batch correction method. It removes batch effects by iteratively computing
cluster centroids and projecting samples onto a shared subspace. It is powerful but has a key
limitation: it assumes the batch effect is additive and linear. The CCLE–TCGA domain gap
is not purely linear — the absence of TME creates a complex non-linear manifold deformation.
A deep non-linear encoder (our dual-tower MLP) can in principle capture this non-linearity.
We compare to Harmony as a strong baseline, not as a prior method we are replacing.

### Why Log₂(TPM+1) Is Already Applied — And Why We Still Z-Score

DepMap CCLE and UCSC Xena TCGA data are both already log₂(TPM+1). The log transform compresses
the heavy tail of gene expression (ACTB, GAPDH expressing at 10,000× a rare marker gene).
However, log₁(TPM+1) still leaves inter-gene scale differences (highly expressed genes have
larger absolute values). Per-gene z-scoring normalises each gene to zero mean and unit variance
across the training set, making gradient updates for low-expressed and high-expressed genes
equally informative.

### Why 3 Cancer Types (Not All 30+ in TCGA)?

1. **Sample size constraint on CCLE:** many lineages have <20 cell lines — insufficient for
   stratified batching (need ≥8 per batch slot).
2. **Biological diversity:** LUAD (epithelial/lung), BRCA (glandular/breast), SKCM (neural crest/melanocyte)
   represent three distinct developmental origins. If the model works across all three, it is
   learning true lineage identity, not just a lung-vs-not-lung classifier.
3. **Interpretability:** 3 clusters make the UMAP visually clear and the kNN threshold meaningful.
   With 10+ lineages, a 70% kNN threshold would be trivially achievable by clustering.

### Why 2,000 HVGs (Not More, Not Fewer)?

- <500 genes: risk of losing meaningful transcriptional signatures for rare pathways
- 2,000 genes: standard in single-cell RNA-seq literature (Seurat HVG default)
- >5,000 genes: increases model size and training time without proportional accuracy gain on 3-class problem
- 2,000 is the sweet spot for a 5-day portfolio timeline on a CPU-only machine

### Why SupCon Instead of Standard CLIP InfoNCE?

In standard CLIP: each anchor has exactly one positive. If we randomly pair CCLE_LUAD_1 with
TCGA_LUAD_5, then TCGA_LUAD_1 through TCGA_LUAD_4 and TCGA_LUAD_6+ are treated as negatives
for that anchor — even though they are biologically appropriate positives.

SupCon allows multiple positives per anchor. For CCLE_LUAD_i: ALL 8 TCGA_LUAD samples in the
batch are positives. This eliminates false negatives and gives a cleaner gradient signal. The
SupCon formulation is strictly more correct for our setting and requires minimal additional code.

---

*Plan version 1.0 — 2026-06-30. Update CLAUDE.md daily with day completion status.*
