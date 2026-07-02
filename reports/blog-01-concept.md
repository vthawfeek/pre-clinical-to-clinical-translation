# The Cell Line Translation Problem: Why 85% of Cancer Drugs Fail Between Lab and Clinic

A drug shrinks a tumour in a petri dish. It shrinks the same tumour type in a mouse. Then it reaches
a Phase II trial in actual patients and does nothing. This happens so often that the oncology Phase II
failure rate sits around 67%, and roughly 85% of cancer drugs that look promising in preclinical
models never make it through human trials. Each of those failures costs years and hundreds of millions
of dollars.

Vemurafenib is the counterexample that makes the pattern obvious. It targets the BRAF V600E mutation,
it worked in melanoma (SKCM) cell lines, and it worked in melanoma patients. The biology that the cell
line captured was the biology that mattered in people. For hundreds of other compounds, the cell line
captured something else: a version of the disease that only exists on plastic.

## Why a cell line is not a patient

A cancer cell line is a population of tumour cells that has been growing in a flask for years, sometimes
decades. That environment rewrites its gene expression in ways that have nothing to do with the disease
in a human body.

Three forces drive the gap. First, there is no tumour microenvironment. A real tumour lives inside an
immune system, a blood supply, and a scaffold of stromal and endothelial cells. A flask has none of
that, so roughly 800 microenvironment-related genes go quiet. Second, culture media is engineered to
make cells divide as fast as possible, which pins proliferation markers like MKI67 and PCNA high.
Third, every passage adds mutations and selects for whatever grows best on a 2D surface, so the line
drifts further from its origin over time.

The result is measurable. If you take bulk RNA-seq from cell lines and from patient tumours and plot
them together, the two groups separate into distinct clouds. The largest axis of variation is not lung
versus breast versus skin. It is dish versus body. That domain gap is the translation problem written
in gene expression.

## What contrastive learning actually fixes

The instinct is to "correct" the batch effect and subtract the difference. Contrastive learning takes
a different route: instead of erasing the gap, it learns what survives across it.

The idea comes from CLIP, the model that aligns images and text captions. Give it two encoders and many
matched pairs, and it learns a shared space where a photo of a dog and the words "a dog" land in the
same place, while a photo of a dog and the words "a car" land far apart. It never needs a label for
"dog." It only needs to know which pairs belong together.

Swap the modalities. One encoder reads cell line expression, the other reads patient expression, and a
matched pair is any cell line and patient of the same cancer lineage. Train the encoders to pull
same-lineage pairs together and push different-lineage pairs apart, and the shared space they build has
to be made of signal that both domains agree on. That signal is cancer lineage identity. The culture
artefacts, which exist in only one domain, get discarded because they cannot help match the pairs.

There is an information-theoretic reason this works. The InfoNCE loss is a lower bound on the mutual
information between the two domains. Pushing it down forces the encoders to keep exactly the shared
information and drop the rest. No manual list of "batch genes" required.

## The experimental setup: CCLE meets TCGA

The setup uses two public datasets, so anyone can reproduce it without an access application.

CCLE (the Cancer Cell Line Encyclopedia, via the DepMap 24Q4 release) supplies the cell lines. TCGA
(The Cancer Genome Atlas, via the UCSC Xena hub) supplies the patient tumours. Both are bulk RNA-seq
measured on the same kind of platform, both use human HUGO gene symbols, so matching them is symbol
lookup rather than cross-species translation.

Three lineages carry the experiment: LUAD (lung adenocarcinoma), BRCA (breast cancer), and SKCM
(melanoma). Together they account for a large share of new cancer diagnoses, and their labels act as
ground truth. If the model works, a LUAD cell line should land near LUAD patients, not near breast or
skin patients. After harmonising the gene spaces and selecting the top 2,000 most variable genes, the
training set holds 183 cell lines and 1,586 patients, with more held out for validation and testing.

## The architecture

Two towers, no shared weights:

```
CCLE RNA-seq (2,000 HVGs)              TCGA RNA-seq (2,000 HVGs)
          |                                       |
    CCLEEncoder (MLP)                       TCGAEncoder (MLP)
   2000-1024-512-256-128-64               2000-1024-512-256-128-64
          |                                       |
   L2-normalised z_c  ------ cosine sim ------ L2-normalised z_t
                              |
             SupCon-InfoNCE loss (learnable temperature)
     "pull same-lineage cross-domain pairs together, push the rest apart"
```

Each tower is a small MLP that compresses 2,000 genes down to a 64-dimensional embedding, then
projects that embedding onto a unit sphere so that a dot product becomes a cosine similarity. The two
towers share their architecture but not their weights, because in production the two domains will not
always share a feature space, and the design should already handle that asymmetry. The whole model is
5,500,288 parameters, small enough to train on a laptop CPU.

One detail matters for honesty about the loss. Standard CLIP assumes exactly one correct partner per
anchor. Here every same-lineage patient is a correct partner for a given cell line, so the loss uses a
multi-positive (SupCon-style) form. That change removes a subtle bug: without it, the 15 other LUAD
patients in a batch would be treated as negatives for a LUAD cell line, punishing the model for
correct alignment. The temperature that controls how sharp the matching is is not hand-tuned; the model
learns it.

## Week 1: the pipeline is alive

By the end of week one the full path runs end to end: download, gene harmonisation, variable-gene
selection, leak-free train/validation/test splits, both encoder towers, the loss, and a training loop
that logs to MLflow and checkpoints on validation retrieval accuracy. The first training epoch
completes with a finite loss (7.68 train, 8.78 validation) and no numerical blowups, and the learnable
temperature holds steady near its 0.07 start. That is the bar for the week-one gate, and it passed.

The validation retrieval number after that single epoch is already well above the 33% you would get
from random guessing across three lineages. I am deliberately not reporting it as a headline result
yet. It comes from the validation set, not the held-out test set, and one epoch is not a trained model.
The honest number is the test-set kNN accuracy after full training, measured against real baselines
(PCA, and Harmony batch correction), and that is next week's job.

## Come back next week for the UMAP reveal

The question week two answers is visual and blunt. Take every test cell line and every test patient,
run them through the trained towers, and project the 64-dimensional embeddings down to two dimensions.
Before training, the map shows two separate clouds: dish and body. If the model learned what it was
supposed to learn, the after map shows three clusters split by lineage, each one mixing cell lines and
patients together. If it did not, the clouds stay apart and the project becomes an honest writeup of
why InfoNCE alignment fails on low-sample cross-domain RNA-seq. Either way, the numbers and the figure
go up next week.

Code: https://github.com/vthawfeek/pre-clinical-to-clinical-translation
