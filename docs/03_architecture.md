# 03 · Architecture

The dual-tower encoder that maps CCLE cell lines and TCGA patients into one shared
64-dimensional unit hypersphere.

Code: [`pctrans/models/encoders.py`](../pctrans/models/encoders.py),
[`pctrans/models/dual_tower.py`](../pctrans/models/dual_tower.py).

## 1. Layer diagram

Both towers share the same MLP *template* but have **independent weights**:

```
 CCLE x (2000-d, z-scored)              TCGA x (2000-d, z-scored)
        │                                       │
 ┌──────▼───────┐                        ┌──────▼───────┐
 │ MLPBlock     │ 2000 → 1024  dropout .3 │ MLPBlock     │ 2000 → 1024  dropout .3
 │ MLPBlock     │ 1024 →  512  dropout .3 │ MLPBlock     │ 1024 →  512  dropout .3
 │ MLPBlock     │  512 →  256  dropout .3 │ MLPBlock     │  512 →  256  dropout .3
 │ MLPBlock     │  256 →  128  dropout .2 │ MLPBlock     │  256 →  128  dropout .2  ← dropout_low
 │ Linear (proj)│  128 →   64  (bare)     │ Linear (proj)│  128 →   64  (bare)
 └──────┬───────┘                        └──────┬───────┘
        │  z_ccle (64-d, raw)                   │  z_tcga (64-d, raw)
        └───────────────► L2 normalise ◄────────┘
                     (DualTowerModel.forward)
        z_ccle, z_tcga on the unit hypersphere ‖z‖₂ = 1
```

`MLPBlock = Linear → BatchNorm1d → ReLU → Dropout`. The final **projection head is a
bare `Linear`** (no BN / activation / dropout) so the pre-normalisation embedding is
unbounded; the unit-sphere constraint is applied once, centrally, in
`DualTowerModel`.

The last hidden block uses `dropout_low = 0.2`; earlier blocks use
`dropout_high = 0.3` — heavier regularisation early where there are the most
parameters, lighter near the embedding so the geometry isn't over-perturbed.

## 2. Why BatchNorm — the small-N / large-N imbalance

BatchNorm sits **before** the activation in every hidden block. Its job here is
specifically about the **domain imbalance**: CCLE has ~hundreds of cell lines,
TCGA has ~thousands of patients, and within a mixed batch the two domains carry
different expression distributions for the same gene. BatchNorm re-centres and
re-scales each layer's pre-activations per mini-batch, which keeps the scale of
inter-layer activations stable as CCLE and TCGA samples flow through the **same
tower template**. Without it, the larger-magnitude domain would repeatedly push
activations into the ReLU's saturated/dead region and destabilise training.

At inference (`model.eval()`) BatchNorm switches to its running statistics, so a
single cell line can be embedded deterministically — a batch of one is valid, which
is what `TranslationEmbedder.embed_cell_line` relies on.

## 3. L2 normalisation — the unit hypersphere

`DualTowerModel.encode_ccle` / `.encode_tcga` apply
`F.normalize(·, dim=-1)` to the raw 64-d projection, so every embedding has unit
L2 norm (`test_l2_norm_unit_sphere`). Consequences:

- A dot product between two embeddings **is** their cosine similarity, so the
  contrastive loss operates directly in cosine space.
- Retrieval geometry is scale-free: only *direction* (expression profile shape)
  matters, not magnitude — the right invariance for comparing two assays.

## 4. Parameter count

The `[2000, 1024, 512, 256, 128, 64]` template is ~**2.75 M** parameters per tower,
so the dual-tower model is ~**5.5 M** total (asserted in
`test_total_param_count_around_5_5m`: `5.0M < total < 6.0M`). Small enough to train
in ~3 minutes on a Colab T4 (~12 minutes on CPU).

## 5. Why asymmetric weights, even with a shared gene space?

The features are harmonised (same 2,000 HUGO symbols, same z-scoring), so a single
shared encoder is *tempting*. We deliberately keep **two separate weight sets**
because harmonised **features** are not harmonised **distributions**:

- A cell line is a pure, proliferating, stroma-free in-vitro population.
- A patient tumour is a mixture — tumour cells plus immune infiltrate, stroma,
  vasculature — profiled in vivo on a different platform with batch effects.

The mapping from "gene-expression vector" to "lineage-discriminative embedding" is
therefore **domain-specific**: the same gene can be informative in one domain and
confounded in the other. Two towers let each domain learn its own transformation
while the contrastive loss pulls the *outputs* into a common space. This is exactly
the CLIP-style dual-encoder pattern (image encoder ≠ text encoder), adapted to two
biological assays. `test_towers_have_separate_weights` confirms the two towers hold
independent parameter tensors.
