# 04 · Training

The SupCon-style multi-positive InfoNCE objective, the stratified batch
construction that makes it valid, and the optimisation schedule.

Code: [`pctrans/models/losses.py`](../pctrans/models/losses.py),
[`pctrans/data/sampler.py`](../pctrans/data/sampler.py),
[`pctrans/training/trainer.py`](../pctrans/training/trainer.py).
CLI: `pctrans-train`. Config: [`configs/training.yaml`](../configs/training.yaml).

## 1. The loss — SupCon-InfoNCE derivation

Standard CLIP InfoNCE treats exactly one pair (i, i) as positive and everything
else in the batch as negative. That is **wrong here**: a batch deliberately
contains many same-lineage samples, so a plain InfoNCE would treat a LUAD patient
as a *negative* for a LUAD cell line — a false negative. We use the **supervised
contrastive (SupCon)** variant: *every* same-lineage cross-domain pair is a
positive.

Let `z^c_i` be L2-normalised CCLE embeddings and `z^t_j` be L2-normalised TCGA
embeddings in a batch, with lineage labels `y^c_i`, `y^t_j`.

**(1) Scaled cross-domain similarity matrix.** With learnable logit scale
`s = exp(log_tau)`:

```
S_ij = s · (z^c_i · z^t_j)                                     # (B_c × B_t)
```

**(2) Positive mask** (supervised — same lineage across the two domains):

```
P_ij = 1  if  y^c_i == y^t_j   else   0
```

**(3) Log-probability of each pair** (row-wise log-softmax over the TCGA gallery):

```
log q_ij = S_ij − logsumexp_j( S_ij )
```

**(4) Per-anchor SupCon loss** (average log-prob over that anchor's positives):

```
L^c→t_i = −(1 / |P_i|) · Σ_j  P_ij · log q_ij         (anchors with ≥1 positive)
```

**(5) Symmetric total** — do it in both directions (CCLE anchored on TCGA, and
TCGA anchored on CCLE, using `Sᵀ` and `Pᵀ`) and add:

```
L = mean_i L^c→t_i  +  mean_j L^t→c_j
```

Anchors with zero positives contribute nothing (`pos_counts.clamp(min=1)` guards the
division; a `valid` mask drops empty rows). Because the sampler guarantees every
lineage appears in both domains in every batch, every anchor has positives *and*
every negative is a genuine different-lineage pair.

> The two directions make the loss **symmetric**: `L ≈ 2 × L^c→t` when the
> similarity matrix is (near) symmetric. Practically it means neither tower is
> privileged as "the query side".

## 2. Learnable temperature

Temperature `τ` controls how sharply the softmax separates positives from
negatives. Rather than fix it, we learn it — but parameterise it in log-space for
stability, storing `log_tau = log(1/τ)` (the CLIP "logit scale"):

```python
self.log_tau = nn.Parameter(torch.tensor(math.log(1.0 / init_tau)))   # init_tau = 0.07
#  init value = log(1/0.07) = log(14.3) ≈ 2.66
tau = exp(-log_tau)          # the actual temperature, always > 0
logit_scale = exp(log_tau)   # multiplies the cosine similarities in eq. (1)
```

Learning `log_tau` (a) keeps `τ > 0` for free (no clamping needed), and (b) lets the
model tune its own confidence. Empirically `τ` stays stable (~0.07 → ~0.074 across
training) — no temperature collapse, which would signal the model cheating by
over-sharpening. `test_log_tau_receives_gradient` confirms the parameter is trained.

## 3. Batch construction — the stratified contrastive sampler

`StratifiedContrastiveBatchSampler` makes the "every negative is a true negative"
guarantee real. For `batch_size = 48` and 3 lineages × 2 domains:

```
per_lineage = 48 // (3 · 2) = 8
each batch  = 8 CCLE + 8 TCGA  for LUAD
            + 8 CCLE + 8 TCGA  for BRCA
            + 8 CCLE + 8 TCGA  for SKCM
            = 24 CCLE + 24 TCGA samples, all three lineages present
```

- **CCLE (small N)** is oversampled **with replacement** — there are far fewer cell
  lines than patients.
- **TCGA (large N)** is drawn **without replacement within an epoch** — no patient is
  reused in a single pass; a fresh shuffle each epoch
  (`test_sampler_reshuffles_between_epochs`).
- Every lineage present in both domains in every batch ⇒ the SupCon positive mask is
  never empty and every cross-lineage pair is a valid negative
  (`test_stratified_sampler_lineage_balance`).

## 4. Optimisation

From [`configs/training.yaml`](../configs/training.yaml):

| Setting | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | `3e-4` |
| Epochs | 30 (early-stopped) |
| Warmup epochs | 5 |
| Gradient clip (max L2 norm) | 1.0 |
| kNN validation `k` | 5 |
| Early-stop patience | 5 |

**Cosine LR schedule with linear warmup** (`ContrastiveTrainer._lr_lambda`): the LR
multiplier ramps linearly from ~0 to 1.0 over the first `warmup_epochs`, then
follows a half-cosine decay `0.5·(1 + cos(π·progress))` to ~0 by the final epoch.
Warmup avoids the early large-gradient instability of a cold Adam start; cosine
decay anneals into a flat minimum. Behaviour asserted by
`test_lr_warmup_then_cosine` (strictly increasing through warmup, `≈1.0` at its end,
`<0.1` near the finish).

**Gradient clipping** to L2 norm 1.0 caps the occasional large contrastive gradient.
The trainer also records per-tower grad norms so tower balance can be monitored (no
10× asymmetry was observed — see the Day 9 analysis).

## 5. Validation, checkpointing, early stopping

Each epoch, `KNNValidationCallback` embeds the val CCLE + val TCGA sets and computes
**val kNN@5 retrieval accuracy** — the same metric the Day 10 gate uses, so
validation tracks the real objective, not a proxy loss. Then:

- **Checkpoint** (`best_model.pt`) is saved whenever val kNN@5 improves, storing
  `model_state_dict`, `loss_state_dict` (so `τ` is restored too), `epoch`, and
  `val_knn_accuracy` (`test_checkpoint_saved_on_improvement`).
- **Early stopping** fires after `patience = 5` epochs with no improvement
  (`test_early_stopping_fires`). The full Day 8 run stopped at epoch 7/30 with the
  best model at epoch 2 (val kNN@5 0.9474).

MLflow logging is enabled only when a run name is supplied, so unit tests (and
`pctrans-train --no-mlflow`) run without writing an experiment.
