"""Stratified contrastive batch sampler for dual-tower InfoNCE training.

Each yielded batch is lineage-balanced across both domains: for every lineage
it draws `per_lineage` CCLE samples and `per_lineage` TCGA samples, so a
default `batch_size=48` gives 8 CCLE + 8 TCGA per lineage across LUAD/BRCA/SKCM
(24 CCLE + 24 TCGA). This construction guarantees every negative in the SupCon
loss is a true between-lineage negative (see PLAN.md's Loss Function section).

CCLE (small N) is oversampled with replacement; TCGA (large N) is shuffled and
drawn without replacement within an epoch, so no TCGA sample is reused inside a
single pass. Each batch is emitted as ``{"ccle_indices": [...], "tcga_indices":
[...]}`` — positional indices into the respective datasets.
"""

import numpy as np

N_DOMAINS = 2  # CCLE + TCGA share the per-lineage budget within a batch.


class StratifiedContrastiveBatchSampler:
    def __init__(self, ccle_dataset, tcga_dataset, batch_size=48, seed=42):
        self.ccle_dataset = ccle_dataset
        self.tcga_dataset = tcga_dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        self.lineages = sorted(set(ccle_dataset.labels) & set(tcga_dataset.labels))
        if not self.lineages:
            raise ValueError("CCLE and TCGA share no lineage labels")

        self.per_lineage = max(1, batch_size // (len(self.lineages) * N_DOMAINS))
        self._ccle_by_lineage = self._group(ccle_dataset.labels)
        self._tcga_by_lineage = self._group(tcga_dataset.labels)

        for lineage in self.lineages:
            if self._ccle_by_lineage[lineage].size == 0:
                raise ValueError(f"no CCLE samples for lineage {lineage}")
            if self._tcga_by_lineage[lineage].size == 0:
                raise ValueError(f"no TCGA samples for lineage {lineage}")

        # An epoch is bounded by TCGA (no replacement): the lineage whose pool
        # yields the fewest full per_lineage draws sets the batch count.
        self._n_batches = min(
            self._tcga_by_lineage[lineage].size // self.per_lineage
            for lineage in self.lineages
        )
        self._n_batches = max(1, self._n_batches)

    def _group(self, labels):
        return {
            lineage: np.flatnonzero(labels == lineage) for lineage in self.lineages
        }

    def __len__(self):
        return self._n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        # TCGA: one shuffled deck per lineage, sliced without replacement.
        tcga_decks = {
            lineage: rng.permutation(self._tcga_by_lineage[lineage])
            for lineage in self.lineages
        }
        for b in range(self._n_batches):
            ccle_indices, tcga_indices = [], []
            start = b * self.per_lineage
            for lineage in self.lineages:
                tcga_indices.extend(
                    tcga_decks[lineage][start : start + self.per_lineage].tolist()
                )
                # CCLE: oversample with replacement (N_CCLE << N_TCGA).
                ccle_indices.extend(
                    rng.choice(
                        self._ccle_by_lineage[lineage],
                        size=self.per_lineage,
                        replace=True,
                    ).tolist()
                )
            yield {"ccle_indices": ccle_indices, "tcga_indices": tcga_indices}
        self.epoch += 1
