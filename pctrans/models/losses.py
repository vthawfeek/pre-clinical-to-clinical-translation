import math

import torch
import torch.nn as nn


class SupConInfoNCELoss(nn.Module):
    """SupCon-style multi-positive InfoNCE for cross-domain lineage alignment.

    Every same-lineage cross-domain pair is a positive; every different-lineage
    cross-domain pair is a true negative. This removes the false negatives that
    plague standard CLIP InfoNCE when a batch contains many samples of the same
    lineage.

    The temperature is learned via ``log_tau`` which stores ``log(1 / tau)`` (the
    CLIP-style logit scale), so ``exp(log_tau)`` is the positive multiplier
    applied to cosine similarities and ``tau`` is the actual temperature.
    """

    def __init__(self, init_tau=0.07):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(math.log(1.0 / init_tau)))

    @property
    def tau(self):
        """Actual temperature tau = 1 / exp(log_tau) = exp(-log_tau)."""
        return self.log_tau.neg().exp()

    @staticmethod
    def _directional_loss(sim, pos_mask):
        """Mean SupCon loss over anchors (rows) of a scaled similarity matrix."""
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos_counts = pos_mask.sum(dim=1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_counts.clamp(min=1.0)
        valid = pos_counts > 0
        if valid.any():
            return -mean_log_prob_pos[valid].mean()
        return sim.new_zeros(())

    def forward(self, z_ccle, z_tcga, lineage_labels_ccle, lineage_labels_tcga):
        labels_ccle = torch.as_tensor(lineage_labels_ccle, device=z_ccle.device)
        labels_tcga = torch.as_tensor(lineage_labels_tcga, device=z_tcga.device)

        pos_mask = (labels_ccle[:, None] == labels_tcga[None, :]).float()
        logit_scale = self.log_tau.exp()
        sim = (z_ccle @ z_tcga.t()) * logit_scale  # (B_c, B_t)

        loss_ccle = self._directional_loss(sim, pos_mask)
        loss_tcga = self._directional_loss(sim.t(), pos_mask.t())
        return loss_ccle + loss_tcga
