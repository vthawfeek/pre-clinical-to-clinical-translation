import math

import torch
import torch.nn.functional as F

from pctrans.models.losses import SupConInfoNCELoss


def _one_hot_embeddings(labels, embed_dim=8):
    """Perfectly separated unit embeddings: lineage L sits on basis vector e_L."""
    z = torch.zeros(len(labels), embed_dim)
    for i, lab in enumerate(labels):
        z[i, lab] = 1.0
    return F.normalize(z, dim=-1)


def test_temperature_is_positive():
    loss_fn = SupConInfoNCELoss(init_tau=0.07)
    assert loss_fn.log_tau.exp().item() > 0


def test_tau_property_matches_init():
    loss_fn = SupConInfoNCELoss(init_tau=0.07)
    assert math.isclose(loss_fn.tau.item(), 0.07, rel_tol=1e-5)


def test_loss_is_positive_finite_scalar():
    loss_fn = SupConInfoNCELoss(init_tau=0.07)
    torch.manual_seed(0)
    z_ccle = F.normalize(torch.randn(6, 8), dim=-1)
    z_tcga = F.normalize(torch.randn(9, 8), dim=-1)
    labels_ccle = [0, 0, 1, 1, 2, 2]
    labels_tcga = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    loss = loss_fn(z_ccle, z_tcga, labels_ccle, labels_tcga)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_loss_decreases_on_correct_batch():
    loss_fn = SupConInfoNCELoss(init_tau=0.07)
    labels_ccle = [0, 1, 2]
    labels_tcga = [0, 1, 2]

    # Aligned: same-lineage cross-domain pairs are identical unit vectors.
    z_ccle = _one_hot_embeddings(labels_ccle)
    z_tcga = _one_hot_embeddings(labels_tcga)
    loss_aligned = loss_fn(z_ccle, z_tcga, labels_ccle, labels_tcga)

    # Misaligned: random embeddings ignore lineage structure.
    torch.manual_seed(1)
    z_ccle_rand = F.normalize(torch.randn(3, 8), dim=-1)
    z_tcga_rand = F.normalize(torch.randn(3, 8), dim=-1)
    loss_random = loss_fn(z_ccle_rand, z_tcga_rand, labels_ccle, labels_tcga)

    assert loss_aligned.item() < loss_random.item()
    assert loss_aligned.item() > 0


def test_log_tau_receives_gradient():
    loss_fn = SupConInfoNCELoss(init_tau=0.07)
    torch.manual_seed(2)
    z_ccle = F.normalize(torch.randn(6, 8), dim=-1)
    z_tcga = F.normalize(torch.randn(6, 8), dim=-1)
    labels = [0, 1, 2, 0, 1, 2]
    loss = loss_fn(z_ccle, z_tcga, labels, labels)
    loss.backward()
    assert loss_fn.log_tau.grad is not None
