import torch.nn as nn
import torch.nn.functional as F


class DualTowerModel(nn.Module):
    """Wraps the CCLE and TCGA encoders and projects onto the unit hypersphere.

    Both encoders emit raw 64-dim embeddings; this module L2-normalises them so
    that dot products become cosine similarities for the contrastive loss.
    """

    def __init__(self, ccle_encoder, tcga_encoder):
        super().__init__()
        self.ccle_encoder = ccle_encoder
        self.tcga_encoder = tcga_encoder

    def forward(self, x_ccle, x_tcga):
        z_ccle = self.encode_ccle(x_ccle)
        z_tcga = self.encode_tcga(x_tcga)
        return z_ccle, z_tcga

    def encode_ccle(self, x):
        return F.normalize(self.ccle_encoder(x), dim=-1)

    def encode_tcga(self, x):
        return F.normalize(self.tcga_encoder(x), dim=-1)
