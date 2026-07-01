import torch.nn as nn


class DualTowerModel(nn.Module):
    def __init__(self, ccle_encoder, tcga_encoder):
        super().__init__()
        raise NotImplementedError

    def forward(self, x_ccle, x_tcga):
        raise NotImplementedError

    def encode_ccle(self, x):
        raise NotImplementedError

    def encode_tcga(self, x):
        raise NotImplementedError
