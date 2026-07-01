import torch.nn as nn


class SupConInfoNCELoss(nn.Module):
    def __init__(self, init_tau=0.07):
        super().__init__()
        raise NotImplementedError

    def forward(self, z_ccle, z_tcga, lineage_labels_ccle, lineage_labels_tcga):
        raise NotImplementedError
