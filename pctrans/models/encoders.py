import torch.nn as nn


class MLPBlock(nn.Module):
    """Linear -> BatchNorm1d -> ReLU -> Dropout.

    BatchNorm is placed before the activation so the scale of inter-layer
    activations stays stable when CCLE (small N) and TCGA (large N) batches with
    different expression distributions flow through the same tower.
    """

    def __init__(self, in_features, out_features, dropout, use_bn=True):
        super().__init__()
        layers = [nn.Linear(in_features, out_features)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _TowerEncoder(nn.Module):
    """Shared MLP template: stacked MLPBlocks + linear projection head.

    The final projection layer has no BatchNorm/activation/dropout so the raw
    embedding is returned unbounded; L2 normalisation happens in DualTowerModel.
    The last hidden block uses ``dropout_low`` (per the architecture spec), all
    earlier blocks use ``dropout``.
    """

    def __init__(
        self,
        input_dim=2000,
        hidden_dims=(1024, 512, 256, 128),
        embed_dim=64,
        dropout=0.3,
        dropout_low=0.2,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims)
        last = len(hidden_dims) - 1
        blocks = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            block_dropout = dropout_low if i == last else dropout
            blocks.append(MLPBlock(prev, h, dropout=block_dropout, use_bn=True))
            prev = h
        self.blocks = nn.Sequential(*blocks)
        self.projection = nn.Linear(prev, embed_dim)

    def forward(self, x):
        return self.projection(self.blocks(x))


class CCLEEncoder(_TowerEncoder):
    """Encoder tower for CCLE cell-line expression vectors."""


class TCGAEncoder(_TowerEncoder):
    """Encoder tower for TCGA patient expression vectors.

    Architecturally identical to CCLEEncoder but a separate nn.Module with its
    own weights, so the two domains are aligned rather than shared.
    """
