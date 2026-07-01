import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout, use_bn=True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class CCLEEncoder(nn.Module):
    def __init__(self, input_dim=2000, hidden_dims=(1024, 512, 256, 128), embed_dim=64, dropout=0.3):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class TCGAEncoder(nn.Module):
    def __init__(self, input_dim=2000, hidden_dims=(1024, 512, 256, 128), embed_dim=64, dropout=0.3):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
