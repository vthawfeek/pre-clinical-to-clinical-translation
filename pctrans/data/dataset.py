from torch.utils.data import Dataset

LINEAGE_TO_IDX = {"LUAD": 0, "BRCA": 1, "SKCM": 2}


class CCLEDataset(Dataset):
    def __init__(self, expr_df, lineage_col):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TCGADataset(Dataset):
    def __init__(self, expr_df, lineage_col):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
