class StratifiedContrastiveBatchSampler:
    def __init__(self, ccle_dataset, tcga_dataset, batch_size=48):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
