class KNNValidationCallback:
    def __call__(self, model, val_ccle_loader, val_tcga_loader, k=5):
        raise NotImplementedError
