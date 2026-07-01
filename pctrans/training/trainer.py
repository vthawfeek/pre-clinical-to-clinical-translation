class ContrastiveTrainer:
    def __init__(self, model, loss_fn, train_sampler, val_ccle, val_tcga, config, mlflow_run_name=None):
        raise NotImplementedError

    def train(self, n_epochs):
        raise NotImplementedError

    def load_checkpoint(self, path):
        raise NotImplementedError
