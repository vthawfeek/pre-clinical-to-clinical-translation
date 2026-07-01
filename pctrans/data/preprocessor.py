class FeatureSynchroniser:
    def load_ccle(self, raw_dir):
        raise NotImplementedError

    def load_tcga(self, raw_dir):
        raise NotImplementedError

    def find_common_genes(self, ccle_genes, tcga_genes):
        raise NotImplementedError

    def select_hvgs(self, ccle_expr, tcga_expr, common_genes, n_hvgs=2000):
        raise NotImplementedError

    def save_filtered(self, ccle_expr, tcga_expr, hvg_list, out_dir):
        raise NotImplementedError


class DataSplitter:
    def stratified_split(self, ccle_df, tcga_df, val_frac=0.15, test_frac=0.15, seed=42):
        raise NotImplementedError

    def fit_scalers(self, ccle_train_expr, tcga_train_expr):
        raise NotImplementedError

    def apply_scalers(self, expr_df, scaler):
        raise NotImplementedError

    def save_splits(self, splits, scalers, out_dir):
        raise NotImplementedError
