class TCGAClient:
    def download_expression(self, out_dir, force=False):
        raise NotImplementedError

    def download_phenotype(self, out_dir, force=False):
        raise NotImplementedError


def filter_tcga_lineages(df_pheno, lineages):
    raise NotImplementedError
