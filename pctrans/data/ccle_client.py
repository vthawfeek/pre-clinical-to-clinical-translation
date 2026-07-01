class CCLEClient:
    def download_expression(self, out_dir, force=False):
        raise NotImplementedError

    def download_metadata(self, out_dir, force=False):
        raise NotImplementedError


def filter_lineages(df_meta, lineages):
    raise NotImplementedError
