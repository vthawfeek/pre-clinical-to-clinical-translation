class TranslationEmbedder:
    def __init__(self, checkpoint_path):
        raise NotImplementedError

    def embed_cell_line(self, cell_line_id):
        raise NotImplementedError

    def query_patients(self, cell_line_id, k=5):
        raise NotImplementedError
