"""``pctrans-query`` CLI: nearest TCGA patients for a given CCLE cell line.

Thin wrapper over `pctrans.inference.TranslationEmbedder`: embed one cell line
with the frozen dual-tower model and print its k nearest TCGA patients (id,
lineage, distance) in the shared 64-d space.
"""

import typer

from pctrans.inference.api import TranslationEmbedder

app = typer.Typer()


@app.command()
def main(
    cell_line: str,
    model: str = "models/best_model.pt",
    data_dir: str = "data/processed/",
    model_config: str = "configs/model.yaml",
    k: int = 5,
):
    """Embed CELL_LINE (a CCLE ModelID) and print its k nearest TCGA patients."""
    embedder = TranslationEmbedder(model, data_dir, model_config)
    z = embedder.embed_cell_line(cell_line)
    typer.echo(f"{cell_line}: {z.shape[1]}-d embedding")

    neighbours = embedder.query_patients(cell_line, k=k)
    typer.echo(f"Top {len(neighbours)} nearest TCGA patients:")
    for rank, neighbour in enumerate(neighbours, start=1):
        typer.echo(
            f"  {rank}. {neighbour['patient_id']}  "
            f"lineage={neighbour['lineage']}  d={neighbour['distance']:.4f}"
        )

    return {
        "cell_line": cell_line,
        "embedding_dim": int(z.shape[1]),
        "neighbours": neighbours,
    }


if __name__ == "__main__":
    app()
