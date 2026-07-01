from typing import Optional

import typer

app = typer.Typer()


@app.command()
def main(
    config: str = "configs/training.yaml",
    data_dir: str = "data/processed/",
    epochs: Optional[int] = None,
):
    raise NotImplementedError


if __name__ == "__main__":
    app()
