import typer

app = typer.Typer()


@app.command()
def main(
    raw_dir: str = "data/raw/",
    out_dir: str = "data/processed/",
    n_hvgs: int = 2000,
    split: bool = False,
):
    raise NotImplementedError


if __name__ == "__main__":
    app()
