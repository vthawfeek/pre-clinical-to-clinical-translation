import typer

app = typer.Typer()


@app.command()
def ccle(out_dir: str = "data/raw/ccle/"):
    raise NotImplementedError


@app.command()
def tcga(out_dir: str = "data/raw/tcga/"):
    raise NotImplementedError


if __name__ == "__main__":
    app()
