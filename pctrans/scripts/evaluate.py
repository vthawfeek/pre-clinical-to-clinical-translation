import typer

app = typer.Typer()


@app.command()
def main(
    model: str = "models/best_model.pt",
    data_dir: str = "data/processed/",
):
    raise NotImplementedError


if __name__ == "__main__":
    app()
