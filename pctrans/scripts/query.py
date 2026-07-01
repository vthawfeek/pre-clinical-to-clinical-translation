import typer

app = typer.Typer()


@app.command()
def main(
    cell_line: str,
    model: str = "models/best_model.pt",
):
    raise NotImplementedError


if __name__ == "__main__":
    app()
