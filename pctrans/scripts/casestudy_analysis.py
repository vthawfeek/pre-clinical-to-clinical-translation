"""``pctrans-casestudy-analysis`` CLI: Day 23 placement + response-link analysis.

Loads the Day 22 assembled table (`data/processed/braf_vemurafenib.parquet`)
and runs the two-part Rung-4 hypothesis test:

- **Part A -- placement:** are BRAF-mutant SKCM cell lines embedded nearer the
  BRAF-mutant SKCM patient centroid than BRAF-WT lines (Mann-Whitney + a
  bootstrapped common-language effect size)?
- **Part B -- response link:** does that same proximity correlate with
  vemurafenib sensitivity among cell lines with a PRISM AUC readout
  (Spearman + bootstrap CI)?

Writes `reports/braf_casestudy.json` and the case-study figure pair
`reports/braf_vemurafenib.png` / `reports/braf_vemurafenib.html`.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import typer

matplotlib.use("Agg")  # headless PNG rendering, no display required

from pctrans.casestudy.braf_vemurafenib import (  # noqa: E402
    braf_mutant_patient_centroid,
    braf_placement_test,
    braf_response_link,
    distance_to_centroid,
)
from pctrans.evaluation.viz import (  # noqa: E402
    braf_casestudy_panel,
    braf_casestudy_panel_interactive,
    umap_projection,
)

app = typer.Typer()


@app.command()
def main(
    table: str = "data/processed/braf_vemurafenib.parquet",
    output: str = "reports/braf_casestudy.json",
    figure_png: str = "reports/braf_vemurafenib.png",
    figure_html: str = "reports/braf_vemurafenib.html",
    n_boot: int = 2000,
    seed: int = 42,
):
    """Run the Day 23 BRAF placement + vemurafenib response-link analysis."""
    df = pd.read_parquet(table)
    cell_lines = df[df["domain"] == "cell_line"].reset_index(drop=True)
    patients = df[df["domain"] == "patient"].reset_index(drop=True)
    typer.echo(f"Loaded {table}: {len(cell_lines)} cell lines + {len(patients)} patients (SKCM)")

    cell_line_embeddings = np.stack(cell_lines["embedding"].to_numpy())
    patient_embeddings = np.stack(patients["embedding"].to_numpy())

    placement = braf_placement_test(
        cell_line_embeddings, cell_lines["BRAF_status"].to_numpy(),
        patient_embeddings, patients["BRAF_status"].to_numpy(),
        n_boot=n_boot, seed=seed,
    )

    with_auc = cell_lines[cell_lines["vemurafenib_auc"].notna()].reset_index(drop=True)
    response = braf_response_link(
        np.stack(with_auc["embedding"].to_numpy()), with_auc["BRAF_status"].to_numpy(),
        with_auc["vemurafenib_auc"].to_numpy(), patient_embeddings, patients["BRAF_status"].to_numpy(),
        n_boot=n_boot, seed=seed,
    )

    bar = "=" * 60
    typer.echo("")
    typer.echo(bar)
    typer.echo("   DAY 23 -- BRAF PLACEMENT + VEMURAFENIB RESPONSE LINK")
    typer.echo(bar)
    typer.echo(
        f"Part A: median dist mutant={placement['median_distance_mutant']:.3f} "
        f"vs WT={placement['median_distance_wt']:.3f}  "
        f"(n_mut={placement['n_mutant']}, n_wt={placement['n_wt']})"
    )
    typer.echo(
        f"        Mann-Whitney p={placement['p_value']:.4g}, "
        f"effect size={placement['effect_size']:.3f} "
        f"(95% CI [{placement['effect_size_ci_low']:.3f}, {placement['effect_size_ci_high']:.3f}])"
    )
    typer.echo(
        f"Part B: Spearman rho(proximity, vemurafenib AUC) = {response['rho']:.3f} "
        f"(95% CI [{response['ci_low']:.3f}, {response['ci_high']:.3f}]); "
        f"p={response['p_value']:.4g}; n={response['n']} "
        f"({response['n_mutant']} mutant / {response['n_wt']} WT)"
    )
    typer.echo(bar)

    # -- Figures ---------------------------------------------------------
    pooled_embeddings = np.concatenate([cell_line_embeddings, patient_embeddings], axis=0)
    pooled_status = np.concatenate([cell_lines["BRAF_status"].to_numpy(), patients["BRAF_status"].to_numpy()])
    pooled_domain = np.array([0] * len(cell_lines) + [1] * len(patients))
    pooled_ids = np.concatenate([cell_lines["sample_id"].to_numpy(), patients["sample_id"].to_numpy()])
    coords = umap_projection(pooled_embeddings, seed=seed)

    centroid = braf_mutant_patient_centroid(patient_embeddings, patients["BRAF_status"].to_numpy())
    proximity = -distance_to_centroid(np.stack(with_auc["embedding"].to_numpy()), centroid)
    auc = with_auc["vemurafenib_auc"].to_numpy()
    response_status = with_auc["BRAF_status"].to_numpy()
    response_ids = with_auc["sample_id"].to_numpy()

    static_fig = braf_casestudy_panel(
        coords, pooled_status, pooled_domain, proximity, auc, response_status, placement, response,
    )
    png_path = Path(figure_png)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    static_fig.savefig(png_path, dpi=150, bbox_inches="tight")
    typer.echo(f"Wrote {png_path}")

    interactive_fig = braf_casestudy_panel_interactive(
        coords, pooled_status, pooled_domain, pooled_ids, proximity, auc, response_status, response_ids,
    )
    html_path = Path(figure_html)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    interactive_fig.write_html(str(html_path), include_plotlyjs="cdn")
    typer.echo(f"Wrote {html_path}")

    summary = {"table": str(table), "placement": placement, "response_link": response}
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    typer.echo(f"Wrote {out_path}")
    return summary


if __name__ == "__main__":
    app()
