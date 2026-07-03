"""``pctrans-casestudy-analysis`` CLI: Day 23 placement/response-link + Day 26 drug-signal probe.

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

Day 26 adds a third question on the same cell lines: does the *embedding*
still carry drug-response signal at all? `drug_signal_retained` (within-CCLE
CV) brackets the 64-d embedding against raw HVG expression and BRAF status
alone, distinguishing "alignment discarded drug-response signal" from "the
proximity probe was the wrong readout." A lightweight, unvalidated ElasticNet
CCLE-to-patient reference (no ground-truth patient AUC exists to score it
against) is also reported as a descriptive bracket, not a CODE-AE
reproduction. Writes `reports/drug_transfer_positioning.json` and adds a
third panel to Figure F6.
"""

import json
import warnings
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
    drug_signal_retained,
)
from pctrans.evaluation.viz import (  # noqa: E402
    braf_casestudy_panel,
    braf_casestudy_panel_interactive,
    umap_projection,
)

app = typer.Typer()


def _ccle_to_patient_reference(cell_line_raw_expr, cell_line_auc, patient_raw_expr, seed=42):
    """Descriptive-only reference: ElasticNet(CCLE expr -> AUC) applied to patient expr.

    No ground-truth vemurafenib AUC exists for TCGA patients, so this cannot
    be scored -- it only reports the predicted-AUC distribution against the
    training range, as a proximity-free bracket for the Part-B null (Day 26
    task 2's optional, lightweight non-alignment reference; not a CODE-AE
    reproduction).
    """
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(cell_line_raw_expr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=50, max_iter=5000, random_state=seed)
        model.fit(scaler.transform(cell_line_raw_expr), cell_line_auc)
    predicted = model.predict(scaler.transform(patient_raw_expr))
    return {
        "n_train_cell_lines": int(cell_line_auc.size),
        "n_patients": int(predicted.size),
        "n_nonzero_coefs": int(np.count_nonzero(model.coef_)),
        "train_auc_min": float(cell_line_auc.min()),
        "train_auc_max": float(cell_line_auc.max()),
        "predicted_patient_auc_mean": float(predicted.mean()),
        "predicted_patient_auc_std": float(predicted.std()),
        "predicted_patient_auc_min": float(predicted.min()),
        "predicted_patient_auc_max": float(predicted.max()),
        "note": "Descriptive reference only -- no ground-truth patient AUC exists to validate against.",
    }


@app.command()
def main(
    table: str = "data/processed/braf_vemurafenib.parquet",
    output: str = "reports/braf_casestudy.json",
    figure_png: str = "reports/braf_vemurafenib.png",
    figure_html: str = "reports/braf_vemurafenib.html",
    ccle_raw_expr: str = "data/processed/ccle_2k.parquet",
    tcga_raw_expr: str = "data/processed/tcga_2k.parquet",
    drug_transfer_output: str = "reports/drug_transfer_positioning.json",
    n_boot: int = 2000,
    n_splits: int = 5,
    seed: int = 42,
):
    """Run the Day 23 BRAF placement/response-link + Day 26 drug-signal-retained analysis."""
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

    # -- Day 26: is drug-response signal even retained by the embedding? ---
    ccle_raw_df = pd.read_parquet(ccle_raw_expr)
    gene_cols = [c for c in ccle_raw_df.columns if c != "lineage"]
    cell_line_raw_expr = ccle_raw_df.loc[with_auc["sample_id"], gene_cols].to_numpy()

    probe = drug_signal_retained(
        np.stack(with_auc["embedding"].to_numpy()), cell_line_raw_expr,
        with_auc["BRAF_status"].to_numpy(), with_auc["vemurafenib_auc"].to_numpy(),
        n_splits=n_splits, seed=seed,
    )

    tcga_raw_df = pd.read_parquet(tcga_raw_expr)
    patient_raw_expr = tcga_raw_df.loc[patients["sample_id"], gene_cols].to_numpy()
    reference = _ccle_to_patient_reference(
        cell_line_raw_expr, with_auc["vemurafenib_auc"].to_numpy(), patient_raw_expr, seed=seed,
    )

    bar26 = "=" * 60
    typer.echo("")
    typer.echo(bar26)
    typer.echo("   DAY 26 -- DRUG-SIGNAL RETAINED (within-CCLE CV, n={})".format(probe["embedding"]["n"]))
    typer.echo(bar26)
    for name, label in (
        ("braf_status", "BRAF status alone"),
        ("raw_expression", "raw HVG expression"),
        ("embedding", "64-d embedding"),
    ):
        block = probe[name]
        typer.echo(
            f"  {label:<20} R^2={block['r2']:+.3f}  rho={block['rho']:+.3f}  p={block['p_value']:.3g}"
        )
    typer.echo(
        f"  CCLE-to-patient ElasticNet reference: predicted patient AUC "
        f"{reference['predicted_patient_auc_mean']:.3f} +/- {reference['predicted_patient_auc_std']:.3f} "
        f"(range [{reference['predicted_patient_auc_min']:.3f}, {reference['predicted_patient_auc_max']:.3f}]); "
        f"training range [{reference['train_auc_min']:.3f}, {reference['train_auc_max']:.3f}] "
        f"-- descriptive only, no ground truth"
    )
    typer.echo(bar26)

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
        drug_signal_result=probe,
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

    drug_transfer_summary = {
        "table": str(table),
        "n_cell_lines_with_auc": probe["embedding"]["n"],
        "drug_signal_retained": probe,
        "ccle_to_patient_reference": reference,
    }
    drug_transfer_path = Path(drug_transfer_output)
    drug_transfer_path.parent.mkdir(parents=True, exist_ok=True)
    with open(drug_transfer_path, "w", encoding="utf-8") as f:
        json.dump(drug_transfer_summary, f, indent=2)
    typer.echo(f"Wrote {drug_transfer_path}")

    return summary


if __name__ == "__main__":
    app()
