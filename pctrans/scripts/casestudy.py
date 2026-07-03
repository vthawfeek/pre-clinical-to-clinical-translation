"""``pctrans-casestudy`` CLI: Day 22 BRAF/vemurafenib data assembly.

Downloads the three real sources the case study needs (PRISM vemurafenib
sensitivity, CCLE BRAF calls, TCGA-SKCM BRAF calls -- see
`pctrans.casestudy.braf_vemurafenib` module docstring for why each source was
chosen), joins them onto the SKCM slice of the already-embedded 3-lineage
cell lines (`ccle_embeddings.npz`, Day 12 -- the full 259-line catalogue) and
TCGA test patients (`embeddings_test.npz`, Day 11), and writes
`data/processed/braf_vemurafenib.parquet` + a coverage summary.
"""

import json
from pathlib import Path

import typer

from pctrans.casestudy.braf_vemurafenib import (
    BRAF_ENTREZ_GENE_ID,
    CCLE_DEPMAP_ID_ATTRIBUTE,
    CCLE_MUTATION_PROFILE,
    CCLE_SAMPLE_LIST,
    CCLE_STUDY_ID,
    TCGA_SKCM_MUTATION_PROFILE,
    TCGA_SKCM_SAMPLE_LIST,
    CBioPortalClient,
    PrismClient,
    assemble_braf_table,
    coverage_summary,
)

app = typer.Typer()


@app.command()
def main(
    ccle_embeddings: str = "data/processed/ccle_embeddings.npz",
    tcga_embeddings: str = "data/processed/embeddings_test.npz",
    raw_dir: str = "data/raw/",
    output: str = "data/processed/braf_vemurafenib.parquet",
    coverage_output: str = "reports/braf_coverage.json",
    force: bool = False,
):
    """Assemble the SKCM BRAF-status + vemurafenib-sensitivity table."""
    raw_dir = Path(raw_dir)
    prism_dir = raw_dir / "prism"
    cbioportal_dir = raw_dir / "cbioportal"

    typer.echo("Downloading PRISM Repurposing 20Q2 secondary-screen dose response...")
    dose_response_path = PrismClient().download_dose_response(prism_dir, force=force)

    typer.echo("Downloading cBioPortal BRAF calls (CCLE + TCGA-SKCM)...")
    cbio = CBioPortalClient()
    ccle_mutations_path = cbio.download_mutations(
        CCLE_MUTATION_PROFILE, CCLE_SAMPLE_LIST, BRAF_ENTREZ_GENE_ID, cbioportal_dir, force=force
    )
    ccle_sequenced_path = cbio.download_sample_ids(CCLE_SAMPLE_LIST, cbioportal_dir, force=force)
    ccle_depmap_id_path = cbio.download_clinical_data(
        CCLE_STUDY_ID, CCLE_DEPMAP_ID_ATTRIBUTE, cbioportal_dir, force=force
    )
    tcga_mutations_path = cbio.download_mutations(
        TCGA_SKCM_MUTATION_PROFILE, TCGA_SKCM_SAMPLE_LIST, BRAF_ENTREZ_GENE_ID, cbioportal_dir, force=force
    )
    tcga_sequenced_path = cbio.download_sample_ids(TCGA_SKCM_SAMPLE_LIST, cbioportal_dir, force=force)

    table = assemble_braf_table(
        ccle_embeddings_path=ccle_embeddings,
        tcga_embeddings_path=tcga_embeddings,
        ccle_mutations_path=ccle_mutations_path,
        ccle_sequenced_ids_path=ccle_sequenced_path,
        ccle_depmap_id_map_path=ccle_depmap_id_path,
        tcga_mutations_path=tcga_mutations_path,
        tcga_sequenced_ids_path=tcga_sequenced_path,
        vemurafenib_path=dose_response_path,
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path)
    typer.echo(f"Wrote {out_path}  ({len(table)} rows)")

    summary = coverage_summary(table)
    bar = "=" * 52
    typer.echo("")
    typer.echo(bar)
    typer.echo("     DAY 22 -- BRAF / VEMURAFENIB DATA COVERAGE")
    typer.echo(bar)
    typer.echo(f"SKCM cell lines (BRAF status resolved): {summary['n_cell_lines']}")
    typer.echo(f"  BRAF split: {summary['cell_line_braf_split']}")
    typer.echo(f"  ...with a vemurafenib readout: {summary['n_cell_lines_with_vemurafenib']}")
    typer.echo(f"  BRAF split (vemurafenib subset): {summary['cell_line_with_vemurafenib_braf_split']}")
    typer.echo(f"SKCM patients (BRAF status resolved): {summary['n_patients']}")
    typer.echo(f"  BRAF split: {summary['patient_braf_split']}")
    typer.echo(bar)

    coverage_path = Path(coverage_output)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    with open(coverage_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    typer.echo(f"Wrote {coverage_path}")

    return summary


if __name__ == "__main__":
    app()
