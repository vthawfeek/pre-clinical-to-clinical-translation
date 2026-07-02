"""``pctrans-precompute`` CLI: Day 12 embedding + metadata precompute for the app.

The Streamlit demo must load *fast* and, once deployed to Streamlit Cloud, must not
depend on the model checkpoint or the multi-hundred-MB raw expression files (only
small, git-tracked artefacts under ``data/processed/`` ship with the app). This
script does the one-time heavy lifting offline:

  * ``data/processed/ccle_embeddings.npz`` — every CCLE cell line embedded with the
    frozen ``CCLEEncoder`` (z, lineage code, ModelID, display name). This is the
    task-2 deliverable: the full catalogue, so the app never runs the model.
  * ``data/processed/app_meta.json`` — deploy-safe metadata distilled from the raw
    files once: CCLE ModelID -> stripped display name (e.g. ``NCIH1975``), and each
    TCGA test patient -> {stage, histology}. Lets the app show real cell-line names
    and a clinical annotation column without shipping ``data/raw/``.

The app's UMAP / neighbours / TFS gauge run on the held-out *test* embeddings
(``embeddings_test.npz``, written Day 11) — the honest, evaluated subset that has
per-cell-line TFS scores from Gate 1. ``ccle_embeddings.npz`` is the fuller
catalogue kept for deployment and future expansion.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader

from pctrans.data.dataset import IDX_TO_LINEAGE, CCLEDataset
from pctrans.data.preprocessor import DataSplitter
from pctrans.evaluation.knn import embed_loader
from pctrans.scripts.evaluate import _build_model, _load_yaml

app = typer.Typer()


def _ccle_display_names(raw_ccle_dir):
    """ModelID -> stripped cell-line name from CCLE ``Model.csv`` (empty if absent)."""
    path = Path(raw_ccle_dir) / "Model.csv"
    if not path.exists():
        return {}
    meta = pd.read_csv(path, usecols=lambda c: c in {"ModelID", "StrippedCellLineName", "CellLineName"})
    names = {}
    for _, row in meta.iterrows():
        name = row.get("StrippedCellLineName")
        if not isinstance(name, str) or not name:
            name = row.get("CellLineName")
        names[str(row["ModelID"])] = str(name) if isinstance(name, str) and name else str(row["ModelID"])
    return names


def _tcga_clinical_meta(raw_tcga_dir, sample_ids):
    """TCGA sample -> {stage, histology} from the Xena phenotype table.

    Restricted to ``sample_ids`` (the test patients the app plots). Returns an empty
    dict if the phenotype file is absent so the app degrades to showing no annotation.
    """
    path = Path(raw_tcga_dir) / "Survival_SupplementalTable_S1_20171025_xena_sp.tsv"
    if not path.exists():
        return {}
    cols = ["sample", "ajcc_pathologic_tumor_stage", "histological_type"]
    pheno = pd.read_csv(path, sep="\t", usecols=lambda c: c in cols).set_index("sample")
    wanted = pheno.index.intersection(list(sample_ids))
    pheno = pheno.loc[wanted]

    def _clean(v):
        if not isinstance(v, str):
            return ""
        v = v.strip()
        return "" if v.lower() in {"", "nan", "[not available]", "[unknown]", "[discrepancy]"} else v

    meta = {}
    for sample_id, row in pheno.iterrows():
        meta[str(sample_id)] = {
            "stage": _clean(row.get("ajcc_pathologic_tumor_stage")),
            "histology": _clean(row.get("histological_type")),
        }
    return meta


@app.command()
def main(
    model: str = "models/best_model.pt",
    data_dir: str = "data/processed/",
    raw_dir: str = "data/raw/",
    model_config: str = "configs/model.yaml",
):
    """Precompute CCLE embeddings + deploy-safe app metadata."""
    data_dir = Path(data_dir)
    raw_dir = Path(raw_dir)
    model_cfg = _load_yaml(model_config)

    ccle_df = pd.read_parquet(data_dir / "ccle_2k.parquet")
    with open(data_dir / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    # Embed every CCLE cell line with the frozen model (z-scored input, no grad).
    scaled = DataSplitter().apply_scalers(ccle_df, scalers)
    ccle_all = CCLEDataset(scaled)
    net = _build_model(model_cfg)
    checkpoint = torch.load(model, map_location="cpu", weights_only=False)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    loader = DataLoader(ccle_all, batch_size=256, shuffle=False)
    z_ccle, y_ccle = embed_loader(net.encode_ccle, loader)
    ids_ccle = ccle_all.ids.astype(str)

    names_map = _ccle_display_names(raw_dir / "ccle")
    names = np.array([names_map.get(i, i) for i in ids_ccle])

    emb_path = data_dir / "ccle_embeddings.npz"
    np.savez(emb_path, z=z_ccle, y=y_ccle, ids=ids_ccle, names=names)
    typer.echo(f"Wrote {emb_path}  ({len(ids_ccle)} cell lines, dim {z_ccle.shape[1]})")

    # Deploy-safe metadata: cell-line names + TCGA test-patient clinical annotation.
    test_emb = data_dir / "embeddings_test.npz"
    tcga_ids = []
    if test_emb.exists():
        with np.load(test_emb, allow_pickle=True) as d:
            tcga_ids = [str(s) for s in d["ids_tcga"]]
    tcga_meta = _tcga_clinical_meta(raw_dir / "tcga", tcga_ids)

    meta = {
        "ccle_names": {i: names_map.get(i, i) for i in ids_ccle},
        "tcga_meta": tcga_meta,
    }
    meta_path = data_dir / "app_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    typer.echo(
        f"Wrote {meta_path}  ({len(meta['ccle_names'])} names, "
        f"{len(tcga_meta)} TCGA annotations)"
    )

    per_lineage = {IDX_TO_LINEAGE[i]: int((y_ccle == i).sum()) for i in IDX_TO_LINEAGE}
    typer.echo(f"CCLE cell lines per lineage: {per_lineage}")
    return {"n_ccle": len(ids_ccle), "n_tcga_meta": len(tcga_meta)}


if __name__ == "__main__":
    app()
