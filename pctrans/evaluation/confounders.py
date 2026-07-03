"""Tumour-purity confounder analysis — Day 20.

Rules out the biggest alternative explanation for the Day 10/19 alignment
result: that the model has learned "pure cultured cell vs. stroma/immune-
contaminated tumour", not cancer-lineage identity. Purity comes from TCGA's
ABSOLUTE consensus calls (`pctrans.data.tcga_client.download_purity`); CCLE
cell lines are grown as monocultures and are assigned purity ~= 1.0.

Three analyses (`PLAN-phase2.md` Day 20):

- `domain_axis_purity_correlation` -- does the residual CCLE->TCGA direction
  track purity? Informative, not fatal: a domain axis IS partly a purity axis
  by construction (cell lines have no stroma), so some correlation is
  expected. What matters is (b)/(c) below.
- `purity_stratified_knn` -- cross-domain kNN@k recomputed separately within
  TCGA high- and low-purity halves. Lineage retrieval must hold in both, or
  the model is leaning on purity rather than lineage biology.
- `residualise_purity` / `purity_residualised_silhouette` -- regress purity
  out of the pooled embeddings and recompute the lineage silhouette. Lineage
  cohesion should survive the ablation.
"""

import numpy as np
import pandas as pd

from pctrans.evaluation.knn import knn_accuracy_from_embeddings
from pctrans.evaluation.silhouette import cross_domain_silhouette

CCLE_PURITY = 1.0  # pure monoculture: no stromal/immune contamination


def load_purity(purity_table_path, ids_ccle, ids_tcga):
    """Join ABSOLUTE purity onto pooled CCLE + TCGA sample IDs.

    ``purity_table_path`` is the raw ABSOLUTE mastercalls table (tab-separated,
    ``array`` = sample barcode, ``purity`` in [0, 1]). Every CCLE id gets
    `CCLE_PURITY` (cell lines have no ABSOLUTE call -- there is no tumour
    stroma to estimate). TCGA ids absent from the table (no ABSOLUTE call for
    that sample) get ``NaN``; downstream analyses drop non-finite purity
    rather than imputing it.
    """
    table = pd.read_csv(purity_table_path, sep="\t")
    purity_by_barcode = table.set_index("array")["purity"].to_dict()

    purity_ccle = np.full(len(ids_ccle), CCLE_PURITY, dtype=np.float64)
    purity_tcga = np.array(
        [purity_by_barcode.get(str(i), np.nan) for i in ids_tcga], dtype=np.float64
    )
    return purity_ccle, purity_tcga


def domain_axis_purity_correlation(z_ccle, z_tcga, purity_ccle, purity_tcga):
    """Pearson r between each sample's domain-axis projection and its purity.

    The domain axis is the (unit) direction from the CCLE centroid to the TCGA
    centroid. Samples with non-finite purity are excluded. Returns
    ``{"r": float, "n": int}``; ``r`` is context (not pass/fail) -- see module
    docstring.
    """
    z_ccle = np.asarray(z_ccle, dtype=np.float64)
    z_tcga = np.asarray(z_tcga, dtype=np.float64)
    axis = z_tcga.mean(axis=0) - z_ccle.mean(axis=0)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return {"r": 0.0, "n": 0}
    axis = axis / norm

    z_pooled = np.concatenate([z_ccle, z_tcga], axis=0)
    purity_pooled = np.concatenate(
        [np.asarray(purity_ccle, dtype=np.float64), np.asarray(purity_tcga, dtype=np.float64)]
    )
    proj = z_pooled @ axis

    mask = np.isfinite(purity_pooled)
    if mask.sum() < 2 or np.std(proj[mask]) == 0 or np.std(purity_pooled[mask]) == 0:
        return {"r": 0.0, "n": int(mask.sum())}
    r = float(np.corrcoef(proj[mask], purity_pooled[mask])[0, 1])
    return {"r": r, "n": int(mask.sum())}


def purity_stratified_knn(
    z_ccle, y_ccle, z_tcga, y_tcga, purity_tcga, k=5, idx_to_lineage=None, lineage_order=None
):
    """Cross-domain kNN@k recomputed within TCGA high-/low-purity halves.

    TCGA test patients are split at the median of their (finite) purity into
    two strata; the CCLE anchor set (query side) is unchanged in both --
    every cell line is ~pure by definition, so there is no "low-purity cell
    line" stratum to compare against. Patients with no ABSOLUTE call (NaN
    purity) are dropped from both strata. Returns
    ``{"high_purity": {...}, "low_purity": {...}}``, each with
    ``overall_accuracy`` (``None`` if the stratum has fewer than ``k``
    patients) and ``n``.
    """
    purity_tcga = np.asarray(purity_tcga, dtype=np.float64)
    mask_finite = np.isfinite(purity_tcga)
    median = np.median(purity_tcga[mask_finite]) if mask_finite.any() else np.nan

    high_mask = mask_finite & (purity_tcga >= median)
    low_mask = mask_finite & (purity_tcga < median)

    results = {}
    for name, mask in (("high_purity", high_mask), ("low_purity", low_mask)):
        n = int(mask.sum())
        if n < k:
            results[name] = {"overall_accuracy": None, "n": n}
            continue
        res = knn_accuracy_from_embeddings(
            z_ccle,
            y_ccle,
            np.asarray(z_tcga)[mask],
            np.asarray(y_tcga)[mask],
            k=k,
            idx_to_lineage=idx_to_lineage,
            lineage_order=lineage_order,
        )
        results[name] = {"overall_accuracy": res["overall_accuracy"], "n": n}
    return results


def residualise_purity(embeddings, purity):
    """Regress purity out of each embedding dimension; return the residuals.

    Fits an ordinary-least-squares line (per dimension) against purity using
    only samples with finite purity, then subtracts the fitted trend from
    those samples. Samples with non-finite purity pass through unchanged
    (there is nothing to residualise against). Same shape as ``embeddings``.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    purity = np.asarray(purity, dtype=np.float64)
    residual = embeddings.copy()

    mask = np.isfinite(purity)
    if mask.sum() < 2:
        return residual

    x = purity[mask]
    design = np.column_stack([np.ones_like(x), x])
    coefs, *_ = np.linalg.lstsq(design, embeddings[mask], rcond=None)  # (2, d)
    fitted = design @ coefs
    residual[mask] = embeddings[mask] - fitted
    return residual


def purity_residualised_silhouette(z_ccle, y_ccle, z_tcga, y_tcga, purity_ccle, purity_tcga):
    """Cross-domain lineage silhouette after regressing purity out of pooled embeddings."""
    z_pooled = np.concatenate([np.asarray(z_ccle), np.asarray(z_tcga)], axis=0)
    y_pooled = np.concatenate([np.asarray(y_ccle), np.asarray(y_tcga)])
    purity_pooled = np.concatenate(
        [np.asarray(purity_ccle, dtype=np.float64), np.asarray(purity_tcga, dtype=np.float64)]
    )
    residual = residualise_purity(z_pooled, purity_pooled)
    return cross_domain_silhouette(residual, y_pooled)
