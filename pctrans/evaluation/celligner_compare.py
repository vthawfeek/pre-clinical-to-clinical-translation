"""Celligner head-to-head benchmark — Day 25.

Celligner (Warren et al., Nat. Commun. 2021) is the incumbent unsupervised
CCLE<->TCGA alignment method: contrastive PCA + mutual nearest neighbours,
using no lineage labels at all. This module runs it on the *same*
HVG-filtered CCLE+TCGA matrices our contrastive model is scored on, then
scores its aligned embedding with the identical cross-domain kNN@k +
silhouette metric, so the only variable between the two methods is the
alignment step itself.

`run_celligner` is gated behind an import guard and returns ``None`` when the
dependency chain is unavailable, the same None-safe pattern
`pctrans.evaluation.baselines` uses for ComBat/Scanorama (Day 17). For
`celligner` specifically this is not just "no C/C++ toolchain": its PyPI
release (1.1.0) declares a dependency on a package literally named ``umap``
(not ``umap-learn``), which has no installable release on PyPI, so `pip`/`uv`
cannot resolve it *at all*, on any platform, without hand-patching its
metadata or installing from GitHub source. Installing from source additionally
requires R plus a bundled `mnnpy` build with no prebuilt wheel. Report the gap
honestly (as Day 17 did for ComBat/Scanorama) rather than fabricating a number.
"""

import numpy as np

from pctrans.evaluation.knn import knn_accuracy_from_embeddings
from pctrans.evaluation.silhouette import cross_domain_silhouette


def run_celligner(ccle_expr, tcga_expr):
    """Fit Celligner on the CCLE reference and align TCGA onto it.

    ``ccle_expr``/``tcga_expr`` are ``(n_samples, n_genes)`` matrices of the
    same HVG-filtered log-expression features used elsewhere in this project.
    Returns the joint aligned embedding as a single ``(n_ccle + n_tcga, d)``
    array with CCLE rows first, then TCGA rows -- matching
    `retrieval_on_embedding`'s expected row order -- or ``None`` if
    `celligner` is not importable.
    """
    try:
        from celligner import Celligner
    except ImportError:
        return None

    model = Celligner()
    model.fit(ccle_expr)
    model.transform(tcga_expr)
    return np.asarray(model.combined_output)


def retrieval_on_embedding(
    joint_emb, ccle_ids, tcga_ids, lineages, k=5, idx_to_lineage=None, lineage_order=None
):
    """Score an externally-produced joint embedding with our own metric.

    ``joint_emb`` has CCLE rows first (length ``len(ccle_ids)``) then TCGA
    rows, matching `run_celligner`'s output order. ``lineages`` maps each
    sample ID (CCLE model ID or TCGA barcode) to its integer lineage label,
    using the same encoding as `pctrans.data.dataset`. Reuses
    `knn_accuracy_from_embeddings` (the exact function `pctrans-evaluate`
    scores our own model with) and `cross_domain_silhouette`, so a Celligner
    number and a contrastive number are directly comparable.
    """
    joint_emb = np.asarray(joint_emb, dtype=np.float64)
    n_ccle = len(ccle_ids)
    z_ccle, z_tcga = joint_emb[:n_ccle], joint_emb[n_ccle:]
    y_ccle = np.asarray([lineages[i] for i in ccle_ids])
    y_tcga = np.asarray([lineages[i] for i in tcga_ids])

    knn_result = knn_accuracy_from_embeddings(
        z_ccle,
        y_ccle,
        z_tcga,
        y_tcga,
        k=k,
        idx_to_lineage=idx_to_lineage,
        lineage_order=lineage_order,
    )
    pooled = np.concatenate([z_ccle, z_tcga], axis=0)
    pooled_labels = np.concatenate([y_ccle, y_tcga])
    silhouette = cross_domain_silhouette(pooled, pooled_labels)

    return {
        "overall_accuracy": knn_result["overall_accuracy"],
        "per_lineage": knn_result["per_lineage"],
        "k_table": knn_result["k_table"],
        "silhouette": silhouette,
    }
