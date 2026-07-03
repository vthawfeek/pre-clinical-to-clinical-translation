"""Real batch-correction baselines + supervised cross-domain ceiling — Day 17.

Phase 1's Gate-1 report cited "~63% (literature)" for Harmony because no batch-
correction library was actually wired into the pipeline. This module computes
Harmony, ComBat, and Scanorama cross-domain kNN@k on the *same* scaled test
features the contrastive model is scored on, plus a supervised ceiling: how
much lineage signal a plain classifier recovers with no cross-domain alignment
at all. If the contrastive model only matches that ceiling, the alignment step
isn't the thing doing the work.

`harmony_knn` always runs for real (`harmonypy` is a pure-Python/PyTorch
dependency with no native build step). `combat_knn` and `scanorama_knn` are
gated behind an import guard: `inmoose` and `scanorama` ship no prebuilt wheels
and require a C/C++ toolchain to build from source, which is not available on
every machine (this project's Windows dev box included). Both return ``None``
when their dependency is absent so callers can report the gap honestly instead
of a fabricated number; install the `baselines` extra (works out of the box on
Colab/Linux) to compute them for real.
"""

import numpy as np
import pandas as pd

from pctrans.evaluation.knn import knn_accuracy_from_embeddings

RANDOM_BASELINE = 1.0 / 3.0


def _pooled_arrays(ccle_ds, tcga_ds):
    x_ccle = ccle_ds.features.numpy()
    x_tcga = tcga_ds.features.numpy()
    pooled = np.concatenate([x_ccle, x_tcga], axis=0)
    return x_ccle, x_tcga, pooled


def pca_knn(
    ccle_ds, tcga_ds, n_components=50, k=5, seed=42, idx_to_lineage=None, lineage_order=None
):
    """No-alignment baseline: PCA on pooled raw features, then cross-domain kNN.

    ``idx_to_lineage``/``lineage_order`` default to the Phase-1 3-lineage module
    constants (see ``pctrans.evaluation.knn``); pass the Day 18 lineage-map output
    to score this baseline on an arbitrary-size lineage set.
    """
    from sklearn.decomposition import PCA

    x_ccle, x_tcga, pooled = _pooled_arrays(ccle_ds, tcga_ds)
    n_comp = min(n_components, pooled.shape[0], pooled.shape[1])
    coords = PCA(n_components=n_comp, random_state=seed).fit_transform(pooled)
    z_ccle, z_tcga = coords[: len(x_ccle)], coords[len(x_ccle):]
    res = knn_accuracy_from_embeddings(
        z_ccle,
        ccle_ds.labels,
        z_tcga,
        tcga_ds.labels,
        k=k,
        idx_to_lineage=idx_to_lineage,
        lineage_order=lineage_order,
    )
    return res["overall_accuracy"]


def harmony_knn(ccle_ds, tcga_ds, k=5):
    """Harmony batch-integration (domain = batch), then cross-domain kNN@k.

    Returns ``None`` if `harmonypy` is not importable.
    """
    try:
        import harmonypy
    except ImportError:
        return None

    x_ccle, x_tcga, pooled = _pooled_arrays(ccle_ds, tcga_ds)
    meta = pd.DataFrame({"domain": ["ccle"] * len(x_ccle) + ["tcga"] * len(x_tcga)})
    n_total = len(x_ccle) + len(x_tcga)
    # harmonypy's default nclust (round(N/30)) collapses to 1 on small test sets
    # and crashes inside the library (sigma stays a bare float); force >= 2.
    nclust = max(2, min(round(n_total / 30), 100))
    ho = harmonypy.run_harmony(pooled, meta, ["domain"], nclust=nclust, verbose=False)
    corrected = np.asarray(ho.Z_corr)
    if corrected.shape[0] != n_total:
        corrected = corrected.T
    z_ccle, z_tcga = corrected[: len(x_ccle)], corrected[len(x_ccle):]
    res = knn_accuracy_from_embeddings(z_ccle, ccle_ds.labels, z_tcga, tcga_ds.labels, k=k)
    return res["overall_accuracy"]


def combat_knn(ccle_ds, tcga_ds, k=5):
    """ComBat batch-correction (domain = batch), then cross-domain kNN@k.

    Returns ``None`` if `inmoose` is not importable (optional `baselines`
    extra; its build has no prebuilt wheel and needs a C/C++ toolchain).
    """
    try:
        from inmoose.pycombat import pycombat_norm
    except ImportError:
        return None

    x_ccle, x_tcga, pooled = _pooled_arrays(ccle_ds, tcga_ds)
    batch = ["ccle"] * len(x_ccle) + ["tcga"] * len(x_tcga)
    # inmoose follows the sva::ComBat convention: rows = genes, columns = samples.
    corrected = pycombat_norm(pd.DataFrame(pooled.T), batch).to_numpy().T
    z_ccle, z_tcga = corrected[: len(x_ccle)], corrected[len(x_ccle):]
    res = knn_accuracy_from_embeddings(z_ccle, ccle_ds.labels, z_tcga, tcga_ds.labels, k=k)
    return res["overall_accuracy"]


def scanorama_knn(ccle_ds, tcga_ds, k=5, dimred=50):
    """Scanorama integration (two batches = domains), then cross-domain kNN@k.

    Returns ``None`` if `scanorama` is not importable (optional `baselines`
    extra; its `annoy` dependency has no prebuilt wheel and needs a C++
    toolchain).
    """
    try:
        import scanorama
    except ImportError:
        return None

    x_ccle, x_tcga, _ = _pooled_arrays(ccle_ds, tcga_ds)
    gene_names = [f"g{i}" for i in range(x_ccle.shape[1])]
    n_comp = min(dimred, x_ccle.shape[1] - 1)
    dimreds, _, _ = scanorama.correct(
        [x_ccle, x_tcga], [gene_names, gene_names], return_dimred=True, dimred=n_comp
    )
    z_ccle, z_tcga = np.asarray(dimreds[0]), np.asarray(dimreds[1])
    res = knn_accuracy_from_embeddings(z_ccle, ccle_ds.labels, z_tcga, tcga_ds.labels, k=k)
    return res["overall_accuracy"]


def supervised_ceiling(ccle_train_ds, tcga_test_ds, seed=0, max_iter=2000):
    """Logistic-regression lineage classifier: train on CCLE, test cross-domain on TCGA.

    The "how easy is the problem, really" ceiling — lineage signal trivially
    recoverable with no cross-domain alignment at all. If the contrastive model
    only matches this, alignment isn't adding anything beyond raw class signal.
    """
    from sklearn.linear_model import LogisticRegression

    x_train = ccle_train_ds.features.numpy()
    y_train = ccle_train_ds.labels
    x_test = tcga_test_ds.features.numpy()
    y_test = tcga_test_ds.labels

    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    return float((preds == y_test).mean())
