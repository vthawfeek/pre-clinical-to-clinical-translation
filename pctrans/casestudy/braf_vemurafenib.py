"""BRAF/vemurafenib case-study data assembly + placement/response analysis.

Day 22 assembles the data (below); Day 23 adds the two-part hypothesis test
that makes the embedding biologically meaningful, using vemurafenib's
established BRAF-V600 dependence as a positive control:

- **Part A -- placement** (`braf_placement_test`): does a BRAF-mutant SKCM
  *cell line* embed nearer the BRAF-mutant SKCM *patient* centroid than a
  BRAF-WT line does? This asks whether the model captured a within-melanoma,
  driver-defined structure -- not just coarse lineage.
- **Part B -- response link** (`braf_response_link`): among SKCM cell lines,
  does proximity to that same patient centroid correlate with vemurafenib
  sensitivity? A positive result ties embedding geometry to a real drug
  response; a null result is a legitimate, reportable finding about the
  model's current resolution, not a failure to hide.

Ties the 3-lineage model to one real, textbook translational fact: vemurafenib
(PLX4032) is only active against BRAF-V600-mutant melanoma. Assembling that
link needs three real-world sources:

- **CCLE vemurafenib sensitivity** -- the classic CCLE compound panel
  (Barretina et al. 2012) never screened vemurafenib itself, only its
  Plexxikon tool-compound precursor PLX4720 (same chemical series, same
  BRAF-V600E target). DepMap's PRISM Repurposing secondary screen (Corsello
  et al. 2020) *does* test the clinical compound by name, so that is the
  source used here (``load_vemurafenib_sensitivity``), keyed by DepMap
  ``ACH-`` id with AUC as the primary metric (PRISM's IC50 column is >85%
  missing -- many lines never reach 50% viability loss in the tested dose
  range -- so AUC, always fit, is the more complete summary; lower AUC = more
  sensitive).
- **CCLE + TCGA-SKCM BRAF status** -- rather than DepMap's full
  ``OmicsSomaticMutations.csv`` (multi-GB, every gene x every line) or the
  pan-cancer MC3 public MAF (~28 GB unpacked), both far larger than a
  single-gene lookup needs, this pulls just the BRAF calls via cBioPortal's
  public REST API against ``ccle_broad_2019`` (CCLE) and
  ``skcm_tcga_pan_can_atlas_2018`` (TCGA SKCM) -- the same MAF-derived calls,
  a few hundred KB instead of gigabytes.

``CBioPortalClient`` mirrors this project's existing download-client
convention (``CCLEClient``/``TCGAClient``): each method caches one JSON
response under ``out_dir`` and skips re-fetching unless ``force=True``.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import mannwhitneyu, spearmanr
from tqdm import tqdm

CBIOPORTAL_API = "https://www.cbioportal.org/api"

# DepMap PRISM Repurposing 20Q2 secondary screen (Corsello et al. 2020),
# hosted on Figshare -- the dataset that actually screened vemurafenib by
# name (the classic CCLE compound panel only has its precursor, PLX4720).
PRISM_DOSE_RESPONSE_URL = "https://ndownloader.figshare.com/files/36794595"
PRISM_DOSE_RESPONSE_FILENAME = "prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv"
PRISM_DOSE_RESPONSE_MIN_BYTES = 200_000_000  # real file is ~290 MB

# PRISM Repurposing 20Q2 secondary screen: two broad_id batches (different
# plated dose ranges) both correspond to vemurafenib.
VEMURAFENIB_BROAD_IDS = ("BRD-K56343971-001-14-8", "BRD-K56343971-001-10-6")

CCLE_STUDY_ID = "ccle_broad_2019"
CCLE_MUTATION_PROFILE = "ccle_broad_2019_mutations"
CCLE_SAMPLE_LIST = "ccle_broad_2019_sequenced"
CCLE_DEPMAP_ID_ATTRIBUTE = "DEPMAPID"

TCGA_SKCM_STUDY_ID = "skcm_tcga_pan_can_atlas_2018"
TCGA_SKCM_MUTATION_PROFILE = "skcm_tcga_pan_can_atlas_2018_mutations"
TCGA_SKCM_SAMPLE_LIST = "skcm_tcga_pan_can_atlas_2018_sequenced"

BRAF_ENTREZ_GENE_ID = 673

_V600_PATTERN = re.compile(r"^V600([A-Z])$")


class PrismClient:
    """Downloads the DepMap PRISM Repurposing 20Q2 secondary-screen dose-response table."""

    def download_dose_response(self, out_dir, force=False):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / PRISM_DOSE_RESPONSE_FILENAME

        if dest.exists() and not force:
            return dest

        response = requests.get(PRISM_DOSE_RESPONSE_URL, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))

        if dest.stat().st_size < PRISM_DOSE_RESPONSE_MIN_BYTES:
            size = dest.stat().st_size
            dest.unlink()
            raise ValueError(
                f"Downloaded {dest.name} is only {size} bytes (expected >= "
                f"{PRISM_DOSE_RESPONSE_MIN_BYTES}); likely a truncated download or an error "
                "page. File removed."
            )

        return dest


class CBioPortalClient:
    """Thin wrapper for the handful of cBioPortal REST calls this case study needs."""

    def __init__(self, base_url: str = CBIOPORTAL_API, timeout: int = 60):
        self.base_url = base_url
        self.timeout = timeout

    def download_mutations(self, molecular_profile_id, sample_list_id, entrez_gene_id, out_dir, force=False):
        """Cache the DETAILED mutation records for one gene in one study."""
        out_path = Path(out_dir) / f"{molecular_profile_id}_gene{entrez_gene_id}_mutations.json"
        if out_path.exists() and not force:
            return out_path
        response = requests.post(
            f"{self.base_url}/molecular-profiles/{molecular_profile_id}/mutations/fetch",
            params={"projection": "DETAILED"},
            json={"entrezGeneIds": [entrez_gene_id], "sampleListId": sample_list_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        self._write_json(out_path, response.json())
        return out_path

    def download_sample_ids(self, sample_list_id, out_dir, force=False):
        """Cache every sample id in a sample list (the "sequenced" denominator)."""
        out_path = Path(out_dir) / f"{sample_list_id}_sample_ids.json"
        if out_path.exists() and not force:
            return out_path
        response = requests.get(
            f"{self.base_url}/sample-lists/{sample_list_id}/sample-ids", timeout=self.timeout
        )
        response.raise_for_status()
        self._write_json(out_path, response.json())
        return out_path

    def download_clinical_data(self, study_id, attribute_id, out_dir, force=False, page_size=20000):
        """Cache one sample-level clinical attribute for every sample in a study."""
        out_path = Path(out_dir) / f"{study_id}_{attribute_id.lower()}_clinical.json"
        if out_path.exists() and not force:
            return out_path
        response = requests.get(
            f"{self.base_url}/studies/{study_id}/clinical-data",
            params={
                "clinicalDataType": "SAMPLE",
                "attributeId": attribute_id,
                "pageSize": page_size,
                "projection": "SUMMARY",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        self._write_json(out_path, response.json())
        return out_path

    @staticmethod
    def _write_json(out_path, payload):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload), encoding="utf-8")


def is_braf_v600(protein_change, mutation_type=""):
    """True for an actual V600 substitution (V600E/K/D/M/R/...), not a synonymous call.

    Guards against both ways a "silent" BRAF hit could otherwise slip through as
    a false mutant: an explicit ``mutation_type`` of ``"Silent"``, or a
    ``protein_change`` of ``"V600V"`` (the reference residue recurring -- a
    synonymous codon change at the hotspot position, not a substitution).
    """
    if str(mutation_type).lower() == "silent":
        return False
    if not protein_change:
        return False
    match = _V600_PATTERN.match(str(protein_change).strip())
    return bool(match) and match.group(1) != "V"


def classify_braf_status(mutation_records):
    """"mutant" if any record is a BRAF V600 substitution, else "WT".

    ``mutation_records`` is an iterable of dicts with a ``proteinChange`` (or
    ``protein_change``) key and optionally ``mutationType``/``mutation_type``
    -- the shape cBioPortal's ``mutations/fetch`` returns directly.
    """
    for record in mutation_records:
        protein_change = record.get("proteinChange", record.get("protein_change"))
        mutation_type = record.get("mutationType", record.get("mutation_type", ""))
        if is_braf_v600(protein_change, mutation_type):
            return "mutant"
    return "WT"


def braf_status_by_sample(mutations_path, sequenced_sample_ids_path):
    """{sampleId: "mutant"/"WT"} for every sequenced sample (unsequenced samples absent)."""
    with open(mutations_path, encoding="utf-8") as f:
        mutations = json.load(f)
    with open(sequenced_sample_ids_path, encoding="utf-8") as f:
        sequenced_ids = json.load(f)

    records_by_sample = {}
    for record in mutations:
        records_by_sample.setdefault(record["sampleId"], []).append(record)

    return {
        sample_id: classify_braf_status(records_by_sample.get(sample_id, []))
        for sample_id in sequenced_ids
    }


def load_ccle_depmap_id_map(clinical_data_path):
    """cBioPortal CCLE sampleId (e.g. "A375_SKIN") -> DepMap ACH id, keyed by sampleId."""
    with open(clinical_data_path, encoding="utf-8") as f:
        records = json.load(f)
    return {record["sampleId"]: record["value"] for record in records}


def load_vemurafenib_sensitivity(dose_response_path):
    """PRISM secondary-screen vemurafenib AUC per CCLE line (mean over replicate screens).

    Returns a DataFrame indexed by ``depmap_id`` with ``vemurafenib_auc`` (raw,
    lower = more sensitive) and ``vemurafenib_auc_z`` (z-scored across the
    lines with a readout).
    """
    df = pd.read_csv(dose_response_path, low_memory=False)
    df = df[df["broad_id"].isin(VEMURAFENIB_BROAD_IDS)]
    agg = df.groupby("depmap_id")["auc"].mean().to_frame("vemurafenib_auc")
    std = agg["vemurafenib_auc"].std(ddof=0)
    agg["vemurafenib_auc_z"] = (agg["vemurafenib_auc"] - agg["vemurafenib_auc"].mean()) / std if std else 0.0
    return agg


def assemble_braf_table(
    ccle_embeddings_path,
    tcga_embeddings_path,
    ccle_mutations_path,
    ccle_sequenced_ids_path,
    ccle_depmap_id_map_path,
    tcga_mutations_path,
    tcga_sequenced_ids_path,
    vemurafenib_path,
    lineage="SKCM",
    lineage_idx=2,
):
    """Tidy sample_id/domain/lineage/BRAF_status/vemurafenib/embedding frame, SKCM only.

    Cell lines get a real ``vemurafenib_auc`` (from PRISM) where tested;
    patients get ``NaN`` there (vemurafenib is not a response readout TCGA
    collected). Samples without a resolvable BRAF call (not in the cBioPortal
    "sequenced" list for that study, or -- CCLE only -- no DepMap id mapping)
    are dropped rather than imputed.
    """
    ccle_ach_to_sample = {v: k for k, v in load_ccle_depmap_id_map(ccle_depmap_id_map_path).items()}
    ccle_braf = braf_status_by_sample(ccle_mutations_path, ccle_sequenced_ids_path)
    vemurafenib = load_vemurafenib_sensitivity(vemurafenib_path)

    with np.load(ccle_embeddings_path, allow_pickle=True) as d:
        ccle_ids = np.array([str(s) for s in d["ids"]])
        ccle_mask = np.asarray(d["y"]) == lineage_idx
        ccle_ids, ccle_z = ccle_ids[ccle_mask], np.asarray(d["z"])[ccle_mask]

    ccle_rows = []
    for ach_id, z in zip(ccle_ids, ccle_z):
        sample_id = ccle_ach_to_sample.get(ach_id)
        status = ccle_braf.get(sample_id) if sample_id is not None else None
        if status is None:
            continue
        auc = vemurafenib["vemurafenib_auc"].get(ach_id, np.nan)
        auc_z = vemurafenib["vemurafenib_auc_z"].get(ach_id, np.nan)
        ccle_rows.append(
            {
                "sample_id": ach_id,
                "domain": "cell_line",
                "lineage": lineage,
                "BRAF_status": status,
                "vemurafenib_auc": auc,
                "vemurafenib_auc_z": auc_z,
                "embedding": np.asarray(z, dtype=np.float64),
            }
        )

    tcga_braf = braf_status_by_sample(tcga_mutations_path, tcga_sequenced_ids_path)
    with np.load(tcga_embeddings_path, allow_pickle=True) as d:
        tcga_ids = np.array([str(s) for s in d["ids_tcga"]])
        tcga_mask = np.asarray(d["y_tcga"]) == lineage_idx
        tcga_ids, tcga_z = tcga_ids[tcga_mask], np.asarray(d["z_tcga"])[tcga_mask]

    tcga_rows = []
    for sample_id, z in zip(tcga_ids, tcga_z):
        status = tcga_braf.get(sample_id)
        if status is None:
            continue
        tcga_rows.append(
            {
                "sample_id": sample_id,
                "domain": "patient",
                "lineage": lineage,
                "BRAF_status": status,
                "vemurafenib_auc": np.nan,
                "vemurafenib_auc_z": np.nan,
                "embedding": np.asarray(z, dtype=np.float64),
            }
        )

    columns = ["sample_id", "domain", "lineage", "BRAF_status", "vemurafenib_auc", "vemurafenib_auc_z", "embedding"]
    return pd.DataFrame(ccle_rows + tcga_rows, columns=columns)


def coverage_summary(table):
    """Counts for the daily report: N per domain, BRAF split, vemurafenib join coverage."""
    cell_lines = table[table["domain"] == "cell_line"]
    patients = table[table["domain"] == "patient"]
    with_vemurafenib = cell_lines["vemurafenib_auc"].notna()
    return {
        "n_cell_lines": int(len(cell_lines)),
        "n_patients": int(len(patients)),
        "n_cell_lines_with_vemurafenib": int(with_vemurafenib.sum()),
        "cell_line_braf_split": cell_lines["BRAF_status"].value_counts().to_dict(),
        "patient_braf_split": patients["BRAF_status"].value_counts().to_dict(),
        "cell_line_with_vemurafenib_braf_split": cell_lines.loc[with_vemurafenib, "BRAF_status"]
        .value_counts()
        .to_dict(),
    }


# --------------------------------------------------------------------------
# Day 23 -- Part A: placement (does the space recover a driver-defined
# subgroup?) and Part B: response link (does proximity predict drug response?)
# --------------------------------------------------------------------------


def braf_mutant_patient_centroid(patient_embeddings, patient_status):
    """Mean embedding of the BRAF-mutant TCGA SKCM patients -- the Part-A reference point."""
    embeddings = np.asarray(list(patient_embeddings), dtype=np.float64)
    mask = np.asarray(patient_status) == "mutant"
    if mask.sum() == 0:
        raise ValueError("No BRAF-mutant patients available to compute a centroid from.")
    return embeddings[mask].mean(axis=0)


def distance_to_centroid(embeddings, centroid):
    """Euclidean distance of every row in ``embeddings`` to a fixed ``centroid``."""
    embeddings = np.asarray(list(embeddings), dtype=np.float64)
    centroid = np.asarray(centroid, dtype=np.float64)
    return np.linalg.norm(embeddings - centroid, axis=1)


def pairwise_greater_fraction(greater_group, less_group):
    """Fraction of (greater, less) pairs where ``greater`` truly exceeds ``less`` (ties = 0.5).

    A direct, easily-tested common-language effect size: 1.0 means every value
    in ``greater_group`` exceeds every value in ``less_group``; 0.5 means no
    separation; 0.0 means the ordering is reversed.
    """
    greater_group = np.asarray(greater_group, dtype=np.float64)
    less_group = np.asarray(less_group, dtype=np.float64)
    if greater_group.size == 0 or less_group.size == 0:
        return float("nan")
    diff = greater_group[:, None] - less_group[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def _bootstrap_two_sample_ci(a, b, statistic, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for a two-sample ``statistic(a, b)``, resampled independently."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    point = float(statistic(a, b))
    if a.size == 0 or b.size == 0:
        return {"point": point, "ci_low": point, "ci_high": point, "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        a_boot = a[rng.integers(0, a.size, size=a.size)]
        b_boot = b[rng.integers(0, b.size, size=b.size)]
        boots[i] = statistic(a_boot, b_boot)
    lo = float(np.percentile(boots, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_boot": n_boot}


def braf_placement_test(
    cell_line_embeddings, cell_line_status, patient_embeddings, patient_status, n_boot=2000, seed=0, alpha=0.05
):
    """Part A: are BRAF-mutant SKCM cell lines closer to the BRAF-mutant-patient centroid than WT lines?

    Computes the BRAF-mutant patient centroid, then a one-sided Mann-Whitney U
    test (mutant-line distances stochastically *less* than WT-line distances)
    plus a bootstrapped common-language effect size (``pairwise_greater_fraction``
    of WT distance > mutant distance -- 1.0 is perfect separation, 0.5 is none).
    """
    centroid = braf_mutant_patient_centroid(patient_embeddings, patient_status)
    distances = distance_to_centroid(cell_line_embeddings, centroid)
    status = np.asarray(cell_line_status)
    mut_d = distances[status == "mutant"]
    wt_d = distances[status == "WT"]
    if mut_d.size == 0 or wt_d.size == 0:
        raise ValueError("Need at least one BRAF-mutant and one BRAF-WT cell line to run the placement test.")

    u_stat, p_value = mannwhitneyu(mut_d, wt_d, alternative="less")
    effect = _bootstrap_two_sample_ci(
        wt_d, mut_d, pairwise_greater_fraction, n_boot=n_boot, seed=seed, alpha=alpha
    )

    return {
        "centroid": centroid.tolist(),
        "n_mutant": int(mut_d.size),
        "n_wt": int(wt_d.size),
        "median_distance_mutant": float(np.median(mut_d)),
        "median_distance_wt": float(np.median(wt_d)),
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "effect_size": effect["point"],
        "effect_size_ci_low": effect["ci_low"],
        "effect_size_ci_high": effect["ci_high"],
        "n_boot": n_boot,
    }


def spearman_with_ci(x, y, n_boot=2000, seed=0, alpha=0.05):
    """Spearman rho with a percentile bootstrap CI (paired resampling of ``x``/``y``)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    rho, p_value = spearmanr(x, y) if n >= 2 else (0.0, 1.0)
    rho = float(rho) if np.isfinite(rho) else 0.0

    if n < 2:
        return {"rho": rho, "p_value": float(p_value), "ci_low": rho, "ci_high": rho, "n_boot": n_boot, "n": n}

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = spearmanr(x[idx], y[idx])
        boots[i] = r if np.isfinite(r) else 0.0
    lo = float(np.percentile(boots, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    return {"rho": rho, "p_value": float(p_value), "ci_low": lo, "ci_high": hi, "n_boot": n_boot, "n": n}


def braf_response_link(
    cell_line_embeddings,
    cell_line_status,
    cell_line_vemurafenib_auc,
    patient_embeddings,
    patient_status,
    n_boot=2000,
    seed=0,
    alpha=0.05,
):
    """Part B: does proximity to the BRAF-mutant-patient centroid track vemurafenib sensitivity?

    ``proximity`` is defined as the negative distance to the Part-A centroid
    (higher = closer). Hypothesis: lines placed closer (higher proximity) are
    the more sensitive lines, i.e. *lower* AUC -- so a positive result is a
    **negative** Spearman rho between proximity and raw AUC. Restricted to
    cell lines with a finite ``vemurafenib_auc`` (not every SKCM line was
    screened -- see Day 22 coverage).
    """
    centroid = braf_mutant_patient_centroid(patient_embeddings, patient_status)
    distances = distance_to_centroid(cell_line_embeddings, centroid)
    auc = np.asarray(cell_line_vemurafenib_auc, dtype=np.float64)
    mask = np.isfinite(auc)
    if mask.sum() < 2:
        raise ValueError("Need at least 2 cell lines with a vemurafenib AUC to compute a correlation.")

    proximity = -distances[mask]
    result = spearman_with_ci(proximity, auc[mask], n_boot=n_boot, seed=seed, alpha=alpha)
    result["n_mutant"] = int((np.asarray(cell_line_status)[mask] == "mutant").sum())
    result["n_wt"] = int((np.asarray(cell_line_status)[mask] == "WT").sum())
    return result
