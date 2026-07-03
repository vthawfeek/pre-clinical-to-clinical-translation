"""UCSC Xena Pan-Cancer Atlas download client for TCGA expression + phenotype."""

import gzip
import logging
import shutil
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# PLAN.md's URLs point at the "tcga-xena-hub" S3 bucket, which returns 403 for these
# files. The PANCAN expression + phenotype matrices actually live in the
# "tcga-pancan-atlas-hub" bucket (confirmed via HEAD request against both hosts).
EXPRESSION_URL = (
    "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/"
    "EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
)
EXPRESSION_GZ_FILENAME = "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
EXPRESSION_FILENAME = "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena"
EXPRESSION_MIN_BYTES = 100_000_000  # gzipped file is ~330 MB

PHENOTYPE_URL = (
    "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/"
    "Survival_SupplementalTable_S1_20171025_xena_sp"
)
PHENOTYPE_FILENAME = "Survival_SupplementalTable_S1_20171025_xena_sp.tsv"
PHENOTYPE_MIN_BYTES = 1_000_000  # real file is ~2.4 MB; already tab-separated, not gzipped

# Day 20: ABSOLUTE consensus purity/ploidy calls (Carter et al. 2012 method,
# PanCanAtlas "mastercalls" release), hosted by GDC under a stable file UUID.
# Keyed by `array` = sample barcode (patient + sample-type code, e.g.
# "TCGA-OR-A5J1-01") -- the same barcode convention as our processed frame
# index, so no ID munging is needed at join time.
PURITY_URL = "https://api.gdc.cancer.gov/data/4f277128-f793-4354-a13d-30cc7fe9f6b5"
PURITY_FILENAME = "TCGA_mastercalls.abs_tables_JSedit.fixed.txt"
PURITY_MIN_BYTES = 500_000  # real file is ~900 KB, 10,786 samples


class TCGAClient:
    """Downloads TCGA Pan-Cancer Atlas expression + phenotype from UCSC Xena (S3-hosted)."""

    def download_expression(self, out_dir, force=False):
        dest = Path(out_dir) / EXPRESSION_FILENAME

        if dest.exists() and not force:
            logger.info("%s already exists at %s, skipping download (force=False)", EXPRESSION_FILENAME, dest)
            return dest

        gz_path = self._download(EXPRESSION_URL, out_dir, EXPRESSION_GZ_FILENAME, EXPRESSION_MIN_BYTES, force)

        with gzip.open(gz_path, "rb") as f_in, open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()

        return dest

    def download_phenotype(self, out_dir, force=False):
        return self._download(PHENOTYPE_URL, out_dir, PHENOTYPE_FILENAME, PHENOTYPE_MIN_BYTES, force)

    def download_purity(self, out_dir, force=False):
        return self._download(PURITY_URL, out_dir, PURITY_FILENAME, PURITY_MIN_BYTES, force)

    def _download(self, url, out_dir, filename, min_bytes, force):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / filename

        if dest.exists() and not force:
            logger.info("%s already exists at %s, skipping download (force=False)", filename, dest)
            return dest

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))

        if dest.stat().st_size < min_bytes:
            size = dest.stat().st_size
            dest.unlink()
            raise ValueError(
                f"Downloaded {filename} is only {size} bytes (expected >= {min_bytes}); "
                "likely a truncated download or an error page. File removed."
            )

        return dest


def filter_tcga_lineages(df_pheno: pd.DataFrame, lineages: list[str]) -> pd.Series:
    """Map each row's 'cancer type abbreviation' to a boolean mask over `lineages`.

    Unlike CCLE's OncotreePrimaryDisease, TCGA's abbreviation column already uses the
    target codes directly (LUAD/BRCA/SKCM), so no alias table is needed.
    """
    return df_pheno["cancer type abbreviation"].isin(lineages)
