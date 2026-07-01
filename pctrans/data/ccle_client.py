"""DepMap 24Q4 download client for CCLE expression + metadata."""

import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

EXPRESSION_URL = "https://ndownloader.figshare.com/files/51065489"
EXPRESSION_FILENAME = "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
EXPRESSION_MIN_BYTES = 100_000_000  # real file is ~500 MB; catches truncated/error-page downloads

METADATA_URL = "https://ndownloader.figshare.com/files/51065297"
METADATA_FILENAME = "Model.csv"
METADATA_MIN_BYTES = 100_000  # real file is ~650 KB

# DepMap 24Q4's OncotreePrimaryDisease has no single "Lung Adenocarcinoma" or
# "Breast Cancer" bucket (NSCLC lumps LUAD+LUSC together; breast carcinomas are split
# across several subtypes) so LUAD/BRCA require the finer OncotreeSubtype label.
# Melanoma resolves at the primary-disease level already.
LINEAGE_ALIASES = {
    "Melanoma": "SKCM",
    "Cutaneous Melanoma": "SKCM",
    "Lung Adenocarcinoma": "LUAD",
    "Invasive Breast Carcinoma": "BRCA",
    "Breast Invasive Ductal Carcinoma": "BRCA",
    "Breast Invasive Lobular Carcinoma": "BRCA",
    "Breast Invasive Carcinoma, NOS": "BRCA",
    "Breast Invasive Cancer, NOS": "BRCA",
    "Breast Ductal Carcinoma In Situ": "BRCA",
    "Breast Neoplasm, NOS": "BRCA",
}


class CCLEClient:
    """Downloads DepMap 24Q4 CCLE expression + model metadata from Figshare."""

    def download_expression(self, out_dir, force=False):
        return self._download(EXPRESSION_URL, out_dir, EXPRESSION_FILENAME, EXPRESSION_MIN_BYTES, force)

    def download_metadata(self, out_dir, force=False):
        return self._download(METADATA_URL, out_dir, METADATA_FILENAME, METADATA_MIN_BYTES, force)

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


def filter_lineages(df_meta: pd.DataFrame, lineages: list[str]) -> pd.Series:
    """Map each row's OncotreePrimaryDisease/OncotreeSubtype to a LUAD/BRCA/SKCM code.

    Returns a boolean Series (aligned to df_meta's index) selecting rows whose resolved
    lineage code is in `lineages`.
    """
    resolved = df_meta["OncotreePrimaryDisease"].map(LINEAGE_ALIASES)
    resolved = resolved.fillna(df_meta["OncotreeSubtype"].map(LINEAGE_ALIASES))
    return resolved.isin(lineages)
