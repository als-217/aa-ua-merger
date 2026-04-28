"""
utils.py — Shared utilities for BTS airline data downloaders.

Provides:
  - Logging setup (rotating file + console)
  - File-based download cache (SHA-256 registry)
  - Chunked HTTP download with tqdm progress bar
  - Parquet save helper with dtype optimisation
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
import zipfile
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Return a logger with console + optional rotating-file handler."""
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on reimport
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File — DEBUG and above (10 MB × 5 backups)
    fname = log_file or f"{name}.log"
    fh = RotatingFileHandler(
        LOG_DIR / fname, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Download cache
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = CACHE_DIR / "download_registry.json"


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_registry(registry: dict) -> None:
    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8"
    )


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def is_cached(url: str, dest: Path) -> bool:
    """Return True if *dest* exists and matches the registry entry for *url*."""
    if not dest.exists():
        return False
    registry = _load_registry()
    key = url
    if key not in registry:
        return False
    recorded_sha = registry[key].get("sha256", "")
    if not recorded_sha:
        return False
    actual_sha = _file_sha256(dest)
    return actual_sha == recorded_sha


def mark_cached(url: str, dest: Path) -> None:
    """Record *dest* in the registry after a successful download."""
    registry = _load_registry()
    registry[url] = {
        "path": str(dest),
        "sha256": _file_sha256(dest),
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _save_registry(registry)


# ---------------------------------------------------------------------------
# HTTP download
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1 << 20   # 1 MB chunks — keeps memory flat during streaming


def download_file(
    url: str,
    dest: Path,
    *,
    session: Optional[requests.Session] = None,
    logger: Optional[logging.Logger] = None,
    headers: Optional[dict] = None,
    post_data: Optional[dict] = None,
    retries: int = 3,
    backoff: float = 5.0,
    skip_cache: bool = False,
) -> Path:
    """
    Download *url* to *dest*, streaming in 1 MB chunks.

    - Skips download if already cached (unless skip_cache=True).
    - Retries up to *retries* times with exponential backoff.
    - Displays a tqdm progress bar.
    - Returns the destination path.
    """
    log = logger or logging.getLogger(__name__)

    if not skip_cache and is_cached(url, dest):
        log.info("CACHE HIT  %s → %s", url, dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    s = session or requests.Session()

    for attempt in range(1, retries + 1):
        try:
            log.debug("GET attempt %d/%d  %s", attempt, retries, url)
            if post_data is not None:
                resp = s.post(url, data=post_data, headers=headers, stream=True, timeout=120)
            else:
                resp = s.get(url, headers=headers, stream=True, timeout=120)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0)) or None
            desc = dest.name[:40]
            tmp = dest.with_suffix(".tmp")

            with open(tmp, "wb") as fh, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
                leave=False,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    fh.write(chunk)
                    bar.update(len(chunk))

            tmp.rename(dest)
            mark_cached(url, dest)
            log.info("DOWNLOADED %s → %s", url, dest.name)
            return dest

        except (requests.RequestException, OSError) as exc:
            log.warning("Attempt %d failed: %s", attempt, exc)
            if attempt < retries:
                sleep_for = backoff * (2 ** (attempt - 1))
                log.info("Retrying in %.0f s …", sleep_for)
                time.sleep(sleep_for)
            else:
                log.error("All %d attempts failed for %s", retries, url)
                raise

    raise RuntimeError(f"Download failed after {retries} attempts: {url}")


# ---------------------------------------------------------------------------
# Parquet save helper
# ---------------------------------------------------------------------------

DTYPE_MAP: dict[str, str] = {
    # strings → category saves ~50–70 % memory for low-cardinality columns
    "object": "category",
}

# Columns known to be low-cardinality codes — force to category
CATEGORY_COLS = {
    "Year", "Quarter", "Month",
    "Origin", "Dest",
    "OriginState", "DestState",
    "OriginCountry", "DestCountry",
    "TkCarrier", "OpCarrier", "RPCarrier",
    "UniqueCarrier", "Carrier",
    "FareClass", "CouponType", "Break",
    "AircraftType", "Class", "ServiceClass",
    "OriginWac", "DestWac",
}


def save_parquet(
    df: pd.DataFrame,
    path: Path,
    *,
    logger: Optional[logging.Logger] = None,
    compress_categories: bool = True,
) -> None:
    """
    Save *df* as Parquet with memory-efficient dtypes.

    - Downcasts int64 → int32 where values fit.
    - Downcasts float64 → float32.
    - Converts known category columns to pandas CategoricalDtype.
    """
    log = logger or logging.getLogger(__name__)
    path.parent.mkdir(parents=True, exist_ok=True)

    if compress_categories:
        for col in df.columns:
            if col in CATEGORY_COLS and df[col].dtype == object:
                df[col] = df[col].astype("category")
        # Downcast numerics
        for col in df.select_dtypes("int64").columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        for col in df.select_dtypes("float64").columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    size_mb = path.stat().st_size / 1e6
    log.info("Saved %s  (%.1f MB, %d rows)", path.name, size_mb, len(df))


# ---------------------------------------------------------------------------
# ZIP extraction helper
# ---------------------------------------------------------------------------

def extract_zip(zip_path: Path, dest_dir: Path, logger: Optional[logging.Logger] = None) -> list[Path]:
    """Extract *zip_path* into *dest_dir*. Returns list of extracted file paths."""
    log = logger or logging.getLogger(__name__)
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc=f"Extracting {zip_path.name}", leave=False):
            zf.extract(member, dest_dir)
            extracted.append(dest_dir / member)
            log.debug("Extracted %s", member)
    return extracted
