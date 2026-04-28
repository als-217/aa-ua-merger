#!/usr/bin/env python3
"""
airline_data_collector.py
=========================
Downloads and assembles airline merger analysis data from BTS sources:
  - DB1B Coupon & Ticket (10% O-D itinerary sample)
  - T-100 Domestic Segment data
  - BTS On-Time Performance (schedule/frequency proxy)

Features:
  - Download cache (JSON manifest) — skips already-downloaded files
  - Chunked merge to stay within 64 GB RAM budget
  - TQDM progress bars for every download and processing step
  - CLI flags: --start-year, --end-year, --start-quarter, --end-quarter,
               --year (downloads all quarters for that year),
               --since (downloads from Q1 of that year to latest available)
               --datasets (select subset: db1b, t100, ontime)
               --output-dir, --chunk-size, --force-redownload

Usage examples:
  python airline_data_collector.py --since 2024
  python airline_data_collector.py --year 2023
  python airline_data_collector.py --start-year 2019 --end-year 2022
  python airline_data_collector.py --start-year 2022 --start-quarter 3 --end-year 2024 --end-quarter 2
  python airline_data_collector.py --since 2023 --datasets db1b t100
  python airline_data_collector.py --since 2022 --force-redownload
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BTS TransStats base URLs
BTS_BASE = "https://transtats.bts.gov/PREZIP"

# Dataset URL patterns  {DATASET_KEY: (url_template, filename_template)}
# BTS naming convention: e.g. Origin_and_Destination_Survey_DB1BCoupon_2023_4.zip
DATASET_URL_PATTERNS = {
    "db1b_coupon": {
        "url": f"{BTS_BASE}/Origin_and_Destination_Survey_DB1BCoupon_{{year}}_{{quarter}}.zip",
        "filename": "Origin_and_Destination_Survey_DB1BCoupon_{year}_{quarter}.zip",
        "csv_prefix": "Origin_and_Destination_Survey_DB1BCoupon_{year}_{quarter}",
        "description": "DB1B Coupon (itinerary legs, fares, carriers)",
    },
    "db1b_ticket": {
        "url": f"{BTS_BASE}/Origin_and_Destination_Survey_DB1BTicket_{{year}}_{{quarter}}.zip",
        "filename": "Origin_and_Destination_Survey_DB1BTicket_{year}_{quarter}.zip",
        "csv_prefix": "Origin_and_Destination_Survey_DB1BTicket_{year}_{quarter}",
        "description": "DB1B Ticket (itinerary-level fares, PAX)",
    },
    "t100_segment": {
        "url": f"{BTS_BASE}/T_T100D_SEGMENT_US_CARRIER_ONLY_{{year}}_{{quarter}}.zip",
        "filename": "T_T100D_SEGMENT_US_CARRIER_ONLY_{year}_{quarter}.zip",
        "csv_prefix": "T_T100D_SEGMENT_US_CARRIER_ONLY_{year}_{quarter}",
        "description": "T-100 Domestic Segment (PAX, seats, departures by segment)",
    },
    "ontime": {
        "url": f"{BTS_BASE}/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{{year}}_{{quarter_month}}.zip",
        "filename": "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{quarter_month}.zip",
        "csv_prefix": "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{quarter_month}",
        "description": "On-Time Performance (schedule frequency, carrier ops)",
    },
}

# Map dataset group names → constituent keys
DATASET_GROUPS = {
    "db1b": ["db1b_coupon", "db1b_ticket"],
    "t100": ["t100_segment"],
    "ontime": ["ontime"],
}

# Carriers of interest for the UA/AA merger analysis
TARGET_CARRIERS = {
    "UA": "United Air Lines Inc.",
    "AA": "American Airlines Inc.",
    "DL": "Delta Air Lines Inc.",
    "WN": "Southwest Airlines Co.",
    "B6": "JetBlue Airways Corporation",
    "AS": "Alaska Airlines Inc.",
    "NK": "Spirit Air Lines",
    "F9": "Frontier Airlines Inc.",
    "G4": "Allegiant Air",
}

# DB1B columns to retain (memory optimization)
DB1B_COUPON_COLS = [
    "ItinID", "MktID", "SeqNum", "Origin", "Dest", "Break",
    "OpCarrier", "TkCarrier", "FareClass", "Distance", "DistanceGroup",
    "Year", "Quarter",
]
DB1B_TICKET_COLS = [
    "ItinID", "Origin", "OriginAptInd", "OriginCityMarketID",
    "RoundTrip", "OnLine", "DollarCred", "FarePerMile", "RPCarrier",
    "Passengers", "ItinFare", "BulkFare", "Distance", "DistanceGroup",
    "Year", "Quarter",
]
T100_COLS = [
    "YEAR", "QUARTER", "MONTH", "AIRLINE_ID", "UNIQUE_CARRIER",
    "UNIQUE_CARRIER_NAME", "ORIGIN", "ORIGIN_CITY_MARKET_ID",
    "DEST", "DEST_CITY_MARKET_ID", "AIRCRAFT_TYPE",
    "DEPARTURES_SCHEDULED", "DEPARTURES_PERFORMED",
    "PAYLOAD", "SEATS", "PASSENGERS", "FREIGHT", "MAIL",
    "DISTANCE", "RAMP_TO_RAMP", "AIR_TIME",
]
ONTIME_COLS = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
    "UniqueCarrier", "AirlineID", "Carrier", "TailNum",
    "Origin", "OriginCityMarketID", "Dest", "DestCityMarketID",
    "CRSDepTime", "DepTime", "DepDelay", "CRSArrTime", "ArrTime",
    "ArrDelay", "Cancelled", "Diverted",
    "CRSElapsedTime", "ActualElapsedTime", "AirTime", "Distance",
]

# Latest available quarter (update as BTS releases new data)
LATEST_YEAR = 2024
LATEST_QUARTER = 3   # Q3 2024 is typically the most recent complete quarter

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "collector.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("airline_collector")


# ---------------------------------------------------------------------------
# Cache / manifest
# ---------------------------------------------------------------------------

class DownloadCache:
    """
    JSON manifest tracking completed downloads.
    Schema: { "<filename>": { "status": "ok"|"failed", "size_bytes": int,
                               "md5": str, "timestamp": str } }
    """

    def __init__(self, cache_path: Path):
        self.path = cache_path
        self._data: dict = {}
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def is_complete(self, filename: str, dest_path: Path) -> bool:
        """Return True if file was previously downloaded successfully and still exists."""
        if filename not in self._data:
            return False
        rec = self._data[filename]
        if rec.get("status") != "ok":
            return False
        if not dest_path.exists():
            return False
        return True

    def mark_complete(self, filename: str, dest_path: Path):
        size = dest_path.stat().st_size
        md5 = self._md5(dest_path)
        self._data[filename] = {
            "status": "ok",
            "size_bytes": size,
            "md5": md5,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(dest_path),
        }
        self._save()

    def mark_failed(self, filename: str, reason: str):
        self._data[filename] = {
            "status": "failed",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._save()

    @staticmethod
    def _md5(path: Path, chunk=65536) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            while True:
                block = f.read(chunk)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Quarter / period utilities
# ---------------------------------------------------------------------------

def quarters_in_range(
    start_year: int, start_quarter: int,
    end_year: int, end_quarter: int,
):
    """Yield (year, quarter) tuples inclusive of endpoints."""
    y, q = start_year, start_quarter
    while (y, q) <= (end_year, end_quarter):
        yield y, q
        q += 1
        if q > 4:
            q = 1
            y += 1


def quarter_to_months(quarter: int):
    """Return the three calendar months in a quarter (for on-time data)."""
    return [(quarter - 1) * 3 + m for m in range(1, 4)]


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def download_file(
    url: str,
    dest_path: Path,
    logger: logging.Logger,
    retries: int = 3,
    timeout: int = 120,
) -> bool:
    """
    Stream-download url → dest_path with a TQDM progress bar.
    Returns True on success.
    """
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    logger.warning(f"  404 Not Found: {url}")
                    return False
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                desc = dest_path.name[:50]
                with (
                    open(dest_path, "wb") as f,
                    tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"  ↓ {desc}",
                        leave=False,
                    ) as bar,
                ):
                    for chunk in r.iter_content(chunk_size=1 << 16):  # 64 KB
                        f.write(chunk)
                        bar.update(len(chunk))
            return True
        except (requests.RequestException, OSError) as e:
            logger.warning(f"  Attempt {attempt}/{retries} failed: {e}")
            if dest_path.exists():
                dest_path.unlink()
            if attempt < retries:
                time.sleep(5 * attempt)
    return False


def extract_csv(zip_path: Path, extract_dir: Path, logger: logging.Logger) -> Optional[Path]:
    """Extract the first CSV found in a zip archive. Returns path to CSV."""
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                logger.warning(f"  No CSV in {zip_path.name}")
                return None
            target = csv_names[0]
            out_path = extract_dir / target
            if not out_path.exists():
                logger.info(f"  Extracting {target}")
                z.extract(target, extract_dir)
            return out_path
    except zipfile.BadZipFile as e:
        logger.error(f"  Bad zip {zip_path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-dataset download orchestrators
# ---------------------------------------------------------------------------

def download_db1b(
    years_quarters: list,
    raw_dir: Path,
    cache: DownloadCache,
    logger: logging.Logger,
    force: bool = False,
):
    """Download DB1B Coupon and Ticket zips."""
    tasks = []
    for y, q in years_quarters:
        for key in ("db1b_coupon", "db1b_ticket"):
            pat = DATASET_URL_PATTERNS[key]
            fname = pat["filename"].format(year=y, quarter=q)
            url = pat["url"].format(year=y, quarter=q)
            tasks.append((key, fname, url, y, q))

    pbar = tqdm(tasks, desc="DB1B downloads", unit="file")
    for key, fname, url, y, q in pbar:
        pbar.set_postfix(file=fname[:40])
        dest = raw_dir / "db1b" / fname
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not force and cache.is_complete(fname, dest):
            logger.info(f"[CACHE] Skipping {fname}")
            continue

        logger.info(f"[DOWNLOAD] {fname}")
        ok = download_file(url, dest, logger)
        if ok:
            cache.mark_complete(fname, dest)
        else:
            cache.mark_failed(fname, "download failed")
            logger.error(f"[FAILED] {fname}")


def download_t100(
    years_quarters: list,
    raw_dir: Path,
    cache: DownloadCache,
    logger: logging.Logger,
    force: bool = False,
):
    """Download T-100 Domestic Segment zips."""
    tasks = []
    for y, q in years_quarters:
        pat = DATASET_URL_PATTERNS["t100_segment"]
        fname = pat["filename"].format(year=y, quarter=q)
        url = pat["url"].format(year=y, quarter=q)
        tasks.append((fname, url))

    pbar = tqdm(tasks, desc="T-100 downloads", unit="file")
    for fname, url in pbar:
        pbar.set_postfix(file=fname[:40])
        dest = raw_dir / "t100" / fname
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not force and cache.is_complete(fname, dest):
            logger.info(f"[CACHE] Skipping {fname}")
            continue

        logger.info(f"[DOWNLOAD] {fname}")
        ok = download_file(url, dest, logger)
        if ok:
            cache.mark_complete(fname, dest)
        else:
            cache.mark_failed(fname, "download failed")
            logger.error(f"[FAILED] {fname}")


def download_ontime(
    years_quarters: list,
    raw_dir: Path,
    cache: DownloadCache,
    logger: logging.Logger,
    force: bool = False,
):
    """
    Download On-Time Performance zips.
    BTS on-time files are monthly (not quarterly), so each quarter = 3 files.
    """
    tasks = []
    for y, q in years_quarters:
        for month in quarter_to_months(q):
            pat = DATASET_URL_PATTERNS["ontime"]
            fname = pat["filename"].format(year=y, quarter_month=month)
            url = pat["url"].format(year=y, quarter_month=month)
            tasks.append((fname, url))

    pbar = tqdm(tasks, desc="On-Time downloads", unit="file")
    for fname, url in pbar:
        pbar.set_postfix(file=fname[:40])
        dest = raw_dir / "ontime" / fname
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not force and cache.is_complete(fname, dest):
            logger.info(f"[CACHE] Skipping {fname}")
            continue

        logger.info(f"[DOWNLOAD] {fname}")
        ok = download_file(url, dest, logger)
        if ok:
            cache.mark_complete(fname, dest)
        else:
            cache.mark_failed(fname, "download failed")
            logger.error(f"[FAILED] {fname}")


# ---------------------------------------------------------------------------
# Processing: CSV → Parquet (chunked, memory-safe)
# ---------------------------------------------------------------------------

def csv_to_parquet_chunked(
    csv_path: Path,
    out_path: Path,
    usecols: list,
    chunk_size: int,
    logger: logging.Logger,
    filter_carriers: bool = False,
    carrier_col: Optional[str] = None,
):
    """
    Read a large CSV in chunks, optionally filter to target carriers,
    keep only needed columns, and write to Parquet.
    
    Uses chunked reading so peak RAM ≈ chunk_size rows × ~200 bytes ≈ well under 1 GB
    per file regardless of file size.
    """
    if out_path.exists():
        logger.info(f"  [PARQUET EXISTS] {out_path.name}")
        return

    writer = None

    # Probe actual columns in file to avoid KeyError on missing cols
    header = pd.read_csv(csv_path, nrows=0)
    available = [c for c in usecols if c in header.columns]
    missing = set(usecols) - set(available)
    if missing:
        logger.warning(f"  Missing columns (will skip): {missing}")

    total_rows = 0
    chunk_iter = pd.read_csv(
        csv_path,
        usecols=available,
        chunksize=chunk_size,
        low_memory=False,
        encoding="latin-1",
    )
    pbar = tqdm(chunk_iter, desc=f"  → Parquet {out_path.name[:40]}", unit="chunk", leave=False)

    for chunk in pbar:
        if filter_carriers and carrier_col and carrier_col in chunk.columns:
            chunk = chunk[chunk[carrier_col].isin(TARGET_CARRIERS.keys())]
        if chunk.empty:
            continue
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)
        total_rows += len(chunk)
        pbar.set_postfix(rows=f"{total_rows:,}")

    if writer:
        writer.close()
    logger.info(f"  Wrote {total_rows:,} rows → {out_path.name}")


def process_db1b(
    raw_dir: Path,
    processed_dir: Path,
    years_quarters: list,
    chunk_size: int,
    logger: logging.Logger,
):
    logger.info("=== Processing DB1B ===")
    out_dir = processed_dir / "db1b"
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = raw_dir / "db1b" / "_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    for y, q in tqdm(years_quarters, desc="DB1B periods", unit="quarter"):
        for key, usecols, carrier_col in [
            ("db1b_coupon",  DB1B_COUPON_COLS,  "OpCarrier"),
            ("db1b_ticket",  DB1B_TICKET_COLS,  "RPCarrier"),
        ]:
            pat = DATASET_URL_PATTERNS[key]
            zip_name = pat["filename"].format(year=y, quarter=q)
            zip_path = raw_dir / "db1b" / zip_name
            if not zip_path.exists():
                logger.warning(f"  ZIP not found, skipping: {zip_name}")
                continue

            csv_path = extract_csv(zip_path, extract_dir, logger)
            if csv_path is None:
                continue

            label = "coupon" if "coupon" in key else "ticket"
            out_path = out_dir / f"db1b_{label}_{y}_Q{q}.parquet"
            csv_to_parquet_chunked(
                csv_path, out_path, usecols, chunk_size, logger,
                filter_carriers=True, carrier_col=carrier_col,
            )


def process_t100(
    raw_dir: Path,
    processed_dir: Path,
    years_quarters: list,
    chunk_size: int,
    logger: logging.Logger,
):
    logger.info("=== Processing T-100 ===")
    out_dir = processed_dir / "t100"
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = raw_dir / "t100" / "_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    for y, q in tqdm(years_quarters, desc="T-100 periods", unit="quarter"):
        pat = DATASET_URL_PATTERNS["t100_segment"]
        zip_name = pat["filename"].format(year=y, quarter=q)
        zip_path = raw_dir / "t100" / zip_name
        if not zip_path.exists():
            logger.warning(f"  ZIP not found, skipping: {zip_name}")
            continue

        csv_path = extract_csv(zip_path, extract_dir, logger)
        if csv_path is None:
            continue

        out_path = out_dir / f"t100_{y}_Q{q}.parquet"
        csv_to_parquet_chunked(
            csv_path, out_path, T100_COLS, chunk_size, logger,
            filter_carriers=True, carrier_col="UNIQUE_CARRIER",
        )


def process_ontime(
    raw_dir: Path,
    processed_dir: Path,
    years_quarters: list,
    chunk_size: int,
    logger: logging.Logger,
):
    logger.info("=== Processing On-Time ===")
    out_dir = processed_dir / "ontime"
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = raw_dir / "ontime" / "_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for y, q in years_quarters:
        for month in quarter_to_months(q):
            tasks.append((y, q, month))

    for y, q, month in tqdm(tasks, desc="On-Time periods", unit="month"):
        pat = DATASET_URL_PATTERNS["ontime"]
        zip_name = pat["filename"].format(year=y, quarter_month=month)
        zip_path = raw_dir / "ontime" / zip_name
        if not zip_path.exists():
            logger.warning(f"  ZIP not found, skipping: {zip_name}")
            continue

        csv_path = extract_csv(zip_path, extract_dir, logger)
        if csv_path is None:
            continue

        out_path = out_dir / f"ontime_{y}_M{month:02d}.parquet"
        csv_to_parquet_chunked(
            csv_path, out_path, ONTIME_COLS, chunk_size, logger,
            filter_carriers=True, carrier_col="UniqueCarrier",
        )


# ---------------------------------------------------------------------------
# Assembly: merge parquet files into analysis-ready panels (chunked)
# ---------------------------------------------------------------------------

def assemble_panel(
    processed_dir: Path,
    output_dir: Path,
    dataset: str,
    chunk_size_rows: int,
    logger: logging.Logger,
):
    """
    Concatenate all quarterly parquet files for a dataset into a single
    analysis panel parquet, written in chunks to stay within RAM budget.

    Peak RAM per chunk ≈ chunk_size_rows × avg_row_bytes.
    For DB1B Coupon (~150 bytes/row) and chunk_size=500_000:
        500_000 × 150 = ~75 MB per chunk — safely under any RAM budget.
    """

    src_dir = processed_dir / dataset
    files = sorted(src_dir.glob("*.parquet"))
    if not files:
        logger.warning(f"No parquet files found in {src_dir}")
        return

    out_path = output_dir / f"{dataset}_panel.parquet"
    if out_path.exists():
        logger.info(f"[EXISTS] {out_path.name} — skipping assembly")
        return

    logger.info(f"Assembling {len(files)} files → {out_path.name}")
    writer = None
    total_rows = 0

    for fpath in tqdm(files, desc=f"Assembling {dataset}", unit="file"):
        pf = pq.ParquetFile(fpath)
        # Read each source file in batches to bound RAM
        for batch in pf.iter_batches(batch_size=chunk_size_rows):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += len(batch)

    if writer:
        writer.close()
    logger.info(f"  Panel written: {total_rows:,} rows → {out_path}")


def build_od_market_panel(
    output_dir: Path,
    chunk_size_rows: int,
    logger: logging.Logger,
):
    """
    Join DB1B Coupon + Ticket at the ItinID level to construct O-D market panel:
      - Origin, Dest (directional O-D pair)
      - Operating carrier, Ticketing carrier
      - Itinerary fare, passengers
      - Nonstop indicator (SeqNum==1 and Break is not a connection)
      - Distance
      - Year, Quarter

    Written in chunks from the panel parquets to stay within 64 GB RAM.
    DB1B Coupon panel may be 10–30 GB; reading in 500K-row batches ≈ 100–300 MB peak.
    """

    coupon_path = output_dir / "db1b_panel.parquet"
    ticket_path = output_dir / "db1b_panel.parquet"  # Ticket assembled separately

    # Check for separate coupon/ticket panels
    coupon_p = output_dir / "db1b_coupon_panel.parquet"
    ticket_p = output_dir / "db1b_ticket_panel.parquet"

    if not coupon_p.exists() or not ticket_p.exists():
        logger.warning("DB1B coupon or ticket panel not found — skipping O-D market build.")
        return

    out_path = output_dir / "od_market_panel.parquet"
    if out_path.exists():
        logger.info("[EXISTS] od_market_panel.parquet — skipping")
        return

    logger.info("Building O-D market panel from DB1B Coupon + Ticket …")

    # Step 1: Load ticket panel (smaller) fully into memory as a lookup
    logger.info("  Loading DB1B Ticket panel …")
    ticket_df = pd.read_parquet(ticket_p, columns=["ItinID", "Passengers", "ItinFare",
                                                     "RPCarrier", "RoundTrip", "OnLine",
                                                     "Year", "Quarter"])
    logger.info(f"  Ticket panel: {len(ticket_df):,} rows in memory")

    pf_coupon = pq.ParquetFile(coupon_p)
    writer = None
    total_rows = 0

    for batch in tqdm(
        pf_coupon.iter_batches(batch_size=chunk_size_rows),
        desc="  Building O-D panel",
        unit="batch",
    ):
        coupon_chunk = batch.to_pandas()

        # Filter to first coupon only (nonstop or first leg) for O-D market definition
        coupon_chunk = coupon_chunk[coupon_chunk["SeqNum"] == 1].copy()

        # Merge with ticket
        merged = coupon_chunk.merge(
            ticket_df[["ItinID", "Passengers", "ItinFare", "RPCarrier", "RoundTrip"]],
            on="ItinID",
            how="left",
        )

        # Nonstop proxy: single-coupon itinerary (Break == 'X' means through/nonstop)
        merged["Nonstop"] = merged["Break"].isin(["X", " "]).astype("int8")

        # Adjust for round-trip (fare splitting is standard in DB1B work)
        merged["OneWayFare"] = merged["ItinFare"] / merged.get("RoundTrip", 1).replace(0, 1)

        keep = ["ItinID", "Origin", "Dest", "OpCarrier", "TkCarrier", "RPCarrier",
                "Passengers", "ItinFare", "OneWayFare", "FareClass", "Distance",
                "Nonstop", "Year", "Quarter"]
        merged = merged[[c for c in keep if c in merged.columns]]

        table = pa.Table.from_pandas(merged, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)
        total_rows += len(merged)

    if writer:
        writer.close()
    logger.info(f"  O-D market panel: {total_rows:,} rows → {out_path.name}")


def aggregate_market_shares(
    output_dir: Path,
    logger: logging.Logger,
):
    """
    From the O-D market panel, compute:
      - Market-level shares by carrier (for HHI computation)
      - Average fares by carrier-route
      - Passenger volumes by carrier-route
    Writes a compact summary parquet used directly in demand estimation.
    """
    od_path = output_dir / "od_market_panel.parquet"
    if not od_path.exists():
        logger.warning("O-D market panel not found — skipping share aggregation")
        return

    out_path = output_dir / "market_shares.parquet"
    if out_path.exists():
        logger.info("[EXISTS] market_shares.parquet — skipping")
        return

    logger.info("Aggregating market shares …")
    df = pd.read_parquet(od_path)

    # Market = (Origin, Dest, Year, Quarter)
    mkt_total = (
        df.groupby(["Origin", "Dest", "Year", "Quarter"])["Passengers"]
        .sum()
        .rename("MktPassengers")
        .reset_index()
    )

    carrier_mkt = (
        df.groupby(["Origin", "Dest", "OpCarrier", "Year", "Quarter"])
        .agg(
            Passengers=("Passengers", "sum"),
            AvgFare=("OneWayFare", "mean"),
            AvgDistance=("Distance", "mean"),
            NonstopShare=("Nonstop", "mean"),
        )
        .reset_index()
    )

    merged = carrier_mkt.merge(mkt_total, on=["Origin", "Dest", "Year", "Quarter"])
    merged["Share"] = merged["Passengers"] / merged["MktPassengers"]
    merged["ShareSq"] = merged["Share"] ** 2

    # HHI per market-quarter
    hhi = (
        merged.groupby(["Origin", "Dest", "Year", "Quarter"])["ShareSq"]
        .sum()
        .mul(10000)
        .rename("HHI")
        .reset_index()
    )

    final = merged.merge(hhi, on=["Origin", "Dest", "Year", "Quarter"])
    final.to_parquet(out_path, index=False, compression="snappy")
    logger.info(f"  Market shares: {len(final):,} rows → {out_path.name}")

    # Print summary
    overlap = final[
        final["Origin"].isin(final["Origin"]) &
        final.groupby(["Origin", "Dest", "Year", "Quarter"])["OpCarrier"]
        .transform(lambda x: ("UA" in x.values) and ("AA" in x.values))
    ]
    logger.info(
        f"  Markets with both UA & AA: "
        f"{overlap[['Origin','Dest']].drop_duplicates().shape[0]:,} unique O-D pairs"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and process BTS airline data for merger analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Time range flags ──────────────────────────────────────────────────
    time_group = parser.add_argument_group("Time range (mutually exclusive shortcuts)")
    time_group.add_argument(
        "--since", metavar="YEAR",
        type=int,
        help="Download from Q1 of YEAR through the latest available quarter.",
    )
    time_group.add_argument(
        "--year", metavar="YEAR",
        type=int,
        help="Download all four quarters of a single year.",
    )

    detail_group = parser.add_argument_group("Fine-grained time range")
    detail_group.add_argument("--start-year",  type=int, default=2018)
    detail_group.add_argument("--start-quarter", type=int, default=1, choices=[1,2,3,4])
    detail_group.add_argument("--end-year",    type=int, default=LATEST_YEAR)
    detail_group.add_argument("--end-quarter",  type=int, default=LATEST_QUARTER, choices=[1,2,3,4])

    # ── Dataset selection ─────────────────────────────────────────────────
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["db1b", "t100", "ontime", "all"],
        default=["all"],
        help="Datasets to download/process (default: all).",
    )

    # ── Paths ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./airline_data"),
        help="Root output directory (default: ./airline_data).",
    )

    # ── Memory / performance ──────────────────────────────────────────────
    parser.add_argument(
        "--chunk-size", type=int, default=500_000,
        help=(
            "Rows per processing chunk (default: 500,000). "
            "Lower this if you hit memory pressure; raise for speed."
        ),
    )

    # ── Behaviour ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--force-redownload", action="store_true",
        help="Ignore download cache and re-download all files.",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Only download raw ZIPs; skip CSV extraction and parquet conversion.",
    )
    parser.add_argument(
        "--process-only", action="store_true",
        help="Skip downloads; only (re)process already-downloaded ZIPs.",
    )
    parser.add_argument(
        "--no-assemble", action="store_true",
        help="Skip final panel assembly step.",
    )

    return parser.parse_args()


def resolve_time_range(args) -> tuple[int, int, int, int]:
    """Resolve CLI flags to (start_year, start_quarter, end_year, end_quarter)."""
    if args.since:
        return args.since, 1, LATEST_YEAR, LATEST_QUARTER
    if args.year:
        return args.year, 1, args.year, 4
    return args.start_year, args.start_quarter, args.end_year, args.end_quarter


def resolve_datasets(args) -> list[str]:
    if "all" in args.datasets:
        return ["db1b", "t100", "ontime"]
    return args.datasets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    start_year, start_quarter, end_year, end_quarter = resolve_time_range(args)
    selected_datasets = resolve_datasets(args)
    output_dir: Path = args.output_dir
    chunk_size: int = args.chunk_size

    # Directory layout
    raw_dir       = output_dir / "raw"
    processed_dir = output_dir / "processed"
    panels_dir    = output_dir / "panels"
    for d in [raw_dir, processed_dir, panels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    cache  = DownloadCache(output_dir / "download_cache.json")

    yq_list = list(quarters_in_range(start_year, start_quarter, end_year, end_quarter))

    logger.info("=" * 60)
    logger.info("Airline Merger Data Collector")
    logger.info(f"  Period   : {start_year} Q{start_quarter} → {end_year} Q{end_quarter}")
    logger.info(f"  Quarters : {len(yq_list)}")
    logger.info(f"  Datasets : {selected_datasets}")
    logger.info(f"  Output   : {output_dir.resolve()}")
    logger.info(f"  Chunk sz : {chunk_size:,} rows")
    logger.info("=" * 60)

    # ── 1. Downloads ───────────────────────────────────────────────────────
    if not args.process_only:
        if "db1b" in selected_datasets:
            download_db1b(yq_list, raw_dir, cache, logger, force=args.force_redownload)
        if "t100" in selected_datasets:
            download_t100(yq_list, raw_dir, cache, logger, force=args.force_redownload)
        if "ontime" in selected_datasets:
            download_ontime(yq_list, raw_dir, cache, logger, force=args.force_redownload)

    if args.download_only:
        logger.info("--download-only set; exiting after downloads.")
        return

    # ── 2. Extract & convert to Parquet ───────────────────────────────────
    if "db1b" in selected_datasets:
        process_db1b(raw_dir, processed_dir, yq_list, chunk_size, logger)
    if "t100" in selected_datasets:
        process_t100(raw_dir, processed_dir, yq_list, chunk_size, logger)
    if "ontime" in selected_datasets:
        process_ontime(raw_dir, processed_dir, yq_list, chunk_size, logger)

    # ── 3. Assemble panels ─────────────────────────────────────────────────
    if not args.no_assemble:
        for ds_key, subdir in [
            ("db1b",   "db1b"),
            ("t100",   "t100"),
            ("ontime", "ontime"),
        ]:
            if ds_key not in selected_datasets:
                continue
            # Coupon and ticket are stored separately under db1b/
            if ds_key == "db1b":
                for label in ("coupon", "ticket"):
                    sub = processed_dir / "db1b"
                    # Create temporary subdirs so assemble_panel can glob correctly
                    coupon_subdir = processed_dir / f"db1b_{label}"
                    coupon_subdir.mkdir(exist_ok=True)
                    for f in sub.glob(f"db1b_{label}_*.parquet"):
                        link = coupon_subdir / f.name
                        if not link.exists():
                            link.symlink_to(f.resolve())
                    assemble_panel(processed_dir / f"db1b_{label}", panels_dir,
                                   f"db1b_{label}", chunk_size, logger)
            else:
                assemble_panel(processed_dir / ds_key, panels_dir,
                               ds_key, chunk_size, logger)

        # ── 4. Build O-D market panel ──────────────────────────────────────
        if "db1b" in selected_datasets:
            build_od_market_panel(panels_dir, chunk_size, logger)
            aggregate_market_shares(panels_dir, logger)

        # ── 5. T-100 frequency summary ─────────────────────────────────────
        if "t100" in selected_datasets:
            t100_panel = panels_dir / "t100_panel.parquet"
            if t100_panel.exists():
                freq_out = panels_dir / "t100_frequency_summary.parquet"
                if not freq_out.exists():
                    logger.info("Building T-100 frequency summary …")
                    t100 = pd.read_parquet(t100_panel)
                    freq = (
                        t100.groupby(["YEAR", "QUARTER", "UNIQUE_CARRIER", "ORIGIN", "DEST"])
                        .agg(
                            DeparturesPerformed=("DEPARTURES_PERFORMED", "sum"),
                            SeatsTotal=("SEATS", "sum"),
                            PassengersTotal=("PASSENGERS", "sum"),
                            AvgDistance=("DISTANCE", "mean"),
                        )
                        .reset_index()
                    )
                    freq["LoadFactor"] = freq["PassengersTotal"] / freq["SeatsTotal"].replace(0, pd.NA)
                    freq.to_parquet(freq_out, index=False, compression="snappy")
                    logger.info(f"  T-100 frequency summary: {len(freq):,} rows")

    logger.info("=" * 60)
    logger.info("Done. Output structure:")
    logger.info(f"  {output_dir}/raw/           — raw ZIP downloads")
    logger.info(f"  {output_dir}/processed/     — per-quarter Parquet files")
    logger.info(f"  {output_dir}/panels/        — assembled analysis panels")
    logger.info(f"  {output_dir}/download_cache.json — download manifest")
    logger.info(f"  {output_dir}/collector.log  — full run log")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()