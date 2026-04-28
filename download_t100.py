#!/usr/bin/env python3
"""
download_t100.py — Download BTS T-100 Air Carrier Statistics data.

T-100 contains monthly non-stop segment data (capacity, departures, load) and
market-level passenger data. Both tables are essential for:
  - Flight frequency (departures per week) — product characteristic in BLP
  - Seat capacity / load factor — instrument candidate
  - Carrier presence per route — ownership matrix construction

Usage
-----
  python download_t100.py --start-year 2024
  python download_t100.py --start-year 2018 --end-year 2023 --tables segment
  python download_t100.py --start-year 2024 --skip-cache

Arguments
---------
  --start-year INT    First year to download (inclusive). Default: 2019
  --end-year   INT    Last year to download (inclusive). Default: current year
  --tables     STR…   segment | market | both. Default: both
  --out-dir    PATH   Output directory for Parquet files. Default: ./parquet/t100
  --raw-dir    PATH   Directory for raw zips. Default: ./t100
  --skip-cache FLAG   Force re-download even if cached
  --workers    INT    Parallel threads. Default: 2
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import download_file, extract_zip, get_logger, save_parquet

# ---------------------------------------------------------------------------
# URL pattern — BTS stores annual T-100 prezipped files
# Segment: T_T100D_SEGMENT_US_CARRIER_ONLY
# Market:  T_T100D_MARKET_US_CARRIER_ONLY
# ---------------------------------------------------------------------------

BASE_URL_SEGMENT = (
    "https://transtats.bts.gov/PREZIP/T_T100D_SEGMENT_US_CARRIER_ONLY_{year}.zip"
)
BASE_URL_MARKET = (
    "https://transtats.bts.gov/PREZIP/T_T100D_MARKET_US_CARRIER_ONLY_{year}.zip"
)

# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------

SEGMENT_COLS = [
    "YEAR",
    "QUARTER",
    "MONTH",
    "ORIGIN",           # IATA airport
    "ORIGIN_CITY_NAME",
    "ORIGIN_STATE_ABR",
    "ORIGIN_WAC",
    "DEST",
    "DEST_CITY_NAME",
    "DEST_STATE_ABR",
    "DEST_WAC",
    "AIRLINE_ID",       # DOT numeric carrier ID (stable over time)
    "UNIQUE_CARRIER",   # carrier code (may change with rebranding)
    "UNIQUE_CARRIER_NAME",
    "CARRIER",          # IATA code
    "CARRIER_NAME",
    "AIRCRAFT_TYPE",    # DOT aircraft type code
    "CLASS",            # service class (F=First, J=Business, Y=Coach, etc.)
    "PASSENGERS",       # passengers transported
    "SEATS",            # available seats
    "FREIGHT",
    "DEPARTURES_SCHEDULED",
    "DEPARTURES_PERFORMED",
    "PAYLOAD",
    "DISTANCE",
    "RAMP_TO_RAMP",     # block hours — cost proxy
    "AIR_TIME",
]

MARKET_COLS = [
    "YEAR",
    "QUARTER",
    "MONTH",
    "ORIGIN",
    "ORIGIN_CITY_NAME",
    "ORIGIN_STATE_ABR",
    "ORIGIN_WAC",
    "DEST",
    "DEST_CITY_NAME",
    "DEST_STATE_ABR",
    "DEST_WAC",
    "AIRLINE_ID",
    "UNIQUE_CARRIER",
    "UNIQUE_CARRIER_NAME",
    "CARRIER",
    "CARRIER_NAME",
    "CARRIER_GROUP_NEW",  # 1=Majors, 2=Nationals, 3=LCC, etc.
    "CLASS",
    "PASSENGERS",
    "FREIGHT",
    "MAIL",
    "DISTANCE",
    "DISTANCE_GROUP",
]

TABLE_META = {
    "segment": {
        "url_template": BASE_URL_SEGMENT,
        "cols": SEGMENT_COLS,
        "label": "T100_Segment",
    },
    "market": {
        "url_template": BASE_URL_MARKET,
        "cols": MARKET_COLS,
        "label": "T100_Market",
    },
}

DTYPE_HINTS = {
    "YEAR": "int16",
    "QUARTER": "int8",
    "MONTH": "int8",
    "PASSENGERS": "int32",
    "SEATS": "int32",
    "DEPARTURES_SCHEDULED": "float32",
    "DEPARTURES_PERFORMED": "float32",
    "DISTANCE": "float32",
    "DISTANCE_GROUP": "int8",
    "RAMP_TO_RAMP": "float32",
    "AIR_TIME": "float32",
    "PAYLOAD": "float32",
    "FREIGHT": "float32",
    "MAIL": "float32",
    "CARRIER_GROUP_NEW": "int8",
    "ORIGIN_WAC": "int16",
    "DEST_WAC": "int16",
}

logger = get_logger("t100", "t100.log")


def _download_year(
    year: int,
    table: str,
    raw_dir: Path,
    out_dir: Path,
    session: requests.Session,
    skip_cache: bool,
) -> Optional[Path]:
    """Download one (year, table) combination and save as Parquet."""
    meta = TABLE_META[table]
    url = meta["url_template"].format(year=year)
    label = meta["label"]
    zip_path = raw_dir / f"{label}_{year}.zip"
    parquet_path = out_dir / table / f"{year}.parquet"

    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP (parquet exists) %s %d", label, year)
        return parquet_path

    try:
        download_file(url, zip_path, session=session, logger=logger, skip_cache=skip_cache)
    except Exception as exc:
        logger.error("Failed to download %s %d: %s", label, year, exc)
        return None

    try:
        extracted = extract_zip(zip_path, raw_dir / "extracted", logger=logger)
    except Exception as exc:
        logger.error("Failed to extract %s: %s", zip_path.name, exc)
        return None

    csv_files = [p for p in extracted if p.suffix.lower() == ".csv"]
    if not csv_files:
        logger.error("No CSV in %s", zip_path.name)
        return None
    csv_path = csv_files[0]

    cols = meta["cols"]
    chunks = []
    try:
        reader = pd.read_csv(
            csv_path,
            usecols=lambda c: c in cols,
            dtype={k: v for k, v in DTYPE_HINTS.items() if k in cols},
            chunksize=500_000,
            low_memory=True,
            na_values=["", " "],
        )
        for chunk in tqdm(reader, desc=f"  Reading {label} {year}", leave=False):
            # Keep only coach/passenger service classes; drop cargo-only
            if "CLASS" in chunk.columns:
                chunk = chunk[chunk["CLASS"].isin(["F", "G", "L", "Y", "C", "J", "K"])]
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)
    except Exception as exc:
        logger.error("Failed reading CSV %s: %s", csv_path.name, exc)
        return None

    # Derive useful columns
    if table == "segment" and "DEPARTURES_PERFORMED" in df.columns and "MONTH" in df.columns:
        # Weekly frequency approximation (departures per month / 4.33)
        df["FREQ_WEEKLY"] = (df["DEPARTURES_PERFORMED"] / 4.33).round(1).astype("float32")

    if "SEATS" in df.columns and "PASSENGERS" in df.columns:
        df["LOAD_FACTOR"] = (df["PASSENGERS"] / df["SEATS"].replace(0, float("nan"))).round(4)
        df["LOAD_FACTOR"] = df["LOAD_FACTOR"].astype("float32")

    save_parquet(df, parquet_path, logger=logger)

    try:
        csv_path.unlink()
    except OSError:
        pass

    del df, chunks
    return parquet_path


def parse_args() -> argparse.Namespace:
    current_year = datetime.now().year
    p = argparse.ArgumentParser(
        description="Download BTS T-100 data and save as Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start-year", type=int, default=2019)
    p.add_argument("--end-year", type=int, default=current_year)
    p.add_argument("--tables", nargs="+", default=["segment", "market"],
                   choices=["segment", "market"])
    p.add_argument("--out-dir", type=Path, default=Path("parquet/t100"))
    p.add_argument("--raw-dir", type=Path, default=Path("t100"))
    p.add_argument("--skip-cache", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        logger.error("--start-year must be ≤ --end-year")
        sys.exit(1)

    # T-100 is released with a ~3-month lag; limit end year accordingly
    available_end = datetime.now().year if datetime.now().month > 3 else datetime.now().year - 1

    tasks = [
        (year, table)
        for year in range(args.start_year, min(args.end_year, available_end) + 1)
        for table in args.tables
    ]

    if not tasks:
        logger.info("No years to download.")
        return

    logger.info("Downloading %d T-100 file(s) for years %d–%d, tables %s",
                len(tasks), args.start_year, args.end_year, args.tables)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; airline-research-bot/1.0)"
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    results: list[Optional[Path]] = []
    with tqdm(total=len(tasks), desc="T-100 overall", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    _download_year, year, table,
                    args.raw_dir, args.out_dir,
                    session, args.skip_cache,
                ): (year, table)
                for (year, table) in tasks
            }
            for future in as_completed(future_map):
                yr, tbl = future_map[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "OK" if result else "FAIL"
                except Exception as exc:
                    logger.error("Unhandled error %d %s: %s", yr, tbl, exc)
                    results.append(None)
                    status = "ERROR"
                pbar.set_postfix_str(f"{yr} {tbl} → {status}")
                pbar.update(1)

    n_ok = sum(1 for r in results if r is not None)
    logger.info("Done. %d/%d succeeded.", n_ok, len(results))
    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
