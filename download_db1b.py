#!/usr/bin/env python3
"""
download_db1b.py — Download BTS DB1B (Origin & Destination Survey) data.

DB1B is a 10 % random sample of domestic airline tickets. It contains three
linked tables: Coupon, Market, and Ticket. This script downloads quarterly
zip files from transtats.bts.gov, extracts the CSVs, retains only the
columns needed for BLP demand estimation of merger effects, and saves to
Parquet for fast downstream loading.

Usage
-----
  python download_db1b.py --start-year 2024
  python download_db1b.py --start-year 2018 --end-year 2023 --tables coupon market
  python download_db1b.py --start-year 2024 --quarters 1 2 --skip-cache

Arguments
---------
  --start-year INT    First year to download (inclusive). Default: 2019
  --end-year   INT    Last year to download (inclusive). Default: current year
  --quarters   INT…   Quarters to include (1–4). Default: all four
  --tables     STR…   Which DB1B tables: coupon, market, ticket. Default: all
  --out-dir    PATH   Output directory for Parquet files. Default: ./parquet/db1b
  --raw-dir    PATH   Directory to keep raw zips + CSVs. Default: ./db1b
  --skip-cache FLAG   Force re-download even if cached
  --workers    INT    Parallel download threads. Default: 2 (BTS rate-limits)
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

# Local shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    download_file,
    extract_zip,
    get_logger,
    is_cached,
    save_parquet,
)

# ---------------------------------------------------------------------------
# BTS prezipped-file URL pattern
# DB1B prezipped files live at a stable URL structure.
# Table codes:
#   Coupon → Origin_and_Destination_Survey_DB1BCoupon_<YYYY>_<Q>.zip
#   Market → Origin_and_Destination_Survey_DB1BMarket_<YYYY>_<Q>.zip
#   Ticket → Origin_and_Destination_Survey_DB1BTicket_<YYYY>_<Q>.zip
# ---------------------------------------------------------------------------

BASE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "Origin_and_Destination_Survey_DB1B{table}_{year}_{quarter}.zip"
)

TABLE_NAMES = {
    "coupon": "Coupon",
    "market": "Market",
    "ticket": "Ticket",
}

# ---------------------------------------------------------------------------
# Columns to keep per table  (all others are dropped after read to minimise
# peak RAM; the raw CSV is never held fully in memory).
# ---------------------------------------------------------------------------

COUPON_COLS = [
    "ItinID",          # join key → Market / Ticket
    "MktID",           # join key → Market
    "SeqNum",          # coupon sequence (determines nonstop vs. connecting)
    "Coupons",         # total coupons in itinerary → n_connections = Coupons-1
    "Year",
    "Quarter",
    "Origin",          # IATA airport code
    "OriginCityNum",   # city code (for city-pair market definition)
    "OriginState",
    "OriginWac",
    "Dest",
    "DestCityNum",
    "DestState",
    "DestWac",
    "Break",           # trip-break indicator
    "CouponType",
    "TkCarrier",       # ticketing carrier
    "OpCarrier",       # operating carrier
    "RPCarrier",       # reporting carrier
    "Passengers",
    "FareClass",
    "Distance",        # coupon distance (miles)
    "DistanceGroup",
    "Gateway",         # gateway indicator
]

MARKET_COLS = [
    "ItinID",
    "MktID",
    "MktCoupons",      # coupons in this O-D leg
    "Year",
    "Quarter",
    "Origin",
    "OriginCityNum",
    "OriginState",
    "OriginWac",
    "Dest",
    "DestCityNum",
    "DestState",
    "DestWac",
    "TkCarrier",
    "OpCarrier",
    "RPCarrier",
    "BulkFare",        # bulk-fare indicator (exclude from fare analysis)
    "Passengers",
    "MktFare",         # prorated market fare — primary price variable
    "MktDistance",
    "MktMilesFlown",
    "NonStopMiles",    # great-circle miles → used in instruments
]

TICKET_COLS = [
    "ItinID",
    "Coupons",
    "Year",
    "Quarter",
    "Origin",
    "OriginCityNum",
    "OriginState",
    "RoundTrip",
    "OnLine",
    "DollarCred",      # dollar-credibility flag — filter implausible fares
    "ItinYield",       # fare per mile (useful for outlier detection)
    "RPCarrier",
    "Passengers",
    "ItinFare",        # total itinerary fare
    "BulkFare",
    "Distance",
    "MilesFlown",
]

TABLE_COLS = {
    "coupon": COUPON_COLS,
    "market": MARKET_COLS,
    "ticket": TICKET_COLS,
}

# Dtype hints to speed up CSV parsing and reduce peak RAM
DTYPE_HINTS: dict[str, str] = {
    "ItinID": "int64",
    "MktID": "int64",
    "SeqNum": "int8",
    "Coupons": "int8",
    "MktCoupons": "int8",
    "Year": "int16",
    "Quarter": "int8",
    "Passengers": "int32",
    "Distance": "float32",
    "MktDistance": "float32",
    "MktMilesFlown": "float32",
    "NonStopMiles": "float32",
    "MktFare": "float32",
    "ItinFare": "float32",
    "ItinYield": "float32",
    "DistanceGroup": "int8",
    "Gateway": "int8",
    "RoundTrip": "int8",
    "OnLine": "int8",
    "DollarCred": "int8",
    "BulkFare": "int8",
}

# ---------------------------------------------------------------------------
# Core download + process function
# ---------------------------------------------------------------------------

logger = get_logger("db1b", "db1b.log")


def _download_quarter(
    year: int,
    quarter: int,
    table: str,
    raw_dir: Path,
    out_dir: Path,
    session: requests.Session,
    skip_cache: bool,
) -> Optional[Path]:
    """Download one (year, quarter, table) combination and save as Parquet."""
    tbl_name = TABLE_NAMES[table]
    url = BASE_URL.format(table=tbl_name, year=year, quarter=quarter)
    zip_path = raw_dir / f"DB1B{tbl_name}_{year}_Q{quarter}.zip"
    parquet_path = out_dir / table / f"{year}_Q{quarter}.parquet"

    # Check if final Parquet already exists
    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP (parquet exists) %s Q%d %s", year, quarter, table)
        return parquet_path

    # Download zip
    try:
        download_file(
            url, zip_path,
            session=session,
            logger=logger,
            skip_cache=skip_cache,
        )
    except Exception as exc:
        logger.error("Failed to download %s Q%d %s: %s", year, quarter, table, exc)
        return None

    # Extract — BTS zips contain a single CSV
    try:
        extracted = extract_zip(zip_path, raw_dir / "extracted", logger=logger)
    except Exception as exc:
        logger.error("Failed to extract %s: %s", zip_path.name, exc)
        return None

    csv_files = [p for p in extracted if p.suffix.lower() == ".csv"]
    if not csv_files:
        logger.error("No CSV found in %s", zip_path.name)
        return None
    csv_path = csv_files[0]

    # Read CSV in chunks to cap RAM usage
    cols = TABLE_COLS[table]
    chunks = []
    chunk_size = 500_000  # rows per chunk — safe for 64 GB RAM

    try:
        reader = pd.read_csv(
            csv_path,
            usecols=lambda c: c in cols,
            dtype={k: v for k, v in DTYPE_HINTS.items() if k in cols},
            chunksize=chunk_size,
            low_memory=True,
            na_values=["", " "],
        )
        for chunk in tqdm(reader, desc=f"  Reading {tbl_name} {year}Q{quarter}", leave=False):
            # Keep only domestic US records (OriginCountry / DestCountry == 'US')
            # Market table has country cols; filter silently if absent
            for country_col in ("OriginCountry", "DestCountry"):
                if country_col in chunk.columns:
                    chunk = chunk[chunk[country_col].isin(["US", "PR", "VI"])]
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)
    except Exception as exc:
        logger.error("Failed to read CSV %s: %s", csv_path.name, exc)
        return None

    # Save as Parquet
    save_parquet(df, parquet_path, logger=logger)

    # Remove extracted CSV to free disk space (keep the zip for cache validation)
    try:
        csv_path.unlink()
    except OSError:
        pass

    del df, chunks
    return parquet_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    current_year = datetime.now().year
    p = argparse.ArgumentParser(
        description="Download BTS DB1B data and save as Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start-year", type=int, default=2019, metavar="YEAR",
                   help="First year to download (inclusive)")
    p.add_argument("--end-year", type=int, default=current_year, metavar="YEAR",
                   help="Last year to download (inclusive)")
    p.add_argument("--quarters", type=int, nargs="+", default=[1, 2, 3, 4],
                   choices=[1, 2, 3, 4], metavar="Q",
                   help="Quarters to download (1–4)")
    p.add_argument("--tables", nargs="+", default=["coupon", "market", "ticket"],
                   choices=list(TABLE_NAMES.keys()),
                   help="DB1B tables to download")
    p.add_argument("--out-dir", type=Path, default=Path("parquet/db1b"),
                   help="Output directory for Parquet files")
    p.add_argument("--raw-dir", type=Path, default=Path("db1b"),
                   help="Directory for raw zips / CSVs")
    p.add_argument("--skip-cache", action="store_true",
                   help="Force re-download even if cached")
    p.add_argument("--workers", type=int, default=2,
                   help="Parallel download threads (keep ≤ 3 to respect BTS rate limits)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Validate year range
    if args.start_year > args.end_year:
        logger.error("--start-year must be ≤ --end-year")
        sys.exit(1)

    # Build task list
    tasks: list[tuple[int, int, str]] = []
    for year in range(args.start_year, args.end_year + 1):
        for quarter in args.quarters:
            # DB1B data lags ~6 months; skip future quarters
            current_q = (datetime.now().month - 1) // 3 + 1
            if year == datetime.now().year and quarter >= current_q:
                logger.debug("Skipping future quarter %d Q%d", year, quarter)
                continue
            for table in args.tables:
                tasks.append((year, quarter, table))

    if not tasks:
        logger.info("No quarters to download.")
        return

    logger.info(
        "Downloading %d file(s): years %d–%d, quarters %s, tables %s",
        len(tasks), args.start_year, args.end_year,
        args.quarters, args.tables,
    )

    # Shared session with browser-like headers (BTS requires these)
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (compatible; airline-research-bot/1.0; "
            "+https://github.com/als-217/aa-ua-merger)"
        ),
        "Accept-Encoding": "gzip, deflate",
    })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    results: list[Optional[Path]] = []

    with tqdm(total=len(tasks), desc="DB1B overall", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    _download_quarter,
                    year, quarter, table,
                    args.raw_dir, args.out_dir,
                    session, args.skip_cache,
                ): (year, quarter, table)
                for (year, quarter, table) in tasks
            }
            for future in as_completed(future_map):
                yr, qr, tbl = future_map[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "OK" if result else "FAIL"
                except Exception as exc:
                    logger.error("Unhandled error %d Q%d %s: %s", yr, qr, tbl, exc)
                    results.append(None)
                    status = "ERROR"
                pbar.set_postfix_str(f"{yr} Q{qr} {tbl} → {status}")
                pbar.update(1)

    n_ok = sum(1 for r in results if r is not None)
    n_fail = len(results) - n_ok
    logger.info("Done. %d succeeded, %d failed.", n_ok, n_fail)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
