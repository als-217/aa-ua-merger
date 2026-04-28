#!/usr/bin/env python3
"""
download_form41.py — Download BTS Form 41 financial data for structural cost estimation.

Form 41 is the DOT's mandatory financial reporting for US air carriers. For
merger analysis, the most relevant sub-tables are:

  P-12    Fuel cost and consumption by carrier-month  (Table ID 294)
  P-1.2   Balance sheet summary                        (Table ID 298)
  P-7     Operating expenses by carrier-year           (Table ID 41)
  P-5.2   Income statement                             (Table ID 256)
  Schedule P-52: Operating expenses by function       (Table ID 297)

This script downloads the annual / monthly prezipped CSV files, keeps the
columns needed for:
  - Recovering carrier-level marginal cost proxies
  - Constructing fuel cost instruments (price × distance)
  - Building cost-function controls for BLP supply side

Usage
-----
  python download_form41.py --start-year 2024
  python download_form41.py --start-year 2018 --end-year 2023
  python download_form41.py --start-year 2019 --tables p12 p52 --skip-cache

Arguments
---------
  --start-year INT    First year (inclusive). Default: 2019
  --end-year   INT    Last year (inclusive). Default: current year
  --tables     STR…   p12 | p52 | p7 | p1 | all. Default: p12 p52
  --out-dir    PATH   Parquet output dir. Default: ./parquet/form41
  --raw-dir    PATH   Raw zip dir. Default: ./form41
  --skip-cache FLAG   Force re-download
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
# BTS Form 41 prezipped file URLs
# These files are typically released annually with a ~6-month lag.
# ---------------------------------------------------------------------------

# Table definitions: name → (url_template, table_id, description)
# URL pattern: https://transtats.bts.gov/PREZIP/Form_41_Schedules_{CODE}_{YEAR}.zip
# where CODE varies by sub-schedule. We use the stable "prezipped" endpoint.

TABLE_REGISTRY: dict[str, dict] = {
    # P-12: Fuel Cost and Consumption — MOST IMPORTANT for cost instruments
    "p12": {
        "url": "https://transtats.bts.gov/PREZIP/Form_41_Schedules_P_12_{year}.zip",
        "label": "Form41_P12_Fuel",
        "freq": "annual",   # annual file covering all months
    },
    # P-52: Operating Expenses by Functional Category
    "p52": {
        "url": "https://transtats.bts.gov/PREZIP/Form_41_Schedules_P_52_{year}.zip",
        "label": "Form41_P52_OpEx",
        "freq": "annual",
    },
    # P-7: Summary of Operations (revenue, expenses, passengers at carrier level)
    "p7": {
        "url": "https://transtats.bts.gov/PREZIP/Form_41_Schedules_P_7_{year}.zip",
        "label": "Form41_P7_Ops",
        "freq": "annual",
    },
    # P-1.2: Balance Sheet — useful for fixed cost estimation
    "p1": {
        "url": "https://transtats.bts.gov/PREZIP/Form_41_Schedules_P_1_2_{year}.zip",
        "label": "Form41_P1_Balance",
        "freq": "annual",
    },
}

# ---------------------------------------------------------------------------
# Column selection per table
# ---------------------------------------------------------------------------

P12_COLS = [
    # Core fuel variables
    "YEAR",
    "MONTH",
    "UNIQUE_CARRIER",
    "UNIQUE_CARRIER_NAME",
    "CARRIER_GROUP_NEW",
    "REGION",
    # Fuel consumption and cost
    "TDOMT_COST",       # Total domestic fuel cost ($)
    "TDOMT_GALLONS",    # Total domestic fuel consumed (gallons)
    "TDOMT_CPC",        # Cost per gallon domestic
    "TINT_COST",        # International fuel cost
    "TINT_GALLONS",
    "TINT_CPC",
    # Derived domestic unit cost (computed below)
    # "TDOMT_CPC" is the key instrument variable
]

P52_COLS = [
    "YEAR",
    "MONTH",
    "UNIQUE_CARRIER",
    "UNIQUE_CARRIER_NAME",
    "CARRIER_GROUP_NEW",
    "REGION",
    # Major functional expense categories
    "OP_REVENUES",
    "OP_EXPENSES",
    "LABOR",            # Labor costs (wages + benefits)
    "MAT",              # Materials (includes fuel in some versions)
    "LANDING_FEES",     # Airport fees
    "RENTALS",          # Aircraft/facility rentals
    "DEPR_AMORT",       # Depreciation
    "OTHER",
    "TRANS_EXP",        # Transport-related expenses
]

P7_COLS = [
    "YEAR",
    "MONTH",
    "UNIQUE_CARRIER",
    "UNIQUE_CARRIER_NAME",
    "CARRIER_GROUP_NEW",
    "REGION",
    "PASS_REV",         # Passenger revenue
    "FREIGHT_REV",
    "OP_EXP",
    "TOT_REV",
    "NET_INCOME",
    "REV_PAX_MILES",    # Revenue passenger miles
    "AVAIL_SEAT_MILES", # Available seat miles (capacity measure)
    "PAX_ENPLANED",
]

P1_COLS = [
    "YEAR",
    "MONTH",
    "UNIQUE_CARRIER",
    "UNIQUE_CARRIER_NAME",
    "CARRIER_GROUP_NEW",
    "ASSETS",
    "CURR_ASSETS",
    "LONG_ASSETS",
    "LIABILITIES",
    "LONG_DEBT",
    "NET_WORTH",
]

TABLE_COLS = {
    "p12": P12_COLS,
    "p52": P52_COLS,
    "p7": P7_COLS,
    "p1": P1_COLS,
}

DTYPE_HINTS = {
    "YEAR": "int16",
    "MONTH": "int8",
    "CARRIER_GROUP_NEW": "int8",
    # All financial columns → float32 (sufficient precision for $M-scale values)
}

logger = get_logger("form41", "form41.log")


def _download_table_year(
    year: int,
    table: str,
    raw_dir: Path,
    out_dir: Path,
    session: requests.Session,
    skip_cache: bool,
) -> Optional[Path]:
    """Download one (year, table) combination and save as Parquet."""
    meta = TABLE_REGISTRY[table]
    url = meta["url"].format(year=year)
    label = meta["label"]
    zip_path = raw_dir / f"{label}_{year}.zip"
    parquet_path = out_dir / table / f"{year}.parquet"

    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP (parquet exists) %s %d", label, year)
        return parquet_path

    try:
        download_file(url, zip_path, session=session, logger=logger, skip_cache=skip_cache)
    except Exception as exc:
        logger.error("Download failed %s %d: %s", label, year, exc)
        return None

    try:
        extracted = extract_zip(zip_path, raw_dir / "extracted", logger=logger)
    except Exception as exc:
        logger.error("Extract failed %s: %s", zip_path.name, exc)
        return None

    csv_files = [p for p in extracted if p.suffix.lower() == ".csv"]
    if not csv_files:
        logger.error("No CSV in %s", zip_path.name)
        return None
    csv_path = csv_files[0]

    desired_cols = TABLE_COLS[table]
    chunks: list[pd.DataFrame] = []
    try:
        reader = pd.read_csv(
            csv_path,
            # Use flexible column matching: BTS sometimes adds underscores / caps
            usecols=lambda c: c.upper().replace(" ", "_") in [d.upper() for d in desired_cols],
            dtype={k: v for k, v in DTYPE_HINTS.items()},
            chunksize=200_000,
            low_memory=True,
            na_values=["", " ", "N/A"],
        )
        for chunk in tqdm(reader, desc=f"  Reading {label} {year}", leave=False):
            # Standardise column names to UPPER_SNAKE
            chunk.columns = [c.upper().replace(" ", "_") for c in chunk.columns]
            chunks.append(chunk)
    except Exception as exc:
        logger.error("Read failed %s: %s", csv_path.name, exc)
        return None

    if not chunks:
        logger.warning("Empty file: %s %d", label, year)
        return None

    df = pd.concat(chunks, ignore_index=True)

    # --- Derived variables ---
    if table == "p12":
        # Fuel cost per gallon (sanity-check: should be ~$2–$5 for domestic jet)
        if "TDOMT_COST" in df.columns and "TDOMT_GALLONS" in df.columns:
            df["FUEL_COST_PER_GAL"] = (
                df["TDOMT_COST"] / df["TDOMT_GALLONS"].replace(0, float("nan"))
            ).astype("float32")
            logger.debug(
                "%d %s: median fuel $/gal = %.2f",
                year, label,
                df["FUEL_COST_PER_GAL"].median()
            )

    if table == "p7":
        # Cost per available seat mile (CASM) — key supply-side variable
        if "OP_EXP" in df.columns and "AVAIL_SEAT_MILES" in df.columns:
            df["CASM"] = (
                df["OP_EXP"] / df["AVAIL_SEAT_MILES"].replace(0, float("nan"))
            ).astype("float32")

    # Downcast float64 columns
    for col in df.select_dtypes("float64").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

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
        description="Download BTS Form 41 financial data and save as Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start-year", type=int, default=2019)
    p.add_argument("--end-year", type=int, default=current_year)
    p.add_argument("--tables", nargs="+", default=["p12", "p52"],
                   choices=list(TABLE_REGISTRY.keys()) + ["all"])
    p.add_argument("--out-dir", type=Path, default=Path("parquet/form41"))
    p.add_argument("--raw-dir", type=Path, default=Path("form41"))
    p.add_argument("--skip-cache", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if "all" in args.tables:
        args.tables = list(TABLE_REGISTRY.keys())

    if args.start_year > args.end_year:
        logger.error("--start-year must be ≤ --end-year")
        sys.exit(1)

    # Form 41 lags ~6 months
    available_end = datetime.now().year - 1

    tasks = [
        (year, table)
        for year in range(args.start_year, min(args.end_year, available_end) + 1)
        for table in args.tables
        if table in TABLE_REGISTRY
    ]

    if not tasks:
        logger.info("No tasks — nothing to download.")
        return

    logger.info(
        "Downloading %d Form 41 file(s) — years %d–%d, tables %s",
        len(tasks), args.start_year, args.end_year, args.tables,
    )

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; airline-research-bot/1.0)"
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    results: list[Optional[Path]] = []
    with tqdm(total=len(tasks), desc="Form 41 overall", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    _download_table_year,
                    year, table,
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
