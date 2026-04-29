#!/usr/bin/env python3
"""
download_t100.py — Download BTS T-100 Domestic Segment and Market data.

HOW BTS DOWNLOAD WORKS
-----------------------
T-100 data is NOT available as static prezipped files. The correct mechanism is:
  1. GET the DL_SelectFields.aspx page to obtain ASP.NET form tokens
     (__VIEWSTATE, __VIEWSTATEGENERATOR, __EVENTVALIDATION)
  2. POST those tokens back with cboYear, cboPeriod=All, and field checkboxes
     to the same page — which returns the zip file directly in the response body.

Table URLs:
  Segment : https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIM
  Market  : https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIL

Data is released monthly with a ~3-month lag. We download one full year per
request (cboPeriod=All), which is the most efficient unit BTS supports.

Usage
-----
  python download_t100.py --start-year 2024
  python download_t100.py --start-year 2019 --end-year 2023 --tables segment
  python download_t100.py --start-year 2024 --skip-cache

Arguments
---------
  --start-year INT    First year to download (inclusive). Default: 2019
  --end-year   INT    Last year to download (inclusive). Default: current year
  --tables     STR    segment | market | both. Default: both
  --out-dir    PATH   Parquet output dir. Default: ./parquet/t100
  --raw-dir    PATH   Raw zip dir. Default: ./t100
  --skip-cache FLAG   Force re-download
  --workers    INT    Parallel threads. Default: 1 (BTS blocks concurrent POSTs)
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import extract_zip, get_logger, mark_cached, save_parquet

# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------

TABLE_META = {
    "segment": {
        "label": "T100_Segment",
        "url": "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIM&QO_fu146_anzr=Nv4+Pn44vr45",
    },
    "market": {
        "label": "T100_Market",
        "url": "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIL&QO_fu146_anzr=Nv4+Pn44vr45",
    },
}

# ---------------------------------------------------------------------------
# Columns to keep after download
# ---------------------------------------------------------------------------

SEGMENT_COLS = {
    "YEAR", "QUARTER", "MONTH",
    "ORIGIN", "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR", "ORIGIN_WAC",
    "DEST",   "DEST_CITY_NAME",   "DEST_STATE_ABR",   "DEST_WAC",
    "AIRLINE_ID", "UNIQUE_CARRIER", "UNIQUE_CARRIER_NAME",
    "CARRIER", "CARRIER_NAME", "CARRIER_GROUP_NEW",
    "AIRCRAFT_TYPE", "CLASS",
    "PASSENGERS", "SEATS", "FREIGHT",
    "DEPARTURES_SCHEDULED", "DEPARTURES_PERFORMED",
    "PAYLOAD", "DISTANCE", "RAMP_TO_RAMP", "AIR_TIME",
}

MARKET_COLS = {
    "YEAR", "QUARTER", "MONTH",
    "ORIGIN", "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR", "ORIGIN_WAC",
    "DEST",   "DEST_CITY_NAME",   "DEST_STATE_ABR",   "DEST_WAC",
    "AIRLINE_ID", "UNIQUE_CARRIER", "UNIQUE_CARRIER_NAME",
    "CARRIER", "CARRIER_NAME", "CARRIER_GROUP_NEW", "CLASS",
    "PASSENGERS", "FREIGHT", "MAIL", "DISTANCE", "DISTANCE_GROUP",
}

TABLE_COLS = {"segment": SEGMENT_COLS, "market": MARKET_COLS}

DTYPE_HINTS = {
    "YEAR": "int16", "QUARTER": "int8", "MONTH": "int8",
    "PASSENGERS": "int32", "SEATS": "int32",
    "DEPARTURES_SCHEDULED": "float32", "DEPARTURES_PERFORMED": "float32",
    "DISTANCE": "float32", "DISTANCE_GROUP": "int8",
    "RAMP_TO_RAMP": "float32", "AIR_TIME": "float32",
    "PAYLOAD": "float32", "FREIGHT": "float32", "MAIL": "float32",
    "CARRIER_GROUP_NEW": "int8", "ORIGIN_WAC": "int16", "DEST_WAC": "int16",
}

# All field checkbox names for a full download (both tables share most names)
ALL_FIELDS = [
    "DEPARTURES_SCHEDULED", "DEPARTURES_PERFORMED", "PAYLOAD", "SEATS",
    "PASSENGERS", "FREIGHT", "MAIL", "RAMP_TO_RAMP", "AIR_TIME",
    "UNIQUE_CARRIER", "AIRLINE_ID", "UNIQUE_CARRIER_NAME",
    "UNIQUE_CARRIER_ENTITY", "REGION", "CARRIER", "CARRIER_NAME",
    "CARRIER_GROUP", "CARRIER_GROUP_NEW",
    "ORIGIN_AIRPORT_ID", "ORIGIN_AIRPORT_SEQ_ID", "ORIGIN_CITY_MARKET_ID",
    "ORIGIN", "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR", "ORIGIN_STATE_FIPS",
    "ORIGIN_STATE_NM", "ORIGIN_WAC",
    "DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID", "DEST_CITY_MARKET_ID",
    "DEST", "DEST_CITY_NAME", "DEST_STATE_ABR", "DEST_STATE_FIPS",
    "DEST_STATE_NM", "DEST_WAC",
    "AIRCRAFT_GROUP", "AIRCRAFT_TYPE", "AIRCRAFT_CONFIG",
    "YEAR", "QUARTER", "MONTH", "DISTANCE", "DISTANCE_GROUP",
    "CLASS", "DATA_SOURCE",
]

logger = get_logger("t100", "t100.log")

# ---------------------------------------------------------------------------
# ASP.NET token scraper
# ---------------------------------------------------------------------------

_VS_RE   = re.compile(r'id="__VIEWSTATE"\s+value="([^"]*)"')
_GEN_RE  = re.compile(r'id="__VIEWSTATEGENERATOR"\s+value="([^"]*)"')
_EV_RE   = re.compile(r'id="__EVENTVALIDATION"\s+value="([^"]*)"')


def _scrape_tokens(session: requests.Session, url: str) -> dict:
    """GET a BTS download page and return its hidden ASP.NET form tokens."""
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    html = resp.text

    def _find(pattern, name):
        m = pattern.search(html)
        if not m:
            raise ValueError(
                f"Could not find {name} in BTS form page. "
                "The page layout may have changed — check the URL manually."
            )
        return m.group(1)

    return {
        "__VIEWSTATE":          _find(_VS_RE,  "__VIEWSTATE"),
        "__VIEWSTATEGENERATOR": _find(_GEN_RE, "__VIEWSTATEGENERATOR"),
        "__EVENTVALIDATION":    _find(_EV_RE,  "__EVENTVALIDATION"),
    }


# ---------------------------------------------------------------------------
# Core download + process function
# ---------------------------------------------------------------------------

def _download_year(
    year: int,
    table: str,
    raw_dir: Path,
    out_dir: Path,
    skip_cache: bool,
) -> Optional[Path]:
    """Download one (year, table) via BTS form POST and save as Parquet."""
    meta         = TABLE_META[table]
    label        = meta["label"]
    page_url     = meta["url"]
    zip_path     = raw_dir / f"{label}_{year}.zip"
    parquet_path = out_dir / table / f"{year}.parquet"

    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP (parquet exists) %s %d", label, year)
        return parquet_path

    # Each call gets its own session so cookies don't collide across parallel workers
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })

    # Step 1 — scrape tokens
    logger.info("Fetching form tokens for %s %d …", label, year)
    try:
        tokens = _scrape_tokens(session, page_url)
    except Exception as exc:
        logger.error("Token scrape failed %s %d: %s", label, year, exc)
        return None

    # Step 2 — build POST body (tokens + year + all field checkboxes)
    post_data: dict = {
        "__EVENTTARGET":        "",
        "__EVENTARGUMENT":      "",
        "__LASTFOCUS":          "",
        "__VIEWSTATE":          tokens["__VIEWSTATE"],
        "__VIEWSTATEGENERATOR": tokens["__VIEWSTATEGENERATOR"],
        "__EVENTVALIDATION":    tokens["__EVENTVALIDATION"],
        "txtSearch":            "",
        "cboGeography":         "All",
        "cboYear":              str(year),
        "cboPeriod":            "All",
        "btnDownload":          "Download",
    }
    for field in ALL_FIELDS:
        post_data[field] = "on"

    # Step 3 — POST and stream to disk
    logger.info("Requesting %s %d …", label, year)
    try:
        resp = session.post(
            page_url,
            data=post_data,
            headers={
                "Referer":      page_url,
                "Origin":       "https://www.transtats.bts.gov",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            logger.warning(
                "SKIP %s %d — BTS returned an HTML page instead of a ZIP. "
                "Year is likely not yet published.", label, year
            )
            return None

        raw_dir.mkdir(parents=True, exist_ok=True)
        tmp = zip_path.with_suffix(".tmp")
        total = int(resp.headers.get("content-length", 0)) or None
        first_bytes = b""

        with open(tmp, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=f"  {label} {year}", leave=False,
        ) as bar:
            for i, chunk in enumerate(resp.iter_content(chunk_size=1 << 20)):
                if i == 0:
                    first_bytes = chunk[:8]
                fh.write(chunk)
                bar.update(len(chunk))

        # Validate ZIP magic bytes PK\x03\x04
        if not first_bytes.startswith(b"PK\x03\x04"):
            logger.warning(
                "SKIP %s %d — response is not a ZIP (magic: %s). "
                "Year probably not published yet.",
                label, year, first_bytes[:4].hex()
            )
            tmp.unlink(missing_ok=True)
            return None

        tmp.rename(zip_path)
        mark_cached(f"{page_url}|year={year}", zip_path)
        logger.info("Downloaded %s %d  (%.1f MB)", label, year,
                    zip_path.stat().st_size / 1e6)

    except Exception as exc:
        logger.error("POST failed %s %d: %s", label, year, exc)
        return None

    # Step 4 — extract CSV
    try:
        extracted = extract_zip(zip_path, raw_dir / "extracted", logger=logger)
    except Exception as exc:
        logger.error("Extract failed %s: %s", zip_path.name, exc)
        return None

    csv_files = [p for p in extracted if p.suffix.lower() == ".csv"]
    if not csv_files:
        logger.error("No CSV in %s", zip_path.name)
        return None

    # Step 5 — read, filter, save
    desired_cols = TABLE_COLS[table]
    chunks = []
    try:
        reader = pd.read_csv(
            csv_files[0],
            usecols=lambda c: c.upper().strip() in desired_cols,
            dtype={k: v for k, v in DTYPE_HINTS.items()},
            chunksize=500_000,
            low_memory=True,
            na_values=["", " "],
        )
        for chunk in tqdm(reader, desc=f"  Parsing {label} {year}", leave=False):
            chunk.columns = [c.upper().strip() for c in chunk.columns]
            if "CLASS" in chunk.columns:
                chunk = chunk[chunk["CLASS"].isin(["F", "G", "L", "Y", "C", "J", "K"])]
            chunks.append(chunk)
    except Exception as exc:
        logger.error("CSV read failed %s %d: %s", label, year, exc)
        return None

    if not chunks:
        logger.warning("Empty result %s %d", label, year)
        return None

    df = pd.concat(chunks, ignore_index=True)

    if table == "segment":
        if "DEPARTURES_PERFORMED" in df.columns:
            df["FREQ_WEEKLY"] = (
                df["DEPARTURES_PERFORMED"] / 4.33
            ).round(1).astype("float32")
        if "SEATS" in df.columns and "PASSENGERS" in df.columns:
            df["LOAD_FACTOR"] = (
                df["PASSENGERS"] / df["SEATS"].replace(0, float("nan"))
            ).round(4).astype("float32")

    save_parquet(df, parquet_path, logger=logger)

    try:
        csv_files[0].unlink()
    except OSError:
        pass

    del df, chunks
    return parquet_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    current_year = datetime.now().year
    p = argparse.ArgumentParser(
        description="Download BTS T-100 data via form POST and save as Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start-year", type=int, default=2019)
    p.add_argument("--end-year",   type=int, default=current_year)
    p.add_argument("--tables", nargs="+", default=["segment", "market"],
                   choices=["segment", "market"])
    p.add_argument("--out-dir", type=Path, default=Path("parquet/t100"))
    p.add_argument("--raw-dir", type=Path, default=Path("t100"))
    p.add_argument("--skip-cache", action="store_true")
    p.add_argument(
        "--workers", type=int, default=1,
        help="Parallel download threads. Keep at 1 unless you know BTS allows it.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        logger.error("--start-year must be ≤ --end-year")
        sys.exit(1)

    now = datetime.now()
    available_end = now.year if now.month > 3 else now.year - 1
    end = min(args.end_year, available_end)

    tasks = [
        (year, table)
        for year in range(args.start_year, end + 1)
        for table in args.tables
    ]
    if not tasks:
        logger.info("No years to download.")
        return

    logger.info("Downloading %d T-100 file(s): years %d–%d, tables %s",
                len(tasks), args.start_year, end, args.tables)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    results: list[Optional[Path]] = []
    with tqdm(total=len(tasks), desc="T-100 overall", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    _download_year, year, table,
                    args.raw_dir, args.out_dir,
                    args.skip_cache,
                ): (year, table)
                for (year, table) in tasks
            }
            for future in as_completed(future_map):
                yr, tbl = future_map[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "OK" if result else "SKIP/FAIL"
                except Exception as exc:
                    logger.error("Unhandled %d %s: %s", yr, tbl, exc)
                    results.append(None)
                    status = "ERROR"
                pbar.set_postfix_str(f"{yr} {tbl} → {status}")
                pbar.update(1)
                time.sleep(2)   # polite delay

    n_ok = sum(1 for r in results if r is not None)
    logger.info("Done. %d/%d succeeded.", n_ok, len(results))
    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
