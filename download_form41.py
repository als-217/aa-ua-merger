#!/usr/bin/env python3
"""
download_form41.py — Download BTS Form 41 financial data.

HOW BTS DOWNLOAD WORKS
-----------------------
Form 41 data (like T-100) requires a two-step form POST — there are no static
prezipped files at the URLs used in the original script.

  1. GET  https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=<ID>
     to scrape __VIEWSTATE / __VIEWSTATEGENERATOR / __EVENTVALIDATION tokens.
  2. POST those tokens back with cboYear, cboPeriod=All, field checkboxes
     → BTS returns a zip containing a CSV.

Form 41 Schedule Table IDs (gnoyr_VQ codes)
---------------------------------------------
  P-12  Fuel Cost & Consumption        FMF
  P-5.2 Aircraft Operating Expenses    FME  (Group II & III)
  P-7   Operating Expenses (Func.)     FMG
  P-1.2 Profit & Loss (Group I+,II,III) FMD

NOTE: Some Form 41 schedules (P-12a, P-1a, B-43) are listed as "restricted
public" on the DOT data portal and may require a registered BTS account for
download. The P-12 aggregate fuel cost schedule (gnoyr_VQ=FMF) is publicly
accessible and covers the key cost-instrument variable (fuel cost per gallon
by carrier-month). The P-7 and P-5.2 operating expense schedules are also
publicly downloadable.

Usage
-----
  python download_form41.py --start-year 2024
  python download_form41.py --start-year 2019 --end-year 2023 --tables p12 p52
  python download_form41.py --start-year 2019 --skip-cache

Arguments
---------
  --start-year INT    First year (inclusive). Default: 2019
  --end-year   INT    Last year (inclusive). Default: current year
  --tables     STR…   p12 | p52 | p7 | p1 | all. Default: p12 p52
  --out-dir    PATH   Parquet output dir. Default: ./parquet/form41
  --raw-dir    PATH   Raw zip dir. Default: ./form41
  --skip-cache FLAG   Force re-download
  --workers    INT    Parallel threads. Default: 1
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
# Table definitions: gnoyr_VQ codes confirmed from BTS Fields.asp pages
# ---------------------------------------------------------------------------

TABLE_META = {
    # P-12: Fuel Cost and Consumption by carrier-month — PRIMARY cost instrument
    "p12": {
        "gnoyr_VQ": "FMF",
        "label": "Form41_P12_Fuel",
        "description": "Schedule P-12: Fuel Cost and Consumption",
    },
    # P-5.2: Aircraft Operating Expenses (Group II & III carriers)
    "p52": {
        "gnoyr_VQ": "FME",
        "label": "Form41_P52_AircraftOpEx",
        "description": "Schedule P-5.2: Aircraft Operating Expenses",
    },
    # P-7: Operating Expenses by Functional Grouping
    "p7": {
        "gnoyr_VQ": "FMG",
        "label": "Form41_P7_OpExpFunc",
        "description": "Schedule P-7: Operating Expenses by Function",
    },
    # P-1.2: Profit & Loss Statement (Group I+, II & III)
    "p1": {
        "gnoyr_VQ": "FMD",
        "label": "Form41_P1_ProfitLoss",
        "description": "Schedule P-1.2: Profit & Loss Statement",
    },
}

BASE_URL = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ={gnoyr_VQ}&QO_fu146_anzr=Nv4+Pn44vr45"

# ---------------------------------------------------------------------------
# Columns to retain per schedule
# ---------------------------------------------------------------------------

P12_COLS = {
    "YEAR", "MONTH", "UNIQUE_CARRIER", "UNIQUE_CARRIER_NAME",
    "CARRIER_GROUP_NEW", "REGION",
    # Domestic fuel
    "TDOMT_COST", "TDOMT_GALLONS", "TDOMT_CPC",
    # International fuel
    "TINT_COST", "TINT_GALLONS", "TINT_CPC",
    # System total
    "SYSDOMS_COST", "SYSDOMS_GALLONS",
}

P52_COLS = {
    "YEAR", "MONTH", "UNIQUE_CARRIER", "UNIQUE_CARRIER_NAME",
    "CARRIER_GROUP_NEW", "REGION",
    "FLYING_OPS", "MAINTENANCE", "PAX_SVC", "AIRCRAFT_SVCING",
    "TRAFFIC_SVCING", "RESERVATIONS", "ADM_GENERAL", "DEPR_AMORT",
    "TRANSPORT_REL", "TOTAL_OP_EXP",
}

P7_COLS = {
    "YEAR", "MONTH", "UNIQUE_CARRIER", "UNIQUE_CARRIER_NAME",
    "CARRIER_GROUP_NEW", "REGION",
    "OP_REVENUES", "OP_EXPENSES", "NET_INCOME",
    "TRANS_REV_PAX", "TRANS_REV_CARGO",
    "LABOR", "FUEL_OIL", "MATERIALS",
    "AGENT_FEES", "LANDING_FEES", "RENTALS",
    "DEPR_AMORT", "OTHER",
    "PAX_REVENUE_MILES", "AVAIL_SEAT_MILES", "PAX_ENPLANED",
}

P1_COLS = {
    "YEAR", "MONTH", "UNIQUE_CARRIER", "UNIQUE_CARRIER_NAME",
    "CARRIER_GROUP_NEW",
    "OP_REVENUES", "OP_EXPENSES", "NET_INCOME",
    "TOT_ASSETS", "TOT_LIABILITIES", "NET_WORTH",
    "CURR_ASSETS", "LONG_ASSETS", "CURR_LIABILITIES", "LONG_DEBT",
}

TABLE_COLS = {"p12": P12_COLS, "p52": P52_COLS, "p7": P7_COLS, "p1": P1_COLS}

DTYPE_HINTS = {
    "YEAR": "int16", "MONTH": "int8", "CARRIER_GROUP_NEW": "int8",
}

logger = get_logger("form41", "form41.log")

# ---------------------------------------------------------------------------
# ASP.NET token scraper (shared pattern with T-100)
# ---------------------------------------------------------------------------

_VS_RE  = re.compile(r'id="__VIEWSTATE"\s+value="([^"]*)"')
_GEN_RE = re.compile(r'id="__VIEWSTATEGENERATOR"\s+value="([^"]*)"')
_EV_RE  = re.compile(r'id="__EVENTVALIDATION"\s+value="([^"]*)"')


def _scrape_tokens(session: requests.Session, url: str) -> dict:
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    html = resp.text

    def _find(pattern, name):
        m = pattern.search(html)
        if not m:
            raise ValueError(
                f"Could not find {name} on BTS form page at {url}. "
                "Check the URL / gnoyr_VQ code is still valid."
            )
        return m.group(1)

    return {
        "__VIEWSTATE":          _find(_VS_RE,  "__VIEWSTATE"),
        "__VIEWSTATEGENERATOR": _find(_GEN_RE, "__VIEWSTATEGENERATOR"),
        "__EVENTVALIDATION":    _find(_EV_RE,  "__EVENTVALIDATION"),
    }


def _get_all_field_names(session: requests.Session, url: str) -> list[str]:
    """
    Scrape the checkbox input names from the BTS download form.
    This avoids hardcoding field lists that may change between schedules.
    """
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    # Match all checkbox inputs: <input type="checkbox" name="FIELD_NAME" ...>
    fields = re.findall(
        r'<input[^>]+type=["\']checkbox["\'][^>]+name=["\']([A-Z0-9_]+)["\']',
        resp.text, re.IGNORECASE
    )
    return list(dict.fromkeys(fields))   # deduplicate, preserve order


# ---------------------------------------------------------------------------
# Core download function
# ---------------------------------------------------------------------------

def _download_table_year(
    year: int,
    table: str,
    raw_dir: Path,
    out_dir: Path,
    skip_cache: bool,
) -> Optional[Path]:
    meta         = TABLE_META[table]
    label        = meta["label"]
    gnoyr        = meta["gnoyr_VQ"]
    page_url     = BASE_URL.format(gnoyr_VQ=gnoyr)
    zip_path     = raw_dir / f"{label}_{year}.zip"
    parquet_path = out_dir / table / f"{year}.parquet"

    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP (parquet exists) %s %d", label, year)
        return parquet_path

    # Own session per call (avoids ASP.NET cookie collisions)
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

    # Step 1 — scrape tokens and field names together in a single GET
    logger.info("Fetching form for %s %d …", label, year)
    try:
        tokens = _scrape_tokens(session, page_url)
        fields = _get_all_field_names(session, page_url)
    except Exception as exc:
        logger.error("Form scrape failed %s %d: %s", label, year, exc)
        return None

    if not fields:
        logger.warning(
            "No checkbox fields found on %s — schedule may require login "
            "or the gnoyr_VQ code (%s) may have changed.", page_url, gnoyr
        )
        # Fall back to a known minimal set so we can still attempt the download
        fields = ["YEAR", "MONTH", "UNIQUE_CARRIER", "UNIQUE_CARRIER_NAME"]

    # Step 2 — POST
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
    for field in fields:
        post_data[field] = "on"

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
                "SKIP %s %d — BTS returned HTML. "
                "Year may not be available or schedule requires a BTS account.",
                label, year
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

        if not first_bytes.startswith(b"PK\x03\x04"):
            logger.warning(
                "SKIP %s %d — response is not a ZIP (magic: %s). "
                "Schedule may require a BTS login. "
                "Try downloading manually from: %s",
                label, year, first_bytes[:4].hex(), page_url
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

    # Step 3 — extract
    try:
        extracted = extract_zip(zip_path, raw_dir / "extracted", logger=logger)
    except Exception as exc:
        logger.error("Extract failed %s: %s", zip_path.name, exc)
        return None

    csv_files = [p for p in extracted if p.suffix.lower() == ".csv"]
    if not csv_files:
        logger.error("No CSV in %s", zip_path.name)
        return None

    # Step 4 — read + filter columns + save
    desired_cols = TABLE_COLS[table]
    chunks = []
    try:
        reader = pd.read_csv(
            csv_files[0],
            usecols=lambda c: c.upper().strip() in desired_cols,
            dtype={k: v for k, v in DTYPE_HINTS.items()},
            chunksize=200_000,
            low_memory=True,
            na_values=["", " ", "N/A"],
        )
        for chunk in tqdm(reader, desc=f"  Parsing {label} {year}", leave=False):
            chunk.columns = [c.upper().strip() for c in chunk.columns]
            chunks.append(chunk)
    except Exception as exc:
        logger.error("CSV read failed %s %d: %s", label, year, exc)
        return None

    if not chunks:
        logger.warning("Empty result %s %d", label, year)
        return None

    df = pd.concat(chunks, ignore_index=True)

    # Derived variables
    if table == "p12":
        if "TDOMT_COST" in df.columns and "TDOMT_GALLONS" in df.columns:
            df["FUEL_COST_PER_GAL_DOM"] = (
                df["TDOMT_COST"] / df["TDOMT_GALLONS"].replace(0, float("nan"))
            ).astype("float32")
            med = df["FUEL_COST_PER_GAL_DOM"].median()
            logger.info("%d %s: median domestic $/gal = %.2f", year, label, med)
            if not (0.5 < med < 10):
                logger.warning(
                    "Suspicious fuel price median %.2f for %d — check units.", med, year
                )

    if table == "p7":
        if "OP_EXPENSES" in df.columns and "AVAIL_SEAT_MILES" in df.columns:
            df["CASM"] = (
                df["OP_EXPENSES"] / df["AVAIL_SEAT_MILES"].replace(0, float("nan"))
            ).astype("float32")

    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")

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
        description="Download BTS Form 41 financial data via form POST.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start-year", type=int, default=2019)
    p.add_argument("--end-year",   type=int, default=current_year)
    p.add_argument("--tables", nargs="+", default=["p12", "p52"],
                   choices=list(TABLE_META.keys()) + ["all"])
    p.add_argument("--out-dir", type=Path, default=Path("parquet/form41"))
    p.add_argument("--raw-dir", type=Path, default=Path("form41"))
    p.add_argument("--skip-cache", action="store_true")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel threads (keep at 1 for BTS rate limits)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if "all" in args.tables:
        args.tables = list(TABLE_META.keys())

    if args.start_year > args.end_year:
        logger.error("--start-year must be ≤ --end-year")
        sys.exit(1)

    # Form 41 lags ~6 months
    available_end = datetime.now().year - 1
    end = min(args.end_year, available_end)

    tasks = [
        (year, table)
        for year in range(args.start_year, end + 1)
        for table in args.tables
        if table in TABLE_META
    ]
    if not tasks:
        logger.info("No tasks — check year range or table names.")
        return

    logger.info("Downloading %d Form 41 file(s): years %d–%d, tables %s",
                len(tasks), args.start_year, end, args.tables)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    results: list[Optional[Path]] = []
    with tqdm(total=len(tasks), desc="Form 41 overall", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    _download_table_year, year, table,
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
                time.sleep(2)

    n_ok = sum(1 for r in results if r is not None)
    logger.info("Done. %d/%d succeeded.", n_ok, len(results))
    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
