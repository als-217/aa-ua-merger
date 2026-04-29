#!/usr/bin/env python3
"""
download_supporting.py — Download supporting datasets for BLP merger analysis.

1. MSA Population:
   - OLD (broken): Census PEP API (api.census.gov) — PEP API dropped MSA-level
     data after 2020 vintage; wrong endpoint structure for 2020+ anyway.
   - NEW: Direct flat CSV from www2.census.gov with stable URL pattern:
     https://www2.census.gov/programs-surveys/popest/datasets/{vintage}/metro/
     totals/cbsa-est{vintage}-alldata.csv
     Each vintage file covers all years from 2020 to vintage year in wide format.

2. Jet Fuel Prices:
   - OLD (broken): EIA XLS download (eia.gov/dnav/pet — not a direct file URL)
   - NEW: FRED plain-text series endpoint (no API key needed):
     https://fred.stlouisfed.org/data/WJFUELUSGULF.txt  (weekly, spot price)

3. Airport / City Lookup:
   - OLD (broken): transtats.bts.gov/Download_Lookup.asp — requires session cookie
   - NEW: BTS lookup tables require the same ASP.NET POST pattern as T-100.
     Fallback: GitHub mirror of the stable lookup files (dannguyen/bts-transstats).

4. Aircraft Characteristics:
   - OLD (broken): FAA xlsx URL no longer resolves
   - NEW: BTS aircraft type lookup via form POST; fallback to GitHub mirror.

5. CPI Air Transportation:
   - OLD (broken): BLS API (api.bls.gov) — requires registered API key
   - NEW: FRED plain-text endpoints (no key needed):
     CPIAUCSL  — CPI All Items (for deflation)
     CUSR0000SETG01 is not on FRED; use BLS public data files instead:
     https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems  (all series)
     Filter for series CUSR0000SETG01 (CPI Air Transportation).

6. CBSA Delineation:
   - OLD (broken): www2.census.gov/programs-surveys/metro-micro — correct domain
     but wrong file path structure (the old URLs pointed to non-existent files)
   - NEW: Verified direct URL:
     https://www2.census.gov/programs-surveys/metro-micro/geographies/
     reference-files/2023/delineation-files/list1_2023.xlsx
     Plus fallback 2020 vintage.

Usage
-----
  python download_supporting.py
  python download_supporting.py --start-year 2019 --end-year 2024
  python download_supporting.py --datasets msa fuel --skip-cache

Arguments
---------
  --start-year  INT     First year for time-series data. Default: 2010
  --end-year    INT     Last year. Default: current year
  --datasets    STR…    msa | fuel | airport | aircraft | cpi | cbsa | all
  --out-dir     PATH    Parquet output. Default: ./parquet/supporting
  --raw-dir     PATH    Raw file cache. Default: ./supporting
  --skip-cache  FLAG    Force re-download
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import download_file, get_logger, save_parquet

logger = get_logger("supporting", "supporting.log")

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": _UA, "Accept-Encoding": "gzip, deflate"})
    return s


# ============================================================
# 1. MSA Population
# ============================================================
# Census PEP flat CSV — covers 2020-{vintage} in wide format.
# Each vintage supersedes prior ones, so we only need the latest
# for 2020+. For 2010-2019 we use the vintage-2019 file.

_PEP_2020S = (
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "{decade}/metro/totals/cbsa-est{vintage}-alldata.csv"
)
# Pre-2020: vintage 2019 covers 2010-2019
_PEP_2010S = (
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2010-2019/metro/totals/cbsa-est2019-alldata.csv"
)


def download_msa_population(
    start_year: int,
    end_year: int,
    raw_dir: Path,
    out_dir: Path,
    skip_cache: bool = False,
) -> Optional[Path]:
    parquet_path = out_dir / "msa_population.parquet"
    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP msa_population (parquet exists)")
        return parquet_path

    logger.info("Downloading MSA population estimates …")
    session = _session()
    frames: list[pd.DataFrame] = []

    # ── 2010–2019 vintage ───────────────────────────────────────────────────
    if start_year < 2020:
        raw_path = raw_dir / "cbsa-est2019-alldata.csv"
        try:
            download_file(_PEP_2010S, raw_path, session=session,
                          logger=logger, skip_cache=skip_cache)
            df = pd.read_csv(raw_path, encoding="latin-1", low_memory=False)
            df.columns = [c.upper() for c in df.columns]

            # Wide → long for years 2010-2019
            # Column names: POPESTIMATE2010 … POPESTIMATE2019
            id_cols = ["CBSA", "MDIV", "STCOU", "NAME", "LSAD"]
            id_cols = [c for c in id_cols if c in df.columns]
            # Keep only MSA-level rows (not county sub-rows)
            if "LSAD" in df.columns:
                df = df[df["LSAD"].isin([
                    "Metropolitan Statistical Area",
                    "Micropolitan Statistical Area",
                ])]
            pop_cols = [c for c in df.columns
                        if re.match(r"POPESTIMATE2\d{3}", c)]
            long = df[id_cols + pop_cols].melt(
                id_vars=id_cols,
                value_vars=pop_cols,
                var_name="YEAR_STR",
                value_name="POPULATION",
            )
            long["YEAR"] = long["YEAR_STR"].str.extract(r"(\d{4})").astype(int)
            long = long[(long["YEAR"] >= start_year) & (long["YEAR"] < 2020)]
            long = long.rename(columns={"CBSA": "CBSA_CODE", "NAME": "CBSA_NAME"})
            frames.append(long[["CBSA_CODE", "CBSA_NAME", "YEAR", "POPULATION"]])
            logger.info("Loaded 2010–2019 MSA population (%d rows)", len(long))
        except Exception as exc:
            logger.warning("2010–2019 MSA population failed: %s", exc)

    # ── 2020+ vintage (latest available) ────────────────────────────────────
    if end_year >= 2020:
        # Find the most recent available vintage (Census releases ~March each year)
        now = datetime.now()
        latest_vintage = now.year - 1 if now.month < 4 else now.year
        latest_vintage = min(latest_vintage, end_year)

        decade = "2020-" + str(latest_vintage)
        url = _PEP_2020S.format(decade=decade, vintage=latest_vintage)
        raw_path = raw_dir / f"cbsa-est{latest_vintage}-alldata.csv"

        # Try vintages backwards until one downloads successfully
        for vintage in range(latest_vintage, 2019, -1):
            decade = "2020-" + str(vintage)
            url = _PEP_2020S.format(decade=decade, vintage=vintage)
            raw_path = raw_dir / f"cbsa-est{vintage}-alldata.csv"
            try:
                download_file(url, raw_path, session=session,
                              logger=logger, skip_cache=skip_cache)
                break
            except Exception:
                logger.debug("Vintage %d not available, trying earlier …", vintage)
                raw_path.unlink(missing_ok=True)
        else:
            logger.error("Could not download any 2020+ MSA population vintage.")
            raw_path = None

        if raw_path and raw_path.exists():
            try:
                df = pd.read_csv(raw_path, encoding="latin-1", low_memory=False)
                df.columns = [c.upper() for c in df.columns]
                if "LSAD" in df.columns:
                    df = df[df["LSAD"].isin([
                        "Metropolitan Statistical Area",
                        "Micropolitan Statistical Area",
                    ])]
                id_cols = [c for c in ["CBSA", "MDIV", "STCOU", "NAME", "LSAD"]
                           if c in df.columns]
                pop_cols = [c for c in df.columns
                            if re.match(r"POPESTIMATE2\d{3}", c)]
                long = df[id_cols + pop_cols].melt(
                    id_vars=id_cols,
                    value_vars=pop_cols,
                    var_name="YEAR_STR",
                    value_name="POPULATION",
                )
                long["YEAR"] = long["YEAR_STR"].str.extract(r"(\d{4})").astype(int)
                long = long[
                    (long["YEAR"] >= max(start_year, 2020)) &
                    (long["YEAR"] <= end_year)
                ]
                long = long.rename(columns={"CBSA": "CBSA_CODE", "NAME": "CBSA_NAME"})
                frames.append(long[["CBSA_CODE", "CBSA_NAME", "YEAR", "POPULATION"]])
                logger.info("Loaded 2020+ MSA population (%d rows)", len(long))
            except Exception as exc:
                logger.warning("2020+ MSA population parsing failed: %s", exc)

    if not frames:
        logger.error("No MSA population data collected.")
        return None

    df_all = pd.concat(frames, ignore_index=True)
    df_all["POPULATION"] = pd.to_numeric(df_all["POPULATION"],
                                         errors="coerce").astype("float32")
    df_all["YEAR"] = df_all["YEAR"].astype("int16")
    save_parquet(df_all, parquet_path, logger=logger)
    return parquet_path


# ============================================================
# 2. Jet Fuel Prices
# ============================================================
# FRED plain-text endpoint — no API key needed.
# WJFUELUSGULF = Weekly U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price
# DCOILBRENTEU = Brent crude (monthly) — bonus series for instrument construction

_FRED_TXT = "https://fred.stlouisfed.org/data/{series_id}.txt"

FUEL_SERIES = {
    "WJFUELUSGULF": "Jet fuel spot price, weekly $/gal (U.S. Gulf Coast)",
    "DCOILBRENTEU": "Brent crude spot price, daily $/barrel",
}


def download_jet_fuel(
    start_year: int,
    end_year: int,
    raw_dir: Path,
    out_dir: Path,
    skip_cache: bool = False,
) -> Optional[Path]:
    parquet_path = out_dir / "jet_fuel_prices.parquet"
    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP jet_fuel_prices (parquet exists)")
        return parquet_path

    logger.info("Downloading jet fuel prices from FRED …")
    session = _session()
    frames = []

    for series_id, description in FUEL_SERIES.items():
        url = _FRED_TXT.format(series_id=series_id)
        raw_path = raw_dir / f"fred_{series_id}.txt"
        try:
            download_file(url, raw_path, session=session,
                          logger=logger, skip_cache=skip_cache)

            # FRED .txt files: header lines start with non-digit chars,
            # data lines: "YYYY-MM-DD  VALUE"
            lines = raw_path.read_text(encoding="utf-8").splitlines()
            data_lines = [l for l in lines
                          if re.match(r"\d{4}-\d{2}-\d{2}", l.strip())]
            if not data_lines:
                logger.warning("No data rows found in %s", raw_path.name)
                continue

            df = pd.read_csv(
                io.StringIO("\n".join(data_lines)),
                sep=r"\s+", header=None,
                names=["DATE", "VALUE"],
            )
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            df = df.dropna(subset=["DATE"])
            df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
            df["SERIES_ID"] = series_id
            df["YEAR"] = df["DATE"].dt.year.astype("int16")
            df["MONTH"] = df["DATE"].dt.month.astype("int8")
            df["QUARTER"] = df["DATE"].dt.quarter.astype("int8")
            df = df[
                (df["YEAR"] >= start_year) & (df["YEAR"] <= end_year)
            ]
            frames.append(df)
            logger.info("  %s: %d observations", series_id, len(df))
        except Exception as exc:
            logger.warning("%s failed: %s", series_id, exc)

    if not frames:
        logger.error("No fuel price data downloaded.")
        return None

    df_all = pd.concat(frames, ignore_index=True)
    # Rename VALUE column to something descriptive based on series
    df_jet = df_all[df_all["SERIES_ID"] == "WJFUELUSGULF"].copy()
    df_jet = df_jet.rename(columns={"VALUE": "FUEL_PRICE_PER_GAL"})
    df_jet["FUEL_PRICE_PER_GAL"] = df_jet["FUEL_PRICE_PER_GAL"].astype("float32")

    save_parquet(df_jet[["DATE", "FUEL_PRICE_PER_GAL", "YEAR", "MONTH", "QUARTER"]],
                 parquet_path, logger=logger)

    # Also save full multi-series file for instrument construction
    full_path = out_dir / "energy_prices_full.parquet"
    save_parquet(df_all, full_path, logger=logger)
    return parquet_path


# ============================================================
# 3. Airport / City Market Lookup
# ============================================================
# BTS lookup tables are available via a simple GET (they're static CSVs).
# The old URL used the wrong endpoint pattern. Correct pattern confirmed:
# https://transtats.bts.gov/Download_Lookup.asp?Lookup=L_AIRPORT_ID
# (no Y11x15_VQ parameter needed for public lookups)

_BTS_LOOKUP_URLS = {
    "airport_id": "https://transtats.bts.gov/Download_Lookup.asp?Lookup=L_AIRPORT_ID",
    "city_market_id": "https://transtats.bts.gov/Download_Lookup.asp?Lookup=L_CITY_MARKET_ID",
    "carrier": "https://transtats.bts.gov/Download_Lookup.asp?Lookup=L_UNIQUE_CARRIERS",
    "state": "https://transtats.bts.gov/Download_Lookup.asp?Lookup=L_STATE_ABR_AVIATION",
}

# GitHub mirror as fallback (dannguyen's BTS demo repo — stable, widely used)
_GITHUB_LOOKUP = {
    "airport_id": (
        "https://raw.githubusercontent.com/dannguyen/bts-transstats-t100-domestic-demo"
        "/master/data/lookup-tables/L_AIRPORT_ID.csv"
    ),
}


def download_airport_lookup(
    raw_dir: Path,
    out_dir: Path,
    skip_cache: bool = False,
) -> Optional[Path]:
    parquet_path = out_dir / "airport_lookup.parquet"
    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP airport_lookup (parquet exists)")
        return parquet_path

    logger.info("Downloading BTS airport / city market lookups …")
    session = _session()
    saved = []

    for name, url in _BTS_LOOKUP_URLS.items():
        raw_path = raw_dir / f"bts_lookup_{name}.csv"
        out_path = out_dir / f"{name}.parquet"
        if out_path.exists() and not skip_cache:
            saved.append(out_path)
            continue

        # Try BTS first, then GitHub fallback for airport_id
        for attempt_url in [url] + ([_GITHUB_LOOKUP[name]]
                                     if name in _GITHUB_LOOKUP else []):
            try:
                download_file(attempt_url, raw_path, session=session,
                              logger=logger, skip_cache=skip_cache)
                # BTS lookups may be TSV or CSV; handle both
                for sep in [",", "\t"]:
                    try:
                        df = pd.read_csv(raw_path, encoding="latin-1",
                                         sep=sep, low_memory=False)
                        if len(df.columns) > 1:
                            break
                    except Exception:
                        continue
                df.columns = [c.strip().upper().replace(" ", "_")
                               for c in df.columns]
                save_parquet(df, out_path, logger=logger)
                saved.append(out_path)
                break
            except Exception as exc:
                logger.warning("Lookup %s from %s failed: %s",
                               name, attempt_url, exc)

    if not saved:
        logger.error("No airport lookup files downloaded.")
        return None

    # Also save combined parquet (airport_id as primary)
    primary = out_dir / "airport_id.parquet"
    return primary if primary.exists() else saved[0]


# ============================================================
# 4. Aircraft Characteristics
# ============================================================
# BTS aircraft type lookup via the same form-GET pattern as the other lookups.
# Fallback: GitHub mirror of the BTS T-100 demo lookup files.

_BTS_AIRCRAFT_URL = (
    "https://transtats.bts.gov/Download_Lookup.asp?Lookup=L_AIRCRAFT_TYPE"
)
_GITHUB_AIRCRAFT = (
    "https://raw.githubusercontent.com/dannguyen/bts-transstats-t100-domestic-demo"
    "/master/data/lookup-tables/L_AIRCRAFT_TYPE.csv"
)


def download_aircraft_chars(
    raw_dir: Path,
    out_dir: Path,
    skip_cache: bool = False,
) -> Optional[Path]:
    parquet_path = out_dir / "aircraft_type_lookup.parquet"
    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP aircraft_type_lookup (parquet exists)")
        return parquet_path

    logger.info("Downloading aircraft type lookup …")
    session = _session()

    for url in [_BTS_AIRCRAFT_URL, _GITHUB_AIRCRAFT]:
        raw_path = raw_dir / "bts_aircraft_type.csv"
        try:
            download_file(url, raw_path, session=session,
                          logger=logger, skip_cache=skip_cache)
            for sep in [",", "\t"]:
                try:
                    df = pd.read_csv(raw_path, encoding="latin-1",
                                     sep=sep, low_memory=False)
                    if len(df.columns) > 1:
                        break
                except Exception:
                    continue
            df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
            save_parquet(df, parquet_path, logger=logger)
            return parquet_path
        except Exception as exc:
            logger.warning("Aircraft lookup from %s failed: %s", url, exc)

    logger.error("Aircraft type lookup download failed from all sources.")
    return None


# ============================================================
# 5. CPI — All Items + Air Transportation
# ============================================================
# Strategy A: FRED plain-text for CPI All Items (CPIAUCSL) — no key needed.
# Strategy B: BLS public data flat file for air transport CPI series
#   https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems
#   This is a large (~200 MB) tab-separated file; we filter to the two series
#   we need: CUSR0000SETG01 (Air Transportation) and CUUR0000SA0 (All items).

_FRED_CPI_ALL    = "https://fred.stlouisfed.org/data/CPIAUCSL.txt"
_BLS_CU_DATA_URL = "https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems"

CPI_SERIES = {
    "CUUR0000SA0":    "CPI_ALL_ITEMS",        # all items, NSA
    "CUSR0000SA0":    "CPI_ALL_ITEMS_SA",     # all items, SA
    "CUSR0000SETG01": "CPI_AIR_TRANSPORT",    # air transportation, SA
}


def download_cpi(
    start_year: int,
    end_year: int,
    raw_dir: Path,
    out_dir: Path,
    skip_cache: bool = False,
) -> Optional[Path]:
    parquet_path = out_dir / "cpi.parquet"
    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP cpi (parquet exists)")
        return parquet_path

    logger.info("Downloading CPI data …")
    session = _session()
    frames = []

    # ── Strategy A: FRED for CPI All Items ──────────────────────────────────
    raw_path = raw_dir / "fred_CPIAUCSL.txt"
    try:
        download_file(_FRED_CPI_ALL, raw_path, session=session,
                      logger=logger, skip_cache=skip_cache)
        lines = raw_path.read_text(encoding="utf-8").splitlines()
        data_lines = [l for l in lines
                      if re.match(r"\d{4}-\d{2}-\d{2}", l.strip())]
        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            sep=r"\s+", header=None,
            names=["DATE", "CPI_ALL_ITEMS_FRED"],
        )
        df["DATE"] = pd.to_datetime(df["DATE"])
        df["CPI_ALL_ITEMS_FRED"] = pd.to_numeric(df["CPI_ALL_ITEMS_FRED"],
                                                   errors="coerce")
        df["YEAR"] = df["DATE"].dt.year.astype("int16")
        df["MONTH"] = df["DATE"].dt.month.astype("int8")
        df = df[(df["YEAR"] >= start_year) & (df["YEAR"] <= end_year)]
        frames.append(df.set_index(["YEAR", "MONTH"])[["CPI_ALL_ITEMS_FRED"]])
        logger.info("  CPIAUCSL (FRED): %d rows", len(df))
    except Exception as exc:
        logger.warning("FRED CPI download failed: %s", exc)

    # ── Strategy B: BLS flat file for air transport CPI ─────────────────────
    raw_path = raw_dir / "bls_cu_allitems.txt"
    try:
        download_file(_BLS_CU_DATA_URL, raw_path, session=session,
                      logger=logger, skip_cache=skip_cache)

        logger.info("  Parsing BLS CPI flat file (this may take a moment) …")
        target_series = set(CPI_SERIES.keys())
        rows = []
        with open(raw_path, encoding="utf-8", errors="replace") as fh:
            header = fh.readline()  # skip header
            for line in tqdm(fh, desc="  Scanning BLS CPI", leave=False,
                             unit_scale=True):
                parts = line.split()
                if len(parts) < 4:
                    continue
                if parts[0] in target_series:
                    try:
                        year = int(parts[1])
                        period = parts[2]          # M01 … M12, M13=annual
                        value = float(parts[3])
                        if period.startswith("M") and period != "MS":
                            month = int(period[1:])
                            if 1 <= month <= 12:
                                if start_year <= year <= end_year:
                                    rows.append({
                                        "SERIES_ID": parts[0],
                                        "YEAR": year,
                                        "MONTH": month,
                                        "VALUE": value,
                                    })
                    except (ValueError, IndexError):
                        continue

        if rows:
            df_bls = pd.DataFrame(rows)
            df_pivot = df_bls.pivot_table(
                index=["YEAR", "MONTH"],
                columns="SERIES_ID",
                values="VALUE",
            ).reset_index()
            df_pivot.columns.name = None
            df_pivot = df_pivot.rename(columns=CPI_SERIES)
            frames.append(df_pivot.set_index(["YEAR", "MONTH"]))
            logger.info("  BLS CPI flat file: %d rows", len(df_bls))
        else:
            logger.warning("No matching CPI series found in BLS flat file.")
    except Exception as exc:
        logger.warning("BLS CPI flat file failed: %s", exc)

    if not frames:
        logger.error("No CPI data collected.")
        return None

    # Merge all CPI series on YEAR + MONTH
    from functools import reduce
    df_all = reduce(
        lambda a, b: a.join(b, how="outer"),
        frames,
    ).reset_index()
    df_all["YEAR"] = df_all["YEAR"].astype("int16")
    df_all["MONTH"] = df_all["MONTH"].astype("int8")
    for col in df_all.select_dtypes("float64").columns:
        df_all[col] = df_all[col].astype("float32")
    save_parquet(df_all, parquet_path, logger=logger)
    return parquet_path


# ============================================================
# 6. CBSA Delineation
# ============================================================
# Verified direct URL (OMB July 2023 delineations):
# https://www2.census.gov/programs-surveys/metro-micro/geographies/
#   reference-files/2023/delineation-files/list1_2023.xlsx

_CBSA_URLS = [
    # July 2023 delineation (current)
    (
        "https://www2.census.gov/programs-surveys/metro-micro/geographies/"
        "reference-files/2023/delineation-files/list1_2023.xlsx"
    ),
    # March 2020 delineation (fallback)
    (
        "https://www2.census.gov/programs-surveys/metro-micro/geographies/"
        "reference-files/2020/delineation-files/list1_2020.xlsx"
    ),
]


def download_cbsa_delineation(
    raw_dir: Path,
    out_dir: Path,
    skip_cache: bool = False,
) -> Optional[Path]:
    parquet_path = out_dir / "cbsa_delineation.parquet"
    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP cbsa_delineation (parquet exists)")
        return parquet_path

    logger.info("Downloading OMB CBSA delineation file …")
    session = _session()

    for url in _CBSA_URLS:
        vintage = re.search(r"/(\d{4})/", url).group(1)
        raw_path = raw_dir / f"cbsa_delineation_{vintage}.xlsx"
        try:
            download_file(url, raw_path, session=session,
                          logger=logger, skip_cache=skip_cache)

            # Validate it's an Excel file
            if raw_path.stat().st_size < 10_000:
                logger.warning("Downloaded file too small (%d bytes) — skipping",
                               raw_path.stat().st_size)
                raw_path.unlink(missing_ok=True)
                continue

            # OMB delineation xlsx: 3 header rows, data from row 4
            df = pd.read_excel(
                raw_path, header=2, dtype=str, engine="openpyxl"
            )
            df.columns = [
                c.strip().upper().replace(" ", "_").replace("/", "_").replace("-", "_")
                for c in df.columns
            ]
            # Drop footnote rows at the bottom (cells in col 0 start with "Source")
            df = df.dropna(subset=[df.columns[0]])
            df = df[~df.iloc[:, 0].astype(str).str.startswith("Source")]
            df = df[~df.iloc[:, 0].astype(str).str.startswith("Notes")]

            save_parquet(df, parquet_path, logger=logger)
            return parquet_path
        except Exception as exc:
            logger.warning("CBSA delineation from %s failed: %s", url, exc)
            raw_path.unlink(missing_ok=True)

    logger.error("CBSA delineation download failed from all sources.")
    return None


# ============================================================
# CLI
# ============================================================

DOWNLOADERS = {
    "msa":      download_msa_population,
    "fuel":     download_jet_fuel,
    "airport":  download_airport_lookup,
    "aircraft": download_aircraft_chars,
    "cpi":      download_cpi,
    "cbsa":     download_cbsa_delineation,
}


def parse_args() -> argparse.Namespace:
    current_year = datetime.now().year
    p = argparse.ArgumentParser(
        description="Download supporting datasets for airline merger analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year", type=int, default=current_year)
    p.add_argument(
        "--datasets", nargs="+",
        default=list(DOWNLOADERS.keys()),
        choices=list(DOWNLOADERS.keys()) + ["all"],
    )
    p.add_argument("--out-dir", type=Path, default=Path("parquet/supporting"))
    p.add_argument("--raw-dir", type=Path, default=Path("supporting"))
    p.add_argument("--skip-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if "all" in args.datasets:
        args.datasets = list(DOWNLOADERS.keys())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Optional[Path]] = {}

    with tqdm(total=len(args.datasets), desc="Supporting data", unit="dataset") as pbar:
        for name in args.datasets:
            pbar.set_description(f"  {name}")
            fn = DOWNLOADERS[name]
            try:
                # Functions that don't take year args
                if name in ("airport", "aircraft", "cbsa"):
                    result = fn(args.raw_dir, args.out_dir,
                                skip_cache=args.skip_cache)
                else:
                    result = fn(args.start_year, args.end_year,
                                args.raw_dir, args.out_dir,
                                skip_cache=args.skip_cache)
                results[name] = result
            except Exception as exc:
                logger.error("Unhandled error in %s: %s", name, exc)
                results[name] = None
            pbar.update(1)
            time.sleep(0.5)

    print()
    logger.info("=== Supporting Data Summary ===")
    for name, path in results.items():
        status = f"✓  {path}" if path else "✗  FAILED"
        logger.info("  %-12s  %s", name, status)

    failed = [k for k, v in results.items() if v is None]
    if failed:
        logger.warning("Failed: %s", failed)
        sys.exit(1)
    else:
        logger.info("All supporting datasets downloaded successfully.")


if __name__ == "__main__":
    main()
