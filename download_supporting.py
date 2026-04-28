#!/usr/bin/env python3
"""
download_supporting.py — Download all supporting datasets for BLP merger analysis.

Datasets collected
------------------
1. MSA / CBSA Population (Census Bureau)
   - Used to construct market size (potential passengers per O-D pair)
   - Annual estimates via Census Population Estimates API

2. EIA Jet Fuel Prices (U.S. Energy Information Administration)
   - Weekly U.S. kerosene-type jet fuel prices ($/gallon)
   - Used as cost shifter instrument in BLP supply side

3. BTS Airport / City Lookup (Aviation Support Tables)
   - Maps IATA airport codes → city names, state, CBSA codes
   - Downloaded from BTS MASTER_CORD table

4. FAA Aircraft Characteristics (aircraft type → seat capacity, range)
   - Used to map T-100 aircraft type codes → capacity

5. Bureau of Labor Statistics CPI (All Urban, air travel component)
   - Used to deflate nominal fares to real (constant-dollar) fares

6. CBSA Delineation Files (OMB)
   - Maps counties → Core Based Statistical Areas (city-pair market definition)

Usage
-----
  python download_supporting.py
  python download_supporting.py --start-year 2019 --end-year 2024
  python download_supporting.py --datasets msa fuel airport --skip-cache

Arguments
---------
  --start-year INT    First year for time-series data. Default: 2010
  --end-year   INT    Last year. Default: current year
  --datasets   STR…   Which datasets: msa | fuel | airport | aircraft | cpi | cbsa | all
  --out-dir    PATH   Output directory. Default: ./parquet/supporting
  --raw-dir    PATH   Raw file cache dir. Default: ./supporting
  --skip-cache FLAG   Force re-download
  --census-key STR    Census API key (optional; anonymous works for most calls)
"""

from __future__ import annotations

import argparse
import io
import json
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

sys.path.insert(0, str(Path(__file__).parent))
from utils import download_file, get_logger, save_parquet

logger = get_logger("supporting", "supporting.log")

# ---------------------------------------------------------------------------
# 1. MSA Population — Census Population Estimates (PEP)
# ---------------------------------------------------------------------------

def download_msa_population(
    start_year: int,
    end_year: int,
    raw_dir: Path,
    out_dir: Path,
    session: requests.Session,
    census_key: Optional[str] = None,
    skip_cache: bool = False,
) -> Optional[Path]:
    """
    Download MSA-level population estimates from the Census Bureau PEP API.

    Returns a wide Parquet with columns: CBSA_CODE, CBSA_NAME, YEAR, POPULATION
    """
    parquet_path = out_dir / "msa_population.parquet"
    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP msa_population (parquet exists)")
        return parquet_path

    logger.info("Downloading MSA population from Census PEP API …")
    api_key_param = f"&key={census_key}" if census_key else ""

    all_frames: list[pd.DataFrame] = []

    # Census PEP API releases:
    # 2010–2019: vintage 2019
    # 2020–present: vintage 2023 (or latest)
    # We pull each vintage separately to get consistent CBSA boundaries.

    vintage_ranges = [
        # (vintage, years_available, api_base)
        (2019, range(max(start_year, 2010), min(end_year + 1, 2020)),
         "https://api.census.gov/data/2019/pep/population"),
        (2023, range(max(start_year, 2020), min(end_year + 1, 2024)),
         "https://api.census.gov/data/2023/pep/population"),
    ]

    for vintage, years, base_url in vintage_ranges:
        year_list = list(years)
        if not year_list:
            continue

        # PEP returns all years in a vintage in one call with DATE_CODE filtering
        # For simplicity we request CBSA-level totals
        for year in tqdm(year_list, desc=f"  Census PEP vintage {vintage}", leave=False):
            url = (
                f"{base_url}?get=NAME,POP,DENSITY&for=metropolitan+statistical+area/"
                f"micropolitan+statistical+area:*&DATE_CODE={year - 2010 + 2}"
                f"{api_key_param}"
            )
            # Alternative (simpler) endpoint for 2020+
            if vintage >= 2023:
                url = (
                    f"{base_url}?get=NAME,POP&for=metropolitan+statistical+area/"
                    f"micropolitan+statistical+area:*&YEAR={year}"
                    f"{api_key_param}"
                )

            cache_path = raw_dir / f"census_pep_msa_{year}.json"
            try:
                if not cache_path.exists() or skip_cache:
                    resp = session.get(url, timeout=30)
                    resp.raise_for_status()
                    cache_path.write_text(resp.text, encoding="utf-8")
                    logger.debug("Downloaded Census PEP MSA %d", year)
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                headers = data[0]
                rows = data[1:]
                df = pd.DataFrame(rows, columns=headers)
                # Rename geographic identifier column
                geo_col = [c for c in df.columns if "metropolitan" in c.lower()]
                if geo_col:
                    df = df.rename(columns={geo_col[0]: "CBSA_CODE"})
                df["YEAR"] = year
                df["POPULATION"] = pd.to_numeric(df.get("POP", df.get("POPESTIMATE", 0)),
                                                 errors="coerce")
                df = df.rename(columns={"NAME": "CBSA_NAME"})
                all_frames.append(df[["CBSA_CODE", "CBSA_NAME", "YEAR", "POPULATION"]])
            except Exception as exc:
                logger.warning("Census PEP MSA %d failed: %s", year, exc)

    if not all_frames:
        logger.error("No MSA population data retrieved.")
        return None

    df_all = pd.concat(all_frames, ignore_index=True)
    df_all["POPULATION"] = pd.to_numeric(df_all["POPULATION"], errors="coerce").astype("float32")
    df_all["YEAR"] = df_all["YEAR"].astype("int16")
    save_parquet(df_all, parquet_path, logger=logger)
    return parquet_path


# ---------------------------------------------------------------------------
# 2. EIA Jet Fuel Prices — Weekly U.S. Kerosene-Type Jet Fuel
# ---------------------------------------------------------------------------

EIA_JET_FUEL_URL = (
    "https://www.eia.gov/dnav/pet/hist/xls/EER_EPJK_PF4_RGC_DPG_w.xls"
)
# Alternatively the EIA API:
EIA_API_SERIES = "PET.EER_EPJK_PF4_RGC_DPG.W"  # U.S. weekly kerosene jet fuel $/gal


def download_jet_fuel(
    start_year: int,
    end_year: int,
    raw_dir: Path,
    out_dir: Path,
    session: requests.Session,
    skip_cache: bool = False,
    eia_api_key: Optional[str] = None,
) -> Optional[Path]:
    """
    Download EIA weekly jet fuel prices.

    Saves: parquet with columns DATE, FUEL_PRICE_PER_GAL, YEAR, MONTH, QUARTER
    """
    parquet_path = out_dir / "jet_fuel_prices.parquet"
    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP jet_fuel_prices (parquet exists)")
        return parquet_path

    logger.info("Downloading EIA jet fuel prices …")

    # Try EIA API first (JSON, no Excel dependency)
    if eia_api_key:
        url = (
            f"https://api.eia.gov/v2/petroleum/pri/wfr/data/"
            f"?frequency=weekly&data[0]=value&facets[duoarea][]=RGC"
            f"&facets[product][]=EPD2DXL0&sort[0][column]=period"
            f"&sort[0][direction]=asc&offset=0&length=5000"
            f"&api_key={eia_api_key}"
        )
        cache_path = raw_dir / "eia_jet_fuel_api.json"
        try:
            if not cache_path.exists() or skip_cache:
                resp = session.get(url, timeout=30)
                resp.raise_for_status()
                cache_path.write_text(resp.text, encoding="utf-8")
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            rows = data["response"]["data"]
            df = pd.DataFrame(rows)
            df = df.rename(columns={"period": "DATE", "value": "FUEL_PRICE_PER_GAL"})
            df["DATE"] = pd.to_datetime(df["DATE"])
            df["FUEL_PRICE_PER_GAL"] = pd.to_numeric(df["FUEL_PRICE_PER_GAL"], errors="coerce").astype("float32")
            df["YEAR"] = df["DATE"].dt.year.astype("int16")
            df["MONTH"] = df["DATE"].dt.month.astype("int8")
            df["QUARTER"] = df["DATE"].dt.quarter.astype("int8")
            df = df[
                (df["YEAR"] >= start_year) & (df["YEAR"] <= end_year)
            ][["DATE", "FUEL_PRICE_PER_GAL", "YEAR", "MONTH", "QUARTER"]]
            save_parquet(df, parquet_path, logger=logger)
            return parquet_path
        except Exception as exc:
            logger.warning("EIA API failed (%s), falling back to XLS …", exc)

    # Fallback: download the XLS directly
    xls_path = raw_dir / "eia_jet_fuel.xls"
    try:
        download_file(
            EIA_JET_FUEL_URL, xls_path,
            session=session, logger=logger, skip_cache=skip_cache,
        )
        # EIA XLS files have 2 header rows; data starts at row 3
        df = pd.read_excel(xls_path, header=2, engine="xlrd")
        df.columns = ["DATE", "FUEL_PRICE_PER_GAL"]
        df = df.dropna(subset=["DATE", "FUEL_PRICE_PER_GAL"])
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["DATE"])
        df["FUEL_PRICE_PER_GAL"] = pd.to_numeric(
            df["FUEL_PRICE_PER_GAL"], errors="coerce"
        ).astype("float32")
        df["YEAR"] = df["DATE"].dt.year.astype("int16")
        df["MONTH"] = df["DATE"].dt.month.astype("int8")
        df["QUARTER"] = df["DATE"].dt.quarter.astype("int8")
        df = df[
            (df["YEAR"] >= start_year) & (df["YEAR"] <= end_year)
        ]
        save_parquet(df, parquet_path, logger=logger)
        return parquet_path
    except Exception as exc:
        logger.error("Jet fuel download failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 3. BTS Airport–City Lookup (MASTER_CORD)
# ---------------------------------------------------------------------------

AIRPORT_LOOKUP_URL = (
    "https://transtats.bts.gov/Download_Lookup.asp?Y11x15_VQ=EFI&Lookup=L_AIRPORT_ID"
)
# Fallback: static CSV from BTS
AIRPORT_CORD_URL = (
    "https://transtats.bts.gov/Download_Lookup.asp?Y11x15_VQ=EFI&Lookup=L_CITY_MARKET_ID"
)


def download_airport_lookup(
    raw_dir: Path,
    out_dir: Path,
    session: requests.Session,
    skip_cache: bool = False,
) -> Optional[Path]:
    """
    Download BTS airport → city market ID lookup table.
    Essential for mapping IATA codes to city-pair markets.
    """
    parquet_path = out_dir / "airport_lookup.parquet"
    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP airport_lookup (parquet exists)")
        return parquet_path

    logger.info("Downloading BTS airport lookup table …")

    # BTS lookup tables are plain CSV (no zip)
    urls = {
        "airport_id": "https://transtats.bts.gov/Download_Lookup.asp?Y11x15_VQ=EFI&Lookup=L_AIRPORT_ID",
        "city_market": "https://transtats.bts.gov/Download_Lookup.asp?Y11x15_VQ=EFI&Lookup=L_CITY_MARKET_ID",
    }

    frames = {}
    for name, url in urls.items():
        cache_path = raw_dir / f"bts_{name}.csv"
        try:
            if not cache_path.exists() or skip_cache:
                resp = session.get(url, timeout=60)
                resp.raise_for_status()
                cache_path.write_bytes(resp.content)
                logger.debug("Downloaded %s", name)
            df = pd.read_csv(cache_path, encoding="latin-1", low_memory=False)
            df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
            frames[name] = df
        except Exception as exc:
            logger.warning("Airport lookup %s failed: %s", name, exc)

    if not frames:
        return None

    # Merge airport → city market
    if "airport_id" in frames and "city_market" in frames:
        df_airport = frames["airport_id"]
        df_city = frames["city_market"]
        # Save combined
        save_parquet(df_airport, out_dir / "airport_id.parquet", logger=logger)
        save_parquet(df_city, out_dir / "city_market_id.parquet", logger=logger)
        logger.info("Saved airport and city market lookup tables.")
        return out_dir / "airport_id.parquet"
    elif frames:
        first = next(iter(frames.values()))
        save_parquet(first, parquet_path, logger=logger)
        return parquet_path

    return None


# ---------------------------------------------------------------------------
# 4. FAA Aircraft Characteristics Database
# ---------------------------------------------------------------------------

FAA_AIRCRAFT_URL = (
    "https://www.faa.gov/airports/engineering/aircraft_char_database/assets/media/"
    "FAA-Aircraft-Char-Database-v2-201810.xlsx"
)


def download_aircraft_chars(
    raw_dir: Path,
    out_dir: Path,
    session: requests.Session,
    skip_cache: bool = False,
) -> Optional[Path]:
    """
    Download FAA Aircraft Characteristics Database.
    Provides seat capacity, MTOW, wingspan by aircraft type.
    """
    parquet_path = out_dir / "aircraft_characteristics.parquet"
    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP aircraft_characteristics (parquet exists)")
        return parquet_path

    logger.info("Downloading FAA aircraft characteristics …")
    xlsx_path = raw_dir / "faa_aircraft_chars.xlsx"

    try:
        download_file(FAA_AIRCRAFT_URL, xlsx_path, session=session,
                      logger=logger, skip_cache=skip_cache)
        df = pd.read_excel(xlsx_path, sheet_name=0, header=0, engine="openpyxl")
        df.columns = [c.strip().upper().replace(" ", "_").replace("/", "_")
                      for c in df.columns]
        # Keep relevant columns
        keep = [c for c in df.columns if any(k in c for k in
                ["MANUFACTURER", "MODEL", "SEATS", "MTOW", "CLASS", "TYPE", "ENGINE"])]
        df = df[keep] if keep else df
        for col in df.select_dtypes("float64").columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        save_parquet(df, parquet_path, logger=logger)
        return parquet_path
    except Exception as exc:
        logger.warning("FAA aircraft chars failed: %s — trying BTS aircraft lookup …", exc)

    # Fallback: BTS aircraft type lookup
    bts_aircraft_url = "https://transtats.bts.gov/Download_Lookup.asp?Y11x15_VQ=EFI&Lookup=L_AIRCRAFT_TYPE"
    cache_path = raw_dir / "bts_aircraft_type.csv"
    try:
        if not cache_path.exists() or skip_cache:
            resp = session.get(bts_aircraft_url, timeout=60)
            resp.raise_for_status()
            cache_path.write_bytes(resp.content)
        df = pd.read_csv(cache_path, encoding="latin-1")
        df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
        save_parquet(df, parquet_path, logger=logger)
        return parquet_path
    except Exception as exc:
        logger.error("Aircraft lookup fallback also failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 5. BLS CPI — All Urban Consumers, Air Transportation (CUSR0000SETG01)
# ---------------------------------------------------------------------------

BLS_CPI_URL = "https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems"
BLS_AIR_SERIES = "CUSR0000SETG01"  # CPI: air transportation


def download_cpi(
    start_year: int,
    end_year: int,
    raw_dir: Path,
    out_dir: Path,
    session: requests.Session,
    skip_cache: bool = False,
) -> Optional[Path]:
    """
    Download BLS CPI air transportation series for fare deflation.
    """
    parquet_path = out_dir / "cpi_air_transportation.parquet"
    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP cpi_air_transportation (parquet exists)")
        return parquet_path

    logger.info("Downloading BLS CPI air transportation series …")

    # BLS public API (no key required for ≤ 25 series / 20 years)
    api_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = {
        "seriesid": [BLS_AIR_SERIES, "CUUR0000SA0"],  # air CPI + all-items CPI
        "startyear": str(start_year),
        "endyear": str(end_year),
        "catalog": False,
        "calculations": False,
        "annualaverage": True,
    }

    cache_path = raw_dir / f"bls_cpi_{start_year}_{end_year}.json"
    try:
        if not cache_path.exists() or skip_cache:
            resp = session.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()
            cache_path.write_text(resp.text, encoding="utf-8")
            logger.debug("Downloaded BLS CPI data")

        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("status") != "REQUEST_SUCCEEDED":
            logger.warning("BLS API status: %s", data.get("status"))

        frames = []
        for series in data.get("Results", {}).get("series", []):
            sid = series["seriesID"]
            for obs in series.get("data", []):
                frames.append({
                    "SERIES_ID": sid,
                    "YEAR": int(obs["year"]),
                    "PERIOD": obs["period"],
                    "VALUE": float(obs["value"]) if obs["value"] != "-" else None,
                })

        df = pd.DataFrame(frames)
        # Keep only monthly (M01–M12) and annual (M13) observations
        df = df[df["PERIOD"].str.startswith("M")]
        df["MONTH"] = df["PERIOD"].str[1:].astype(int)
        df = df[df["MONTH"] <= 13]

        # Pivot to wide: one column per series
        df_pivot = df.pivot_table(
            index=["YEAR", "MONTH"], columns="SERIES_ID", values="VALUE"
        ).reset_index()
        df_pivot.columns.name = None

        # Rename for clarity
        rename = {
            BLS_AIR_SERIES: "CPI_AIR_TRANSPORT",
            "CUUR0000SA0": "CPI_ALL_ITEMS",
        }
        df_pivot = df_pivot.rename(columns=rename)
        for col in df_pivot.select_dtypes("float64").columns:
            df_pivot[col] = df_pivot[col].astype("float32")
        df_pivot["YEAR"] = df_pivot["YEAR"].astype("int16")
        df_pivot["MONTH"] = df_pivot["MONTH"].astype("int8")

        save_parquet(df_pivot, parquet_path, logger=logger)
        return parquet_path

    except Exception as exc:
        logger.error("BLS CPI download failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 6. OMB CBSA Delineation File
# ---------------------------------------------------------------------------

CBSA_URL = (
    "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/"
    "2023/delineation-files/list1_2023.xlsx"
)


def download_cbsa_delineation(
    raw_dir: Path,
    out_dir: Path,
    session: requests.Session,
    skip_cache: bool = False,
) -> Optional[Path]:
    """
    Download OMB CBSA delineation file.
    Maps counties → MSA/CBSA codes and names. Used for market size computation.
    """
    parquet_path = out_dir / "cbsa_delineation.parquet"
    if parquet_path.exists() and not skip_cache:
        logger.info("SKIP cbsa_delineation (parquet exists)")
        return parquet_path

    logger.info("Downloading OMB CBSA delineation file …")
    xlsx_path = raw_dir / "cbsa_delineation_2023.xlsx"

    # Try latest vintage first, then fall back to 2020
    urls_to_try = [
        CBSA_URL,
        "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2020/delineation-files/list1_2020.xlsx",
    ]

    for url in urls_to_try:
        try:
            download_file(url, xlsx_path, session=session, logger=logger, skip_cache=skip_cache)
            # OMB delineation files have 3 header rows
            df = pd.read_excel(xlsx_path, header=2, dtype=str, engine="openpyxl")
            df.columns = [
                c.strip().upper().replace(" ", "_").replace("/", "_")
                for c in df.columns
            ]
            # Drop trailing junk rows (OMB files have footnotes at bottom)
            df = df.dropna(subset=[df.columns[0]])
            df = df[~df.iloc[:, 0].str.startswith("Source", na=True)]
            save_parquet(df, parquet_path, logger=logger)
            return parquet_path
        except Exception as exc:
            logger.warning("CBSA download failed (%s): %s", url, exc)

    logger.error("All CBSA URL attempts failed.")
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DATASET_FUNCS = {
    "msa": "msa",
    "fuel": "fuel",
    "airport": "airport",
    "aircraft": "aircraft",
    "cpi": "cpi",
    "cbsa": "cbsa",
}


def parse_args() -> argparse.Namespace:
    current_year = datetime.now().year
    p = argparse.ArgumentParser(
        description="Download supporting datasets for airline merger analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year", type=int, default=current_year)
    p.add_argument("--datasets", nargs="+",
                   default=["msa", "fuel", "airport", "aircraft", "cpi", "cbsa"],
                   choices=list(DATASET_FUNCS.keys()) + ["all"])
    p.add_argument("--out-dir", type=Path, default=Path("parquet/supporting"))
    p.add_argument("--raw-dir", type=Path, default=Path("supporting"))
    p.add_argument("--skip-cache", action="store_true")
    p.add_argument("--census-key", type=str, default=None,
                   help="Census Bureau API key (free at api.census.gov/data/key_signup.html)")
    p.add_argument("--eia-key", type=str, default=None,
                   help="EIA API key (free at eia.gov/opendata)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if "all" in args.datasets:
        args.datasets = list(DATASET_FUNCS.keys())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; airline-research-bot/1.0)",
        "Accept": "application/json, text/html, */*",
    })

    results: dict[str, Optional[Path]] = {}

    with tqdm(total=len(args.datasets), desc="Supporting data", unit="dataset") as pbar:

        if "msa" in args.datasets:
            pbar.set_description("MSA Population")
            results["msa"] = download_msa_population(
                args.start_year, args.end_year,
                args.raw_dir, args.out_dir, session,
                census_key=args.census_key,
                skip_cache=args.skip_cache,
            )
            pbar.update(1)

        if "fuel" in args.datasets:
            pbar.set_description("EIA Jet Fuel")
            results["fuel"] = download_jet_fuel(
                args.start_year, args.end_year,
                args.raw_dir, args.out_dir, session,
                skip_cache=args.skip_cache,
                eia_api_key=args.eia_key,
            )
            pbar.update(1)

        if "airport" in args.datasets:
            pbar.set_description("Airport Lookup")
            results["airport"] = download_airport_lookup(
                args.raw_dir, args.out_dir, session,
                skip_cache=args.skip_cache,
            )
            pbar.update(1)

        if "aircraft" in args.datasets:
            pbar.set_description("Aircraft Chars")
            results["aircraft"] = download_aircraft_chars(
                args.raw_dir, args.out_dir, session,
                skip_cache=args.skip_cache,
            )
            pbar.update(1)

        if "cpi" in args.datasets:
            pbar.set_description("BLS CPI")
            results["cpi"] = download_cpi(
                args.start_year, args.end_year,
                args.raw_dir, args.out_dir, session,
                skip_cache=args.skip_cache,
            )
            pbar.update(1)

        if "cbsa" in args.datasets:
            pbar.set_description("CBSA Delineation")
            results["cbsa"] = download_cbsa_delineation(
                args.raw_dir, args.out_dir, session,
                skip_cache=args.skip_cache,
            )
            pbar.update(1)

    # Summary
    print()
    logger.info("=== Supporting Data Download Summary ===")
    for name, path in results.items():
        status = f"✓ {path}" if path else "✗ FAILED"
        logger.info("  %-12s %s", name, status)

    failed = [k for k, v in results.items() if v is None]
    if failed:
        logger.warning("Failed datasets: %s", failed)
        sys.exit(1)
    else:
        logger.info("All supporting datasets downloaded successfully.")


if __name__ == "__main__":
    main()
