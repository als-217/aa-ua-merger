"""
collect_merger_data.py
======================
Data collection pipeline for UA/AA merger structural estimation.

Sources:
  - BTS DB1B (Origin-Destination Survey): fares, passengers, routing
  - BTS T-100 Domestic Segment: capacity, frequency, seats
  - BTS Form 41 (Schedule P-12a): operating costs by carrier
  - EIA Jet Fuel Prices: cost instrument
  - BTS Airport Master Coordinates: lat/lon for distance calculation

Memory architecture:
  - All intermediate data written to Parquet (columnar, compressed)
  - Final merge done in route-level chunks to stay within 64 GB RAM
  - Logging + SHA-256 cache manifest prevents redundant downloads

Usage:
  python collect_merger_data.py [--years 2018 2019 2022 2023] [--quarters 1 2 3 4]
                                [--output-dir ./merger_data] [--workers 4]
"""

import os
import sys
import json
import hashlib
import logging
import argparse
import zipfile
import io
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests
import aiohttp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("merger_data")


# ── Cache manifest ─────────────────────────────────────────────────────────────

class DownloadCache:
    """
    JSON manifest that tracks every downloaded file by its URL.
    Stores: local path, file size, SHA-256 of first 1 MB (fast partial hash),
    and download timestamp.  Re-download only if the file is missing or corrupt.
    """

    def __init__(self, cache_file: Path):
        self.path = cache_file
        self.manifest: dict = {}
        if cache_file.exists():
            with open(cache_file) as f:
                self.manifest = json.load(f)

    def _fast_hash(self, filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            h.update(f.read(1024 * 1024))  # hash first 1 MB only
        return h.hexdigest()

    def is_cached(self, url: str, dest: Path) -> bool:
        if url not in self.manifest:
            return False
        entry = self.manifest[url]
        if not dest.exists():
            return False
        if dest.stat().st_size != entry.get("size"):
            return False
        return True  # skip full hash for speed; size match is sufficient

    def record(self, url: str, dest: Path):
        self.manifest[url] = {
            "local": str(dest),
            "size": dest.stat().st_size,
            "hash_1mb": self._fast_hash(dest),
            "downloaded_at": datetime.now().isoformat(),
        }
        with open(self.path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def invalidate(self, url: str):
        self.manifest.pop(url, None)
        with open(self.path, "w") as f:
            json.dump(self.manifest, f, indent=2)


# ── HTTP helpers ───────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AirlineMergerResearch/1.0; "
        "academic-use@university.edu)"
    )
}
MAX_RETRIES = 4
BACKOFF_BASE = 2  # seconds


def download_file(
    url: str,
    dest: Path,
    cache: DownloadCache,
    logger: logging.Logger,
    chunk_size: int = 8 * 1024 * 1024,  # 8 MB chunks
) -> bool:
    """Download url → dest with retry/backoff.  Returns True if newly downloaded."""
    if cache.is_cached(url, dest):
        logger.info(f"CACHE HIT  {dest.name}")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"DOWNLOAD [{attempt}/{MAX_RETRIES}]  {url}")
            with requests.get(url, stream=True, headers=HEADERS, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                tmp = dest.with_suffix(".tmp")
                with open(tmp, "wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                    leave=False,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        bar.update(len(chunk))
            tmp.rename(dest)
            cache.record(url, dest)
            logger.info(f"SAVED  {dest}  ({dest.stat().st_size/1e6:.1f} MB)")
            return True
        except Exception as e:
            wait = BACKOFF_BASE ** attempt
            logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {wait}s…")
            if dest.with_suffix(".tmp").exists():
                dest.with_suffix(".tmp").unlink()
            time.sleep(wait)

    logger.error(f"FAILED after {MAX_RETRIES} attempts: {url}")
    return False


# ── BTS DB1B  ─────────────────────────────────────────────────────────────────
# Each quarter ships as a zip containing DB1B_Coupon_YYYYQ.csv (~500 MB unzipped)
# and DB1B_Market_YYYYQ.csv.  We only need Market (O-D level).

DB1B_BASE = "https://transtats.bts.gov/PREZIP/Origin_and_Destination_Survey_DB1BMarket_{year}_{q}.zip"

DB1B_COLS = {
    # raw column name           : output name
    "YEAR":                       "year",
    "QUARTER":                    "quarter",
    "ORIGIN":                     "origin",
    "DEST":                       "dest",
    "OPERATING_CARRIER":          "op_carrier",
    "TICKETING_CARRIER":          "tkt_carrier",
    "PASSENGERS":                 "passengers",
    "MARKET_FARE":                "avg_fare",
    "MARKET_MILES_FLOWN":         "mkt_miles",
    "NONSTOP_MILES":              "nonstop_miles",
    "NUM_CHANGES":                "n_connections",
    "MARKET_DISTANCE":            "mkt_distance",
    "ROUNDTRIP":                  "roundtrip",
    "ORIGIN_STATE_ABR":           "origin_state",
    "DEST_STATE_ABR":             "dest_state",
    "ORIGIN_COUNTRY":             "origin_country",
    "DEST_COUNTRY":               "dest_country",
    "ORIGIN_STATE_FIPS":          "origin_fips",
    "DEST_STATE_FIPS":            "dest_fips",
    "BULKFARES":                  "bulk_fare",
    "ORIGIN_AIRPORT_ID":          "origin_apt_id",
    "DEST_AIRPORT_ID":            "dest_apt_id",
    "ORIGIN_CITY_MARKET_ID":      "origin_mkt_id",
    "DEST_CITY_MARKET_ID":        "dest_mkt_id",
}

DB1B_DTYPES = {
    "YEAR": "int16",
    "QUARTER": "int8",
    "PASSENGERS": "float32",
    "MARKET_FARE": "float32",
    "MARKET_MILES_FLOWN": "float32",
    "NONSTOP_MILES": "float32",
    "NUM_CHANGES": "int8",
    "MARKET_DISTANCE": "float32",
    "ROUNDTRIP": "int8",
    "BULKFARES": "int8",
    "ORIGIN_AIRPORT_ID": "int32",
    "DEST_AIRPORT_ID": "int32",
    "ORIGIN_CITY_MARKET_ID": "int32",
    "DEST_CITY_MARKET_ID": "int32",
    "ORIGIN_STATE_FIPS": "int16",
    "DEST_STATE_FIPS": "int16",
}


def process_db1b_zip(
    zip_path: Path,
    out_dir: Path,
    year: int,
    quarter: int,
    logger: logging.Logger,
    fare_lo: float = 20.0,
    fare_hi: float = 5000.0,
    min_passengers: float = 1.0,
    domestic_only: bool = True,
    carriers_of_interest: Optional[set] = None,
) -> Path:
    """
    Extract DB1B_Market CSV from zip, clean, and write Parquet.
    Returns path to output Parquet file.
    """
    out_path = out_dir / f"db1b_{year}q{quarter}.parquet"
    if out_path.exists():
        logger.info(f"PARQUET EXISTS  {out_path.name} — skipping processing")
        return out_path

    logger.info(f"PROCESSING DB1B  {year}Q{quarter}")
    with zipfile.ZipFile(zip_path) as zf:
        # find the Market CSV inside the zip
        market_files = [n for n in zf.namelist() if "Market" in n and n.endswith(".csv")]
        if not market_files:
            raise ValueError(f"No Market CSV found in {zip_path}")
        csv_name = market_files[0]
        logger.info(f"  Reading {csv_name} from zip…")

        # Read in chunks to cap peak RAM
        chunks = []
        with zf.open(csv_name) as raw:
            reader = pd.read_csv(
                raw,
                usecols=[c for c in DB1B_COLS if c in _db1b_probe_cols(zf, csv_name)],
                dtype={k: v for k, v in DB1B_DTYPES.items()
                       if k in DB1B_COLS},
                chunksize=500_000,
                low_memory=True,
            )
            for chunk in reader:
                # ── filters ──────────────────────────────────────────────────
                if domestic_only and "ORIGIN_COUNTRY" in chunk.columns:
                    chunk = chunk[
                        (chunk["ORIGIN_COUNTRY"] == "US") &
                        (chunk["DEST_COUNTRY"] == "US")
                    ]
                if "MARKET_FARE" in chunk.columns:
                    chunk = chunk[
                        chunk["MARKET_FARE"].between(fare_lo, fare_hi)
                    ]
                if "PASSENGERS" in chunk.columns:
                    chunk = chunk[chunk["PASSENGERS"] >= min_passengers]
                if "BULKFARES" in chunk.columns:
                    chunk = chunk[chunk["BULKFARES"] == 0]  # exclude bulk fares
                if "ROUNDTRIP" in chunk.columns:
                    chunk = chunk[chunk["ROUNDTRIP"] == 1]  # directional markets
                if carriers_of_interest and "OPERATING_CARRIER" in chunk.columns:
                    # Keep rows where at least one of the focus carriers is present
                    pass  # keep all; filter at merge stage

                # ── rename ────────────────────────────────────────────────────
                present = {k: v for k, v in DB1B_COLS.items() if k in chunk.columns}
                chunk = chunk[list(present.keys())].rename(columns=present)

                # ── directional city-pair market ID ───────────────────────────
                if "origin_mkt_id" in chunk.columns and "dest_mkt_id" in chunk.columns:
                    chunk["market_id"] = (
                        chunk["origin_mkt_id"].astype(str) + "_" +
                        chunk["dest_mkt_id"].astype(str)
                    )

                chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    # ── Aggregate to carrier-market-quarter level ─────────────────────────────
    # Weighted median fare (more robust than mean for BLP price variable)
    logger.info(f"  Aggregating {len(df):,} rows → carrier-market level…")

    def wtd_median(grp):
        fares = grp["avg_fare"].values
        wts = grp["passengers"].values
        idx = np.argsort(fares)
        fares, wts = fares[idx], wts[idx]
        cum = np.cumsum(wts)
        cutoff = cum[-1] / 2.0
        return float(fares[np.searchsorted(cum, cutoff)])

    group_cols = ["year", "quarter", "market_id", "origin_mkt_id", "dest_mkt_id",
                  "op_carrier", "n_connections"]
    agg = (
        df.groupby(group_cols, observed=True)
        .apply(
            lambda g: pd.Series({
                "passengers": g["passengers"].sum(),
                "wtd_median_fare": wtd_median(g),
                "avg_fare": np.average(g["avg_fare"], weights=g["passengers"]),
                "p10_fare": np.percentile(g["avg_fare"], 10),
                "mkt_miles": g["mkt_miles"].median(),
                "nonstop_miles": g["nonstop_miles"].median(),
                "mkt_distance": g["mkt_distance"].median(),
                "n_itineraries": len(g),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    # ── Carrier dummies ────────────────────────────────────────────────────────
    LEGACY_CARRIERS = {"UA", "AA", "DL", "WN", "AS", "B6", "NK", "F9", "G4", "SY"}
    LCC = {"WN", "NK", "F9", "G4", "SY", "B6"}
    agg["is_lcc"] = agg["op_carrier"].isin(LCC).astype("int8")
    agg["is_legacy"] = agg["op_carrier"].isin(LEGACY_CARRIERS - LCC).astype("int8")
    agg["is_ua"] = (agg["op_carrier"] == "UA").astype("int8")
    agg["is_aa"] = (agg["op_carrier"] == "AA").astype("int8")
    agg["is_dl"] = (agg["op_carrier"] == "DL").astype("int8")
    agg["is_wn"] = (agg["op_carrier"] == "WN").astype("int8")

    pq.write_table(
        pa.Table.from_pandas(agg),
        out_path,
        compression="zstd",
        compression_level=3,
    )
    logger.info(f"  WROTE {out_path.name}  ({out_path.stat().st_size/1e6:.1f} MB, "
                f"{len(agg):,} rows)")
    del df, chunks, agg
    return out_path


def _db1b_probe_cols(zf: zipfile.ZipFile, csv_name: str) -> set:
    """Read only header row to discover available columns."""
    with zf.open(csv_name) as raw:
        header = pd.read_csv(raw, nrows=0)
    return set(header.columns)


# ── BTS T-100 Domestic Segment ────────────────────────────────────────────────
# URL pattern for pre-assembled annual files from BTS TranStats
T100_BASE = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{m}.zip"
# Note: T-100 segment data has a different URL pattern:
T100_SEG_BASE = "https://transtats.bts.gov/PREZIP/T_T100D_SEGMENT_US_CARRIER_ONLY_{year}.zip"

T100_COLS = {
    "YEAR": "year",
    "QUARTER": "quarter",
    "MONTH": "month",
    "UNIQUE_CARRIER": "carrier",
    "ORIGIN": "origin",
    "DEST": "dest",
    "ORIGIN_CITY_MARKET_ID": "origin_mkt_id",
    "DEST_CITY_MARKET_ID": "dest_mkt_id",
    "AIRCRAFT_TYPE": "aircraft_type",
    "SEATS": "seats",
    "PASSENGERS": "passengers_t100",
    "FREIGHT": "freight",
    "MAIL": "mail",
    "DISTANCE": "distance_t100",
    "RAMP_TO_RAMP": "ramp_time",
    "AIR_TIME": "air_time",
    "DEPARTURES_SCHEDULED": "dep_scheduled",
    "DEPARTURES_PERFORMED": "dep_performed",
    "PAYLOAD": "payload",
    "TRANS_LOAD": "trans_load",
    "CLASS": "service_class",
}

T100_DTYPES = {
    "YEAR": "int16", "QUARTER": "int8", "MONTH": "int8",
    "SEATS": "float32", "PASSENGERS": "float32",
    "DISTANCE": "float32", "RAMP_TO_RAMP": "float32",
    "AIR_TIME": "float32", "DEPARTURES_SCHEDULED": "float32",
    "DEPARTURES_PERFORMED": "float32", "PAYLOAD": "float32",
    "ORIGIN_CITY_MARKET_ID": "int32", "DEST_CITY_MARKET_ID": "int32",
}


def process_t100_zip(
    zip_path: Path,
    out_dir: Path,
    year: int,
    logger: logging.Logger,
) -> Path:
    out_path = out_dir / f"t100_{year}.parquet"
    if out_path.exists():
        logger.info(f"PARQUET EXISTS  {out_path.name} — skipping processing")
        return out_path

    logger.info(f"PROCESSING T-100  {year}")
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(csv_name) as raw:
            probe = pd.read_csv(raw, nrows=0)
            avail = set(probe.columns)

        with zf.open(csv_name) as raw:
            chunks = []
            for chunk in pd.read_csv(
                raw,
                usecols=[c for c in T100_COLS if c in avail],
                dtype={k: v for k, v in T100_DTYPES.items() if k in avail},
                chunksize=500_000,
                low_memory=True,
            ):
                # Keep only scheduled passenger service (Class F or G)
                if "CLASS" in chunk.columns:
                    chunk = chunk[chunk["CLASS"].isin(["F", "G"])]
                chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    present = {k: v for k, v in T100_COLS.items() if k in df.columns}
    df = df[list(present.keys())].rename(columns=present)

    # Aggregate to carrier-route-quarter
    grp_cols = ["year", "quarter", "carrier", "origin_mkt_id", "dest_mkt_id"]
    agg = df.groupby(grp_cols, observed=True).agg(
        seats=("seats", "sum"),
        passengers_t100=("passengers_t100", "sum"),
        dep_performed=("dep_performed", "sum"),
        dep_scheduled=("dep_scheduled", "sum"),
        distance_t100=("distance_t100", "mean"),
        air_time=("air_time", "mean"),
    ).reset_index()

    agg["load_factor"] = (agg["passengers_t100"] / agg["seats"].replace(0, np.nan)).clip(0, 1)
    agg["weekly_freq"] = agg["dep_performed"] / 13.0  # ~13 weeks per quarter

    agg["market_id"] = (
        agg["origin_mkt_id"].astype(str) + "_" +
        agg["dest_mkt_id"].astype(str)
    )

    pq.write_table(
        pa.Table.from_pandas(agg), out_path,
        compression="zstd", compression_level=3,
    )
    logger.info(f"  WROTE {out_path.name}  ({out_path.stat().st_size/1e6:.1f} MB, "
                f"{len(agg):,} rows)")
    del df, chunks, agg
    return out_path


# ── BTS Form 41 Schedule P-12a (Operating Costs) ─────────────────────────────
# Available via BTS TranStats bulk download
FORM41_URL = "https://transtats.bts.gov/PREZIP/Form_41_Schedules_P_12a_{year}.zip"

FORM41_COLS = {
    "YEAR": "year",
    "QUARTER": "quarter",
    "UNIQUE_CARRIER": "carrier",
    "CARRIER_NAME": "carrier_name",
    "AIRCRAFT_TYPE": "aircraft_type",
    "AIRCRAFT_GROUP": "aircraft_group",
    "CARRIER_GROUP": "carrier_group",
    # Cost components
    "FLYING_OPS": "cost_flying_ops",       # direct flying operations
    "MAINTENANCE": "cost_maintenance",
    "TOTAL_DIRECT": "cost_direct",
    "INDIRECT_COSTS": "cost_indirect",
    "TOTAL_COST": "cost_total",
    "ASM": "asm",                          # available seat miles
    "RPM": "rpm",                          # revenue passenger miles
    "FUEL_COST": "fuel_cost",
    "FUEL_QTY": "fuel_qty_gallons",
    "SALARIES_MGMT": "salaries_mgmt",
    "SALARIES_FLIGHT": "salaries_flight",
    "TRANS_RELATED": "cost_transport",
}

FORM41_DTYPES = {k: "float32" for k in [
    "FLYING_OPS", "MAINTENANCE", "TOTAL_DIRECT", "INDIRECT_COSTS",
    "TOTAL_COST", "ASM", "RPM", "FUEL_COST", "FUEL_QTY",
    "SALARIES_MGMT", "SALARIES_FLIGHT", "TRANS_RELATED",
]}
FORM41_DTYPES.update({"YEAR": "int16", "QUARTER": "int8"})


def process_form41_zip(
    zip_path: Path,
    out_dir: Path,
    year: int,
    logger: logging.Logger,
) -> Path:
    out_path = out_dir / f"form41_{year}.parquet"
    if out_path.exists():
        logger.info(f"PARQUET EXISTS  {out_path.name} — skipping processing")
        return out_path

    logger.info(f"PROCESSING Form 41  {year}")
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(csv_name) as raw:
            probe = pd.read_csv(raw, nrows=0)
            avail = set(probe.columns)
        with zf.open(csv_name) as raw:
            df = pd.read_csv(
                raw,
                usecols=[c for c in FORM41_COLS if c in avail],
                dtype={k: v for k, v in FORM41_DTYPES.items() if k in avail},
                low_memory=True,
            )

    present = {k: v for k, v in FORM41_COLS.items() if k in df.columns}
    df = df[list(present.keys())].rename(columns=present)

    # Derived cost metrics useful as instruments / cost shifters
    if "fuel_cost" in df.columns and "asm" in df.columns:
        df["fuel_cost_per_asm"] = df["fuel_cost"] / df["asm"].replace(0, np.nan)
    if "cost_total" in df.columns and "asm" in df.columns:
        df["cost_per_asm"] = df["cost_total"] / df["asm"].replace(0, np.nan)
    if "fuel_qty_gallons" in df.columns and "asm" in df.columns:
        df["fuel_gal_per_asm"] = df["fuel_qty_gallons"] / df["asm"].replace(0, np.nan)

    pq.write_table(
        pa.Table.from_pandas(df), out_path,
        compression="zstd", compression_level=3,
    )
    logger.info(f"  WROTE {out_path.name}  ({out_path.stat().st_size/1e6:.1f} MB)")
    del df
    return out_path


# ── EIA Jet Fuel Prices ────────────────────────────────────────────────────────
# Weekly kerosene-type jet fuel prices (dollars per gallon), U.S. average
EIA_JET_FUEL_URL = (
    "https://www.eia.gov/dnav/pet/hist_xls/EER_EPJK_PF4_RGC_DPGw.xls"
)


def download_eia_fuel(
    raw_dir: Path,
    out_dir: Path,
    cache: DownloadCache,
    logger: logging.Logger,
) -> Path:
    out_path = out_dir / "eia_jet_fuel_quarterly.parquet"
    raw_path = raw_dir / "eia_jet_fuel.xls"

    if out_path.exists():
        logger.info("PARQUET EXISTS  eia_jet_fuel_quarterly.parquet — skipping")
        return out_path

    download_file(EIA_JET_FUEL_URL, raw_path, cache, logger)

    logger.info("PROCESSING EIA jet fuel prices…")
    df = pd.read_excel(raw_path, sheet_name="Data 1", skiprows=2, header=0)
    df.columns = ["date", "jet_fuel_usd_per_gal"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["jet_fuel_usd_per_gal"])
    df["year"] = df["date"].dt.year.astype("int16")
    df["quarter"] = df["date"].dt.quarter.astype("int8")

    # Quarterly average price
    qtr = (
        df.groupby(["year", "quarter"])["jet_fuel_usd_per_gal"]
        .mean()
        .reset_index()
        .rename(columns={"jet_fuel_usd_per_gal": "jet_fuel_qtr_avg"})
    )
    pq.write_table(pa.Table.from_pandas(qtr), out_path,
                   compression="zstd", compression_level=3)
    logger.info(f"  WROTE {out_path.name}")
    return out_path


# ── BTS Airport Coordinates (for great-circle distance) ───────────────────────
AIRPORTS_URL = (
    "https://raw.githubusercontent.com/datasets/airport-codes/master/data/airport-codes.csv"
)


def download_airports(
    raw_dir: Path,
    out_dir: Path,
    cache: DownloadCache,
    logger: logging.Logger,
) -> Path:
    out_path = out_dir / "airports.parquet"
    raw_path = raw_dir / "airport_codes.csv"
    if out_path.exists():
        return out_path

    download_file(AIRPORTS_URL, raw_path, cache, logger)
    logger.info("PROCESSING airport coordinates…")

    df = pd.read_csv(raw_path, low_memory=True)
    df = df[
        (df["type"].isin(["large_airport", "medium_airport"])) &
        (df["iso_country"] == "US")
    ][["ident", "name", "latitude_deg", "longitude_deg", "municipality",
       "iata_code"]].copy()
    df = df.dropna(subset=["iata_code", "latitude_deg", "longitude_deg"])
    df.columns = ["icao", "airport_name", "lat", "lon", "city", "iata"]

    pq.write_table(pa.Table.from_pandas(df), out_path,
                   compression="zstd", compression_level=3)
    return out_path


# ── BTS CBSA / MSA Population ──────────────────────────────────────────────────
# Used to compute market size (gravity model denominator)
# Census Bureau CBSA population estimates
CBSA_POP_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2020-2023/metro/totals/cbsa-est2023.csv"
)


def download_cbsa_pop(
    raw_dir: Path,
    out_dir: Path,
    cache: DownloadCache,
    logger: logging.Logger,
) -> Path:
    out_path = out_dir / "cbsa_pop.parquet"
    raw_path = raw_dir / "cbsa_pop.csv"
    if out_path.exists():
        return out_path

    download_file(CBSA_POP_URL, raw_path, cache, logger)
    logger.info("PROCESSING CBSA population…")

    df = pd.read_csv(raw_path, encoding="latin-1", low_memory=True)
    # Keep metropolitan statistical areas only (LSAD == "M1")
    if "LSAD" in df.columns:
        df = df[df["LSAD"] == "M1"]

    # Identify year columns  (POPESTIMATE20XX)
    pop_cols = [c for c in df.columns if c.startswith("POPESTIMATE")]
    keep = ["CBSA", "NAME"] + pop_cols
    df = df[[c for c in keep if c in df.columns]].copy()

    # Melt to long format
    df_long = df.melt(id_vars=["CBSA", "NAME"], value_vars=pop_cols,
                      var_name="year_raw", value_name="population")
    df_long["year"] = df_long["year_raw"].str.extract(r"(\d{4})").astype("int16")
    df_long = df_long.drop(columns="year_raw")
    df_long["population"] = pd.to_numeric(df_long["population"], errors="coerce")
    df_long = df_long.dropna(subset=["population"])
    df_long["CBSA"] = df_long["CBSA"].astype("int32")

    pq.write_table(pa.Table.from_pandas(df_long), out_path,
                   compression="zstd", compression_level=3)
    logger.info(f"  WROTE {out_path.name}")
    return out_path


# ── Hub indicator table ────────────────────────────────────────────────────────
# Manually curated hub assignments for major US carriers
# (origin of Berry 1990, BCS 2006 hub variable)

HUB_DATA = {
    # carrier: [hub airport IATA codes]
    "UA": ["ORD", "IAH", "DEN", "SFO", "EWR", "LAX", "IAD"],
    "AA": ["DFW", "ORD", "MIA", "CLT", "PHL", "PHX", "LAX", "JFK", "BOS"],
    "DL": ["ATL", "DTW", "MSP", "SLC", "SEA", "JFK", "LAX", "BOS"],
    "WN": ["DAL", "HOU", "MDW", "BWI", "PHX", "LAS", "DEN", "OAK"],
    "AS": ["SEA", "PDX", "SFO", "LAX", "ANC"],
    "B6": ["JFK", "BOS", "FLL", "LGB", "SJC"],
    "NK": ["FLL", "MCO", "LAS", "DTW", "ORD"],
}


def build_hub_table(out_dir: Path, logger: logging.Logger) -> Path:
    out_path = out_dir / "carrier_hubs.parquet"
    if out_path.exists():
        return out_path
    logger.info("BUILDING hub indicator table…")
    rows = []
    for carrier, hubs in HUB_DATA.items():
        for hub in hubs:
            rows.append({"carrier": carrier, "hub_airport": hub, "is_hub": 1})
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df), out_path,
                   compression="zstd", compression_level=3)
    return out_path


# ── Chunked merge of DB1B + T-100 + instruments ───────────────────────────────

def build_analysis_dataset(
    processed_dir: Path,
    output_dir: Path,
    years: list,
    logger: logging.Logger,
    markets_per_chunk: int = 5000,
) -> Path:
    """
    Merge DB1B, T-100, fuel prices, hub indicators, and CBSA pop
    in chunks of `markets_per_chunk` markets to avoid OOM.
    Writes final Parquet dataset to output_dir/analysis_panel.parquet.
    """
    final_path = output_dir / "analysis_panel.parquet"
    if final_path.exists():
        logger.info(f"FINAL DATASET EXISTS  {final_path} — skipping merge")
        return final_path

    logger.info("BUILDING analysis panel (chunked merge)…")

    # ── Load small reference tables fully into RAM (these are tiny) ───────────
    fuel = pq.read_table(processed_dir / "eia_jet_fuel_quarterly.parquet").to_pandas()
    hubs = pq.read_table(processed_dir / "carrier_hubs.parquet").to_pandas()

    cbsa_path = processed_dir / "cbsa_pop.parquet"
    pop = pq.read_table(cbsa_path).to_pandas() if cbsa_path.exists() else None

    # ── Collect all DB1B parquet paths ────────────────────────────────────────
    db1b_files = sorted(processed_dir.glob("db1b_*.parquet"))
    t100_files = {
        int(p.stem.split("_")[1]): p
        for p in sorted(processed_dir.glob("t100_*.parquet"))
    }
    if not db1b_files:
        raise FileNotFoundError("No DB1B parquet files found — run downloads first.")

    # ── Discover all unique market_ids ────────────────────────────────────────
    logger.info("  Scanning market IDs across all DB1B files…")
    all_market_ids = set()
    for fp in db1b_files:
        tbl = pq.read_table(fp, columns=["market_id"])
        all_market_ids.update(tbl.column("market_id").to_pylist())
    market_ids = sorted(all_market_ids)
    logger.info(f"  Total unique markets: {len(market_ids):,}")

    # ── Process in chunks ─────────────────────────────────────────────────────
    chunk_paths = []
    n_chunks = (len(market_ids) + markets_per_chunk - 1) // markets_per_chunk

    for chunk_idx in tqdm(range(n_chunks), desc="Merging chunks"):
        chunk_mids = set(market_ids[
            chunk_idx * markets_per_chunk:(chunk_idx + 1) * markets_per_chunk
        ])
        chunk_out = output_dir / f"chunk_{chunk_idx:04d}.parquet"

        if chunk_out.exists():
            chunk_paths.append(chunk_out)
            continue

        # Load DB1B rows for this chunk of markets
        db1b_parts = []
        for fp in db1b_files:
            tbl = pq.read_table(fp)
            df_q = tbl.to_pandas()
            df_q = df_q[df_q["market_id"].isin(chunk_mids)]
            if not df_q.empty:
                db1b_parts.append(df_q)
            del tbl, df_q

        if not db1b_parts:
            continue
        db1b_chunk = pd.concat(db1b_parts, ignore_index=True)
        del db1b_parts

        # Load T-100 rows for same markets
        t100_parts = []
        for yr in years:
            if yr in t100_files:
                tbl = pq.read_table(t100_files[yr])
                df_q = tbl.to_pandas()
                df_q = df_q[df_q["market_id"].isin(chunk_mids)]
                if not df_q.empty:
                    t100_parts.append(df_q)
                del tbl, df_q

        # Merge T-100 frequency and load factor
        if t100_parts:
            t100_chunk = pd.concat(t100_parts, ignore_index=True)
            t100_chunk = t100_chunk.rename(columns={"carrier": "op_carrier"})
            db1b_chunk = db1b_chunk.merge(
                t100_chunk[["year", "quarter", "op_carrier", "market_id",
                            "seats", "dep_performed", "weekly_freq", "load_factor"]],
                on=["year", "quarter", "op_carrier", "market_id"],
                how="left",
            )
            del t100_chunk, t100_parts

        # Merge fuel prices
        db1b_chunk = db1b_chunk.merge(fuel, on=["year", "quarter"], how="left")

        # Fuel cost instrument: fuel price × route distance
        if "jet_fuel_qtr_avg" in db1b_chunk.columns and "mkt_distance" in db1b_chunk.columns:
            db1b_chunk["fuel_cost_iv"] = (
                db1b_chunk["jet_fuel_qtr_avg"] * db1b_chunk["mkt_distance"]
            )

        # Merge hub indicator for origin and destination
        origin_hubs = hubs.rename(columns={
            "carrier": "op_carrier", "hub_airport": "origin_iata"
        })[["op_carrier", "origin_iata", "is_hub"]].copy()
        origin_hubs = origin_hubs.rename(columns={"is_hub": "origin_is_hub"})

        dest_hubs = hubs.rename(columns={
            "carrier": "op_carrier", "hub_airport": "dest_iata"
        })[["op_carrier", "dest_iata", "is_hub"]].copy()
        dest_hubs = dest_hubs.rename(columns={"is_hub": "dest_is_hub"})

        if "origin" in db1b_chunk.columns:
            db1b_chunk = db1b_chunk.merge(
                origin_hubs.rename(columns={"origin_iata": "origin"}),
                on=["op_carrier", "origin"], how="left",
            )
            db1b_chunk["origin_is_hub"] = db1b_chunk["origin_is_hub"].fillna(0).astype("int8")

        if "dest" in db1b_chunk.columns:
            db1b_chunk = db1b_chunk.merge(
                dest_hubs.rename(columns={"dest_iata": "dest"}),
                on=["op_carrier", "dest"], how="left",
            )
            db1b_chunk["dest_is_hub"] = db1b_chunk["dest_is_hub"].fillna(0).astype("int8")

        # Hub-to-hub route indicator (key for BCS-style hub premium)
        if "origin_is_hub" in db1b_chunk.columns and "dest_is_hub" in db1b_chunk.columns:
            db1b_chunk["hub_to_hub"] = (
                (db1b_chunk["origin_is_hub"] == 1) &
                (db1b_chunk["dest_is_hub"] == 1)
            ).astype("int8")

        # UA-AA overlap indicator (target merger pair)
        carriers_in_mkt = (
            db1b_chunk.groupby("market_id")["op_carrier"]
            .apply(set)
            .reset_index()
            .rename(columns={"op_carrier": "carriers_in_market"})
        )
        carriers_in_mkt["ua_aa_overlap"] = carriers_in_mkt["carriers_in_market"].apply(
            lambda s: int({"UA", "AA"}.issubset(s))
        ).astype("int8")
        db1b_chunk = db1b_chunk.merge(
            carriers_in_mkt[["market_id", "ua_aa_overlap"]],
            on="market_id", how="left",
        )

        # Number of competitors per market-quarter (market structure variable)
        n_carriers = (
            db1b_chunk.groupby(["market_id", "year", "quarter"])["op_carrier"]
            .nunique()
            .reset_index()
            .rename(columns={"op_carrier": "n_carriers"})
        )
        db1b_chunk = db1b_chunk.merge(
            n_carriers, on=["market_id", "year", "quarter"], how="left"
        )

        # Market share (within market-quarter)
        mkt_total_pax = (
            db1b_chunk.groupby(["market_id", "year", "quarter"])["passengers"]
            .sum()
            .reset_index()
            .rename(columns={"passengers": "mkt_total_pax"})
        )
        db1b_chunk = db1b_chunk.merge(
            mkt_total_pax, on=["market_id", "year", "quarter"], how="left"
        )
        db1b_chunk["share_inside"] = (
            db1b_chunk["passengers"] / db1b_chunk["mkt_total_pax"]
        ).clip(0, 1).astype("float32")

        # Log variables for BLP
        db1b_chunk["ln_fare"] = np.log(db1b_chunk["wtd_median_fare"].clip(lower=1))
        db1b_chunk["ln_distance"] = np.log(db1b_chunk["mkt_distance"].clip(lower=1))
        db1b_chunk["ln_freq"] = np.log(
            db1b_chunk["weekly_freq"].clip(lower=0.01)
        ) if "weekly_freq" in db1b_chunk.columns else np.nan

        # Downcast floats to float32 to save memory
        for col in db1b_chunk.select_dtypes("float64").columns:
            db1b_chunk[col] = db1b_chunk[col].astype("float32")

        pq.write_table(
            pa.Table.from_pandas(db1b_chunk),
            chunk_out,
            compression="zstd",
            compression_level=3,
        )
        chunk_paths.append(chunk_out)
        logger.info(
            f"  Chunk {chunk_idx+1}/{n_chunks}: {len(db1b_chunk):,} rows "
            f"→ {chunk_out.name} ({chunk_out.stat().st_size/1e6:.1f} MB)"
        )
        del db1b_chunk

    # ── Combine chunks into final dataset using PyArrow (zero-copy concat) ────
    logger.info(f"CONCATENATING {len(chunk_paths)} chunks into final parquet…")
    schema = pq.read_schema(chunk_paths[0])
    writer = pq.ParquetWriter(final_path, schema, compression="zstd")
    for cp in tqdm(chunk_paths, desc="Writing final parquet"):
        tbl = pq.read_table(cp)
        writer.write_table(tbl)
        del tbl
    writer.close()

    # Clean up chunk files
    for cp in chunk_paths:
        cp.unlink()

    logger.info(f"FINAL DATASET  {final_path}  ({final_path.stat().st_size/1e6:.1f} MB)")
    return final_path


# ── Dataset summary ────────────────────────────────────────────────────────────

def print_summary(final_path: Path, logger: logging.Logger):
    logger.info("─" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("─" * 60)
    tbl = pq.read_table(final_path)
    df = tbl.to_pandas()
    logger.info(f"  Rows:          {len(df):,}")
    logger.info(f"  Columns:       {len(df.columns)}")
    logger.info(f"  Years:         {sorted(df['year'].unique())}")
    if "quarter" in df.columns:
        logger.info(f"  Quarters:      {sorted(df['quarter'].unique())}")
    logger.info(f"  Unique markets:{df['market_id'].nunique():,}")
    logger.info(f"  Carriers:      {sorted(df['op_carrier'].unique())}")
    if "ua_aa_overlap" in df.columns:
        n_overlap = df[df["ua_aa_overlap"] == 1]["market_id"].nunique()
        logger.info(f"  UA-AA overlap markets: {n_overlap:,}")
    logger.info(f"  Memory usage:  {df.memory_usage(deep=True).sum()/1e9:.2f} GB")
    logger.info("─" * 60)
    del df, tbl


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect BTS data for UA/AA merger structural estimation"
    )
    p.add_argument(
        "--years", nargs="+", type=int,
        default=[2018, 2019, 2022, 2023],
        help="Years to download (default: 2018 2019 2022 2023)",
    )
    p.add_argument(
        "--quarters", nargs="+", type=int,
        default=[1, 2, 3, 4],
        help="Quarters to download for DB1B (default: all 4)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("./merger_data"),
        help="Root output directory",
    )
    p.add_argument(
        "--markets-per-chunk", type=int, default=5000,
        help="Markets per merge chunk (tune to RAM; lower = less RAM, more chunks)",
    )
    p.add_argument(
        "--skip-form41", action="store_true",
        help="Skip Form 41 download (large; not required for baseline BLP estimation)",
    )
    p.add_argument(
        "--summary-only", action="store_true",
        help="Skip all downloads; just print summary of existing final dataset",
    )
    return p.parse_args()


def main():
    args = parse_args()

    root = args.output_dir
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    final_dir = root / "final"
    log_dir = root / "logs"

    for d in [raw_dir, processed_dir, final_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir)
    cache = DownloadCache(root / "download_cache.json")

    logger.info("=" * 60)
    logger.info("UA/AA MERGER DATA COLLECTION PIPELINE")
    logger.info(f"  Years: {args.years}")
    logger.info(f"  Quarters: {args.quarters}")
    logger.info(f"  Output: {root.resolve()}")
    logger.info("=" * 60)

    if args.summary_only:
        final_path = final_dir / "analysis_panel.parquet"
        print_summary(final_path, logger)
        return

    # ── 1. DB1B ───────────────────────────────────────────────────────────────
    logger.info("\n[1/5] DB1B Origin-Destination Survey")
    for year in args.years:
        for quarter in args.quarters:
            url = DB1B_BASE.format(year=year, q=quarter)
            zip_path = raw_dir / f"db1b_{year}q{quarter}.zip"
            newly_downloaded = download_file(url, zip_path, cache, logger)
            if newly_downloaded or not (processed_dir / f"db1b_{year}q{quarter}.parquet").exists():
                try:
                    process_db1b_zip(zip_path, processed_dir, year, quarter, logger)
                except Exception as e:
                    logger.error(f"DB1B processing failed {year}Q{quarter}: {e}")

    # ── 2. T-100 ──────────────────────────────────────────────────────────────
    logger.info("\n[2/5] T-100 Domestic Segment")
    for year in args.years:
        url = T100_SEG_BASE.format(year=year)
        zip_path = raw_dir / f"t100_{year}.zip"
        newly_downloaded = download_file(url, zip_path, cache, logger)
        if newly_downloaded or not (processed_dir / f"t100_{year}.parquet").exists():
            try:
                process_t100_zip(zip_path, processed_dir, year, logger)
            except Exception as e:
                logger.error(f"T-100 processing failed {year}: {e}")

    # ── 3. Form 41 ────────────────────────────────────────────────────────────
    if not args.skip_form41:
        logger.info("\n[3/5] Form 41 Operating Costs")
        for year in args.years:
            url = FORM41_URL.format(year=year)
            zip_path = raw_dir / f"form41_{year}.zip"
            newly_downloaded = download_file(url, zip_path, cache, logger)
            if newly_downloaded or not (processed_dir / f"form41_{year}.parquet").exists():
                try:
                    process_form41_zip(zip_path, processed_dir, year, logger)
                except Exception as e:
                    logger.error(f"Form 41 processing failed {year}: {e}")
    else:
        logger.info("\n[3/5] Form 41 — SKIPPED (--skip-form41)")

    # ── 4. Reference datasets ─────────────────────────────────────────────────
    logger.info("\n[4/5] Reference datasets (fuel, airports, CBSA pop, hubs)")
    try:
        download_eia_fuel(raw_dir, processed_dir, cache, logger)
    except Exception as e:
        logger.error(f"EIA fuel download failed: {e}")

    try:
        download_airports(raw_dir, processed_dir, cache, logger)
    except Exception as e:
        logger.error(f"Airport coordinates download failed: {e}")

    try:
        download_cbsa_pop(raw_dir, processed_dir, cache, logger)
    except Exception as e:
        logger.error(f"CBSA population download failed: {e}")

    try:
        build_hub_table(processed_dir, logger)
    except Exception as e:
        logger.error(f"Hub table build failed: {e}")

    # ── 5. Chunked merge ──────────────────────────────────────────────────────
    logger.info("\n[5/5] Chunked merge → analysis panel")
    try:
        final_path = build_analysis_dataset(
            processed_dir=processed_dir,
            output_dir=final_dir,
            years=args.years,
            logger=logger,
            markets_per_chunk=args.markets_per_chunk,
        )
        print_summary(final_path, logger)
    except Exception as e:
        logger.error(f"Final merge failed: {e}", exc_info=True)
        raise

    logger.info("\nPIPELINE COMPLETE.")
    logger.info(f"Analysis dataset: {final_dir / 'analysis_panel.parquet'}")
    logger.info(f"Cache manifest:   {root / 'download_cache.json'}")


if __name__ == "__main__":
    main()