# Airline Merger Data Pipeline

Data acquisition pipeline for BLP-based structural merger simulation
(United Airlines × American Airlines hypothetical).

## Project Layout

```
airline_data/
├── utils.py                  # Shared: logging, caching, download helpers
├── download_db1b.py          # DB1B: Origin & Destination Survey
├── download_t100.py          # T-100: Air Carrier Statistics (capacity/frequency)
├── download_form41.py        # Form 41: Carrier financials & fuel cost
├── download_supporting.py    # MSA population, EIA fuel prices, CPI, CBSA, airport lookup
├── requirements.txt
│
├── cache/                    # SHA-256 download registry (auto-managed)
├── logs/                     # Rotating log files
│
├── db1b/                     # Raw DB1B zips
├── t100/                     # Raw T-100 zips
├── form41/                   # Raw Form 41 zips
├── supporting/               # Raw supporting files
│
└── parquet/                  # Processed Parquet output (load these downstream)
    ├── db1b/
    │   ├── coupon/           # YYYY_QN.parquet
    │   ├── market/
    │   └── ticket/
    ├── t100/
    │   ├── segment/          # YYYY.parquet
    │   └── market/
    ├── form41/
    │   ├── p12/              # Fuel cost by carrier-month
    │   ├── p52/              # Operating expenses
    │   ├── p7/               # Summary operations
    │   └── p1/               # Balance sheet
    └── supporting/
        ├── msa_population.parquet
        ├── jet_fuel_prices.parquet
        ├── airport_id.parquet
        ├── city_market_id.parquet
        ├── aircraft_characteristics.parquet
        ├── cpi_air_transportation.parquet
        └── cbsa_delineation.parquet
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Download all data from 2019 onward (recommended baseline)

```bash
# DB1B — all three tables, all quarters, 2019–present
python download_db1b.py --start-year 2019

# T-100 — segment and market
python download_t100.py --start-year 2019

# Form 41 — fuel costs + operating expenses
python download_form41.py --start-year 2019 --tables p12 p52 p7

# Supporting: MSA population, fuel prices, lookups, CPI, CBSA
python download_supporting.py --start-year 2010
```

### Minimal download for a quick test run (2024 only)

```bash
python download_db1b.py --start-year 2024 --tables coupon market
python download_t100.py --start-year 2024 --tables segment
python download_form41.py --start-year 2024 --tables p12
python download_supporting.py
```

### Force re-download (ignore cache)

```bash
python download_db1b.py --start-year 2024 --skip-cache
```

### Specific quarters only

```bash
python download_db1b.py --start-year 2023 --end-year 2024 --quarters 3 4
```

### Use API keys for Census / EIA (higher rate limits)

```bash
python download_supporting.py \
    --census-key YOUR_CENSUS_KEY \
    --eia-key YOUR_EIA_KEY
```

Get free keys at:
- Census: https://api.census.gov/data/key_signup.html
- EIA: https://www.eia.gov/opendata/register.php

## Key Variables for BLP Demand Estimation

### From DB1B Market (primary fare table)
| Variable | Description | Role in BLP |
|---|---|---|
| `MktFare` | Prorated market fare | Price variable $p_{jm}$ |
| `Passengers` | Ticket count × 10 | Quantity / market share |
| `MktCoupons` | Coupons in O-D leg | $n_{connections} = MktCoupons - 1$ |
| `NonStopMiles` | Great-circle distance | Product characteristic |
| `TkCarrier` | Ticketing carrier | Firm identity (ownership matrix) |
| `Origin` / `Dest` | Airport codes | Market definition |

### From T-100 Segment (frequency & capacity)
| Variable | Description | Role in BLP |
|---|---|---|
| `DEPARTURES_PERFORMED` | Monthly departures | Flight frequency characteristic |
| `SEATS` | Available seats | Capacity / load factor |
| `LOAD_FACTOR` (derived) | Pax/Seats | Cost-side proxy |
| `AIRCRAFT_TYPE` | DOT type code | Join to FAA seat capacity |

### From Form 41 P-12 (fuel cost instrument)
| Variable | Description | Role in BLP |
|---|---|---|
| `TDOMT_CPC` | Domestic $/gallon | Fuel cost instrument base |
| `FUEL_COST_PER_GAL` (derived) | Computed $/gallon | × Route distance = cost shifter IV |

### Supporting
| Dataset | Key Variable | Use |
|---|---|---|
| MSA Population | `POPULATION` | Market size $M_m$ (gravity model denominator) |
| Jet Fuel Prices | `FUEL_PRICE_PER_GAL` | Time-varying fuel IV (route-level cost shifter) |
| CPI Air Transport | `CPI_AIR_TRANSPORT` | Deflate nominal fares to real |
| CBSA Delineation | CBSA → counties | City-pair market boundary definition |
| Airport Lookup | IATA → CityNum | Aggregate airport-level O-D to city-pair O-D |

## Cache Behaviour

The download registry is stored in `cache/download_registry.json`.
Each entry maps a URL → SHA-256 of the downloaded file. On re-run,
files are skipped if:
1. The destination file exists, AND
2. Its SHA-256 matches the registry entry.

Pass `--skip-cache` to force re-download regardless.

## Memory Notes

- DB1B Market: ~500 MB per quarter on disk; ~1.5 GB in RAM during processing.
  Peak RAM usage: ~6 GB for a single-year annual merge.
- T-100 Segment: ~100 MB per year on disk; minimal RAM footprint.
- Form 41: very small (<50 MB per year).
- Never load all quarters into RAM simultaneously — use per-quarter Parquet
  files and merge only the columns you need with `pd.read_parquet(columns=[...])`.
