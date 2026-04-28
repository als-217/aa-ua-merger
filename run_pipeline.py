#!/usr/bin/env python3
"""
run_pipeline.py — Master orchestration script for the airline merger data pipeline.

Calls download_db1b.py, download_t100.py, download_form41.py, and
download_supporting.py in sequence (or selectively), forwarding a
consistent set of arguments to each.

Each sub-script is launched as a subprocess so that sys.exit() calls,
tqdm output, and log files all behave exactly as if you ran each script
directly. Exit codes are captured and a final report is printed.

Usage examples
--------------
# Full pipeline, 2019–present (recommended starting point)
  python run_pipeline.py --start-year 2019

# Only the most recent two years, only DB1B and T-100
  python run_pipeline.py --start-year 2023 --steps db1b t100

# Specific quarters of 2024, force re-download
  python run_pipeline.py --start-year 2024 --end-year 2024 \\
      --quarters 3 4 --skip-cache

# Minimal smoke-test: one quarter, market table only
  python run_pipeline.py --start-year 2024 --end-year 2024 \\
      --quarters 4 --db1b-tables market --t100-tables segment \\
      --form41-tables p12 --steps db1b t100 form41

# Supporting data only (populations, fuel prices, lookups)
  python run_pipeline.py --steps supporting \\
      --census-key YOUR_KEY --eia-key YOUR_KEY

# Full pipeline, dry run (print commands, do not execute)
  python run_pipeline.py --start-year 2022 --dry-run

Arguments
---------
Global
  --start-year  INT       First year for time-series downloads. Default: 2019
  --end-year    INT       Last year. Default: current year
  --steps       STR…      Which steps to run: db1b t100 form41 supporting
                          Default: all four
  --skip-cache  FLAG      Pass --skip-cache to every sub-script
  --workers     INT       Parallel download threads per script. Default: 2
  --base-dir    PATH      Root directory for all raw/parquet output. Default: .
  --dry-run     FLAG      Print commands that would run; do not execute them

DB1B-specific
  --quarters    INT…      Quarters to download: 1 2 3 4. Default: all
  --db1b-tables STR…      coupon market ticket. Default: all three

T-100-specific
  --t100-tables STR…      segment market. Default: both

Form 41-specific
  --form41-tables STR…    p12 p52 p7 p1 all. Default: p12 p52

Supporting-specific
  --supporting-datasets STR…
                          msa fuel airport aircraft cpi cbsa all
                          Default: all
  --census-key  STR       Census Bureau API key (optional)
  --eia-key     STR       EIA API key (optional)
  --support-start-year INT
                          Override start year for supporting data only.
                          Default: 2010 (population/CPI series need longer history)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── locate sibling scripts relative to this file ──────────────────────────────
_HERE = Path(__file__).parent.resolve()
SCRIPT = {
    "db1b":       _HERE / "download_db1b.py",
    "t100":       _HERE / "download_t100.py",
    "form41":     _HERE / "download_form41.py",
    "supporting": _HERE / "download_supporting.py",
}

ALL_STEPS = ["db1b", "t100", "form41", "supporting"]

# ── colour helpers (gracefully degrade on Windows) ────────────────────────────
try:
    import shutil
    _COLOUR = sys.stdout.isatty() and shutil.which("tput") is not None
except Exception:
    _COLOUR = False

_GREEN  = "\033[32m" if _COLOUR else ""
_RED    = "\033[31m" if _COLOUR else ""
_YELLOW = "\033[33m" if _COLOUR else ""
_BOLD   = "\033[1m"  if _COLOUR else ""
_RESET  = "\033[0m"  if _COLOUR else ""

def _ok(s):  return f"{_GREEN}{_BOLD}{s}{_RESET}"
def _err(s): return f"{_RED}{_BOLD}{s}{_RESET}"
def _warn(s):return f"{_YELLOW}{s}{_RESET}"
def _hdr(s): return f"\n{_BOLD}{'─' * 60}\n  {s}\n{'─' * 60}{_RESET}"


# ── result tracking ───────────────────────────────────────────────────────────
@dataclass
class StepResult:
    step:       str
    cmd:        list[str]
    returncode: int        = -1
    elapsed_s:  float      = 0.0
    skipped:    bool       = False
    error:      str        = ""


# ── argument builder functions ────────────────────────────────────────────────

def _build_db1b_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, str(SCRIPT["db1b"]),
        "--start-year", str(args.start_year),
        "--end-year",   str(args.end_year),
        "--quarters",   *[str(q) for q in args.quarters],
        "--tables",     *args.db1b_tables,
        "--workers",    str(args.workers),
        "--out-dir",    str(Path(args.base_dir) / "parquet" / "db1b"),
        "--raw-dir",    str(Path(args.base_dir) / "db1b"),
    ]
    if args.skip_cache:
        cmd.append("--skip-cache")
    return cmd


def _build_t100_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, str(SCRIPT["t100"]),
        "--start-year", str(args.start_year),
        "--end-year",   str(args.end_year),
        "--tables",     *args.t100_tables,
        "--workers",    str(args.workers),
        "--out-dir",    str(Path(args.base_dir) / "parquet" / "t100"),
        "--raw-dir",    str(Path(args.base_dir) / "t100"),
    ]
    if args.skip_cache:
        cmd.append("--skip-cache")
    return cmd


def _build_form41_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, str(SCRIPT["form41"]),
        "--start-year", str(args.start_year),
        "--end-year",   str(args.end_year),
        "--tables",     *args.form41_tables,
        "--workers",    str(args.workers),
        "--out-dir",    str(Path(args.base_dir) / "parquet" / "form41"),
        "--raw-dir",    str(Path(args.base_dir) / "form41"),
    ]
    if args.skip_cache:
        cmd.append("--skip-cache")
    return cmd


def _build_supporting_cmd(args: argparse.Namespace) -> list[str]:
    start = args.support_start_year if args.support_start_year else args.start_year
    cmd = [
        sys.executable, str(SCRIPT["supporting"]),
        "--start-year", str(start),
        "--end-year",   str(args.end_year),
        "--datasets",   *args.supporting_datasets,
        "--out-dir",    str(Path(args.base_dir) / "parquet" / "supporting"),
        "--raw-dir",    str(Path(args.base_dir) / "supporting"),
    ]
    if args.skip_cache:
        cmd.append("--skip-cache")
    if args.census_key:
        cmd += ["--census-key", args.census_key]
    if args.eia_key:
        cmd += ["--eia-key", args.eia_key]
    return cmd


CMD_BUILDERS = {
    "db1b":       _build_db1b_cmd,
    "t100":       _build_t100_cmd,
    "form41":     _build_form41_cmd,
    "supporting": _build_supporting_cmd,
}

STEP_LABELS = {
    "db1b":       "DB1B  (Origin & Destination Survey)",
    "t100":       "T-100 (Air Carrier Capacity & Traffic)",
    "form41":     "Form 41 (Carrier Financials & Fuel)",
    "supporting": "Supporting Data (MSA / Fuel / CPI / CBSA / Aircraft)",
}


# ── runner ────────────────────────────────────────────────────────────────────

def run_step(step: str, cmd: list[str], dry_run: bool) -> StepResult:
    """Execute one pipeline step and return a StepResult."""
    result = StepResult(step=step, cmd=cmd)

    print(_hdr(STEP_LABELS[step]))
    print(_warn("  Command: ") + " ".join(cmd))

    if dry_run:
        print(_warn("  [DRY RUN] — not executed"))
        result.skipped = True
        result.returncode = 0
        return result

    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, check=False)
        result.returncode = proc.returncode
    except FileNotFoundError as exc:
        result.returncode = -1
        result.error = f"Script not found: {exc}"
        print(_err(f"  ERROR: {result.error}"))
    except KeyboardInterrupt:
        result.returncode = 130
        result.error = "Interrupted by user"
        print(_warn("\n  Interrupted."))
        raise
    finally:
        result.elapsed_s = time.monotonic() - t0

    elapsed = _fmt_elapsed(result.elapsed_s)
    if result.returncode == 0:
        print(_ok(f"  ✓ {step} completed in {elapsed}"))
    else:
        print(_err(f"  ✗ {step} FAILED (exit {result.returncode}) after {elapsed}"))

    return result


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    current_year = datetime.now().year
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=(
            "Master data-acquisition pipeline for airline merger analysis.\n"
            "Calls download_db1b, download_t100, download_form41, and "
            "download_supporting with consistent arguments."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Arguments")[0],   # show usage examples in --help
    )

    # ── Global ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Global")
    g.add_argument(
        "--start-year", type=int, default=2019, metavar="YEAR",
        help="First year for all time-series downloads (default: 2019)",
    )
    g.add_argument(
        "--end-year", type=int, default=current_year, metavar="YEAR",
        help="Last year for all time-series downloads (default: current year)",
    )
    g.add_argument(
        "--steps", nargs="+", default=ALL_STEPS,
        choices=ALL_STEPS, metavar="STEP",
        help=f"Steps to run: {{{', '.join(ALL_STEPS)}}} (default: all)",
    )
    g.add_argument(
        "--skip-cache", action="store_true",
        help="Force re-download even if files are already cached",
    )
    g.add_argument(
        "--workers", type=int, default=2, metavar="N",
        help="Parallel download threads passed to each sub-script (default: 2, max recommended: 3)",
    )
    g.add_argument(
        "--base-dir", type=str, default=".", metavar="PATH",
        help="Root directory for all raw/ and parquet/ output (default: current directory)",
    )
    g.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would run without executing them",
    )
    g.add_argument(
        "--keep-going", action="store_true",
        help="Continue running subsequent steps even if an earlier step fails",
    )

    # ── DB1B ──────────────────────────────────────────────────────────────────
    d = p.add_argument_group("DB1B")
    d.add_argument(
        "--quarters", type=int, nargs="+", default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4], metavar="Q",
        help="Quarters to download (default: 1 2 3 4)",
    )
    d.add_argument(
        "--db1b-tables", nargs="+", default=["coupon", "market", "ticket"],
        choices=["coupon", "market", "ticket"], metavar="TABLE",
        help="DB1B tables: coupon market ticket (default: all three)",
    )

    # ── T-100 ─────────────────────────────────────────────────────────────────
    t = p.add_argument_group("T-100")
    t.add_argument(
        "--t100-tables", nargs="+", default=["segment", "market"],
        choices=["segment", "market"], metavar="TABLE",
        help="T-100 tables: segment market (default: both)",
    )

    # ── Form 41 ───────────────────────────────────────────────────────────────
    f = p.add_argument_group("Form 41")
    f.add_argument(
        "--form41-tables", nargs="+", default=["p12", "p52"],
        choices=["p12", "p52", "p7", "p1", "all"], metavar="TABLE",
        help="Form 41 schedules: p12 p52 p7 p1 all (default: p12 p52)",
    )

    # ── Supporting ────────────────────────────────────────────────────────────
    s = p.add_argument_group("Supporting data")
    s.add_argument(
        "--supporting-datasets", nargs="+",
        default=["msa", "fuel", "airport", "aircraft", "cpi", "cbsa"],
        choices=["msa", "fuel", "airport", "aircraft", "cpi", "cbsa", "all"],
        metavar="DATASET",
        help="Supporting datasets to fetch (default: all six)",
    )
    s.add_argument(
        "--support-start-year", type=int, default=2010, metavar="YEAR",
        help=(
            "Override start year for supporting data only — population and CPI "
            "series benefit from longer history (default: 2010)"
        ),
    )
    s.add_argument(
        "--census-key", type=str, default=None, metavar="KEY",
        help="Census Bureau API key (free at api.census.gov/data/key_signup.html)",
    )
    s.add_argument(
        "--eia-key", type=str, default=None, metavar="KEY",
        help="EIA API key (free at eia.gov/opendata/register.php)",
    )

    return p.parse_args()


# ── validation ────────────────────────────────────────────────────────────────

def validate(args: argparse.Namespace) -> list[str]:
    """Return a list of validation error strings (empty = OK)."""
    errors = []
    if args.start_year > args.end_year:
        errors.append(f"--start-year ({args.start_year}) must be ≤ --end-year ({args.end_year})")
    if args.workers < 1:
        errors.append("--workers must be ≥ 1")
    if args.workers > 4:
        print(_warn("  Warning: --workers > 4 may trigger BTS rate limiting."))
    for step in args.steps:
        if not SCRIPT[step].exists():
            errors.append(f"Script not found: {SCRIPT[step]}")
    return errors


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(results: list[StepResult]) -> None:
    print(_hdr("Pipeline Summary"))
    col_w = max(len(STEP_LABELS[r.step]) for r in results) + 2
    header = f"  {'Step':<{col_w}}  {'Status':<10}  {'Time':>10}"
    print(header)
    print("  " + "─" * (col_w + 24))
    for r in results:
        label = STEP_LABELS[r.step]
        if r.skipped:
            status = _warn("DRY RUN")
            elapsed = "—"
        elif r.returncode == 0:
            status = _ok("OK")
            elapsed = _fmt_elapsed(r.elapsed_s)
        else:
            status = _err(f"FAILED ({r.returncode})")
            elapsed = _fmt_elapsed(r.elapsed_s)
        print(f"  {label:<{col_w}}  {status:<10}  {elapsed:>10}")
    print()


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Validate
    errors = validate(args)
    if errors:
        for e in errors:
            print(_err(f"  ERROR: {e}"))
        sys.exit(2)

    # Print plan
    print(_hdr("Airline Merger Data Pipeline"))
    print(f"  Start year : {args.start_year}")
    print(f"  End year   : {args.end_year}")
    print(f"  Steps      : {', '.join(args.steps)}")
    print(f"  Workers    : {args.workers}")
    print(f"  Base dir   : {Path(args.base_dir).resolve()}")
    print(f"  Skip cache : {args.skip_cache}")
    if args.dry_run:
        print(_warn("  Mode       : DRY RUN — no files will be downloaded"))

    # Run steps
    results: list[StepResult] = []
    try:
        for step in args.steps:
            cmd = CMD_BUILDERS[step](args)
            result = run_step(step, cmd, dry_run=args.dry_run)
            results.append(result)

            if result.returncode != 0 and not result.skipped and not args.keep_going:
                print(_err(
                    f"\n  Step '{step}' failed. "
                    "Use --keep-going to continue on failure."
                ))
                break

    except KeyboardInterrupt:
        print(_warn("\n\nPipeline interrupted by user."))

    # Summary
    print_summary(results)

    # Exit with failure if any step failed
    failed = [r for r in results if r.returncode != 0 and not r.skipped]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
