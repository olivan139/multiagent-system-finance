#!/usr/bin/env python3
"""Fetch yfinance OHLC for the FNSPID slice tickers and SPY (benchmark).

Reads the tickers and date span from ``results/fnspid/predictions.json``,
downloads daily OHLC for each ticker plus SPY through yfinance, then
caches the joined dataframe to ``results/fnspid/prices.parquet``.

We pull 5 extra trading days on each side of the news date span so that
next-day returns are defined for headlines on the last few news dates and
so that the rolling-mean abnormal-return computation has the warmup it
needs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

from mas.config import RESULTS_DIR
from mas.data import SECTOR_ETFS

DEFAULT_OUT_DIR = RESULTS_DIR / "fnspid"


BENCHMARKS: tuple[str, ...] = ("SPY",) + SECTOR_ETFS


def _read_pred_meta(out_dir: Path) -> dict:
    pred_path = out_dir / "predictions.json"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} missing — run scripts/downstream_signal_test.py first."
        )
    return json.loads(pred_path.read_text())["meta"]["fnspid_slice"]


def _download_one(ticker: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
    """Download daily OHLC for one ticker. Empty frame if everything fails."""
    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Close"]].rename(columns={"Close": "close"})
            df["ticker"] = ticker
            df.index = df.index.tz_localize(None)
            df = df.reset_index().rename(columns={"Date": "date"})
            return df
        except Exception as e:
            print(f"    [{ticker}] retry {attempt + 1}/{max_retries}: {e}")
            time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame()


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch yfinance OHLC for FNSPID")
    p.add_argument(
        "--pad-days",
        type=int,
        default=45,
        help="Trading-day padding on each side of the news span "
        "(needs to cover the longest forward-return horizon)",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if prices.parquet exists")
    p.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Override ticker list (default: read from predictions.json)",
    )
    p.add_argument("--date-from", default=None, help="Override news date_min (YYYY-MM-DD)")
    p.add_argument("--date-to", default=None, help="Override news date_max (YYYY-MM-DD)")
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Per-run results directory; reads predictions.json from "
        "it and writes prices.parquet there",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prices_path = out_dir / "prices.parquet"
    meta_path = out_dir / "prices_meta.json"

    if prices_path.exists() and not args.force:
        print(f"  prices.parquet already exists at {prices_path} — pass --force to redownload")
        prices = pd.read_parquet(prices_path)
        print(
            f"  rows={len(prices):,}  tickers={prices['ticker'].nunique()}  "
            f"dates={prices['date'].min().date()}..{prices['date'].max().date()}"
        )
        return

    if args.tickers:
        tickers = list(args.tickers)
        date_min = pd.to_datetime(args.date_from or "2014-01-01")
        date_max = pd.to_datetime(args.date_to or "2020-06-01")
        print(f"  using --tickers override ({len(tickers)} symbols)")
    else:
        meta = _read_pred_meta(out_dir)
        tickers = list(meta["tickers"])
        date_min = pd.to_datetime(meta["date_min"])
        date_max = pd.to_datetime(meta["date_max"])

    start = (date_min - pd.Timedelta(days=args.pad_days)).strftime("%Y-%m-%d")
    end = (date_max + pd.Timedelta(days=args.pad_days)).strftime("%Y-%m-%d")
    universe = list(dict.fromkeys(tickers + list(BENCHMARKS)))

    print(
        f"  date span: {start} → {end}  "
        f"({len(universe)} symbols incl. SPY + {len(SECTOR_ETFS)} sector ETFs)"
    )
    print(f"  pad_days={args.pad_days}")

    frames: list[pd.DataFrame] = []
    misses: list[str] = []
    for sym in universe:
        df = _download_one(sym, start, end)
        if df.empty:
            misses.append(sym)
            print(f"    [{sym}] EMPTY — skipping")
            continue
        frames.append(df)
        print(f"    [{sym}] {len(df)} rows  {df['date'].min().date()}..{df['date'].max().date()}")

    if not frames:
        raise RuntimeError("yfinance returned empty for every ticker; check connectivity.")

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices.to_parquet(prices_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "tickers_requested": universe,
                "tickers_missing": misses,
                "date_start": start,
                "date_end": end,
                "rows": int(len(prices)),
                "tickers_returned": int(prices["ticker"].nunique()),
            },
            indent=2,
        )
    )
    print(f"\n  Saved {len(prices):,} rows to {prices_path}")
    if misses:
        print(f"  Missing tickers ({len(misses)}): {misses}")


if __name__ == "__main__":
    main()
