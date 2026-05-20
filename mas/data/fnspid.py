"""FNSPID slice loader for the downstream signal test (Section 5.X).

FNSPID (Dong, Fan, Peng 2024 — KDD ADS track) ships two huge CSVs of
date-stamped financial-news headlines tagged with the primary ticker. The
``Stock_news/All_external.csv`` is 5.7 GB; we cache it once on disk and
slice from it for every downstream-test invocation.

What we use from FNSPID
-----------------------
- ``Date`` (UTC string, parsed to ``datetime.date``)
- ``Article_title`` (headline text)
- ``Stock_symbol`` (single ticker per row)

Everything else (article body, summaries, URL, publisher) is dropped to
keep the dataframe small and the LLM cost predictable.

Two slice modes
---------------
- ``full``  — read the cached 5.7 GB file in chunks via pandas, filter
  on the fly, return a dataframe spanning the full alphabet. This is
  the default for Option C (broad cross-section).
- ``prefix`` — fall back to an HTTP Range partial-download (50 MB by
  default). Kept for the original Section 5.X reproducibility appendix
  that targeted only A-B tickers.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from ..config import PROJECT_ROOT
from .sp500_universe import DEFAULT_TICKER_WHITELIST as _SP500_FULL

FNSPID_URL = (
    "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/" "Stock_news/All_external.csv"
)
CACHE_DIR = PROJECT_ROOT / "data" / "fnspid"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FULL_CACHE_PATH = CACHE_DIR / "all_external_full.csv"


DEFAULT_TICKER_WHITELIST: tuple[str, ...] = _SP500_FULL


LEGACY_AB_WHITELIST: tuple[str, ...] = (
    "AAPL",
    "ABBV",
    "ABT",
    "ACN",
    "ADBE",
    "ADI",
    "ADP",
    "ADSK",
    "AIG",
    "AKAM",
    "ALK",
    "AMAT",
    "AMD",
    "AMGN",
    "AMT",
    "AMZN",
    "ANTM",
    "AON",
    "APA",
    "APD",
    "ATVI",
    "AVGO",
    "AVY",
    "AXP",
    "AZO",
    "BA",
    "BABA",
    "BAC",
    "BAX",
    "BBY",
    "BIDU",
    "BIIB",
    "BK",
    "BKR",
    "BLK",
    "BMY",
    "BSX",
)


@dataclass
class FnspidSlice:
    """Container for a sliced FNSPID dataframe + provenance metadata."""

    df: pd.DataFrame
    bytes_downloaded: int
    rows_raw: int
    rows_after_filter: int
    date_min: date
    date_max: date
    tickers: tuple[str, ...]
    source_url: str = FNSPID_URL


def _ensure_partial_download(
    n_bytes: int = 50 * 1024 * 1024,
    cache_path: Path | None = None,
    force: bool = False,
) -> Path:
    """Download the first ``n_bytes`` of the FNSPID CSV (cached on disk)."""
    cache_path = cache_path or (CACHE_DIR / f"all_external_first_{n_bytes}.csv")
    if cache_path.exists() and not force:
        return cache_path

    headers = {"Range": f"bytes=0-{n_bytes - 1}"}
    print(f"[FNSPID] Downloading first {n_bytes / 1e6:.0f} MB to {cache_path}")
    with requests.get(FNSPID_URL, headers=headers, stream=True, timeout=600) as r:
        r.raise_for_status()
        cache_path.write_bytes(r.content)
    print(f"[FNSPID] Cached {cache_path.stat().st_size:,} bytes")
    return cache_path


def _trim_partial_last_line(text: str) -> str:
    """Drop the trailing line of a Range-download (almost always truncated)."""
    last_newline = text.rfind("\n")
    if last_newline == -1:
        return ""
    return text[: last_newline + 1]


def load_fnspid_slice(
    n_bytes: int = 50 * 1024 * 1024,
    date_from: str = "2014-01-01",
    date_to: str = "2020-06-01",
    ticker_whitelist: Iterable[str] | None = DEFAULT_TICKER_WHITELIST,
    max_per_ticker_per_day: int = 1,
    max_rows: int | None = 5000,
    cache_path: Path | None = None,
) -> FnspidSlice:
    """Load and slice a date+ticker subset of FNSPID headlines.

    Parameters
    ----------
    n_bytes
        Size of the prefix download. 50 MB ≈ 220k raw rows, plenty of
        headroom after filtering to a small ticker universe.
    date_from, date_to
        ISO-format inclusive date bounds.
    ticker_whitelist
        Tickers to keep. ``None`` keeps every ticker (not recommended:
        FNSPID rows include tickers without yfinance coverage).
    max_per_ticker_per_day
        Per-ticker per-day cap to avoid a single news-storm dominating
        the IC computation. Set to ``1`` to keep one headline per
        (ticker, day) — the row with the longest headline is kept.
    max_rows
        Optional hard cap on the returned dataframe.

    Returns
    -------
    FnspidSlice with a DataFrame in columns ``[text, ticker, date]``
    plus provenance fields for the appendix.
    """
    blob_path = _ensure_partial_download(n_bytes, cache_path=cache_path)
    raw_text = _trim_partial_last_line(blob_path.read_text(encoding="utf-8", errors="replace"))

    df = pd.read_csv(
        io.StringIO(raw_text),
        usecols=["Date", "Article_title", "Stock_symbol"],
        dtype={"Article_title": "string", "Stock_symbol": "string"},
        on_bad_lines="skip",
    )
    rows_raw = len(df)

    df = df.rename(columns={"Date": "date", "Article_title": "text", "Stock_symbol": "ticker"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["date", "text", "ticker"])
    df["text"] = df["text"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df[df["text"].str.len() >= 10]

    lo = pd.to_datetime(date_from).date()
    hi = pd.to_datetime(date_to).date()
    df = df[(df["date"] >= lo) & (df["date"] <= hi)]

    if ticker_whitelist is not None:
        wl = {t.upper() for t in ticker_whitelist}
        df = df[df["ticker"].isin(wl)]

    if max_per_ticker_per_day:
        df["_len"] = df["text"].str.len()
        df = (
            df.sort_values(["ticker", "date", "_len"], ascending=[True, True, False])
            .groupby(["ticker", "date"], as_index=False)
            .head(max_per_ticker_per_day)
            .drop(columns=["_len"])
        )

    df = (
        df.drop_duplicates(subset=["ticker", "date", "text"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    if max_rows is not None and len(df) > max_rows:

        per_ticker = max_rows // df["ticker"].nunique()
        if per_ticker == 0:
            df = df.head(max_rows).reset_index(drop=True)
        else:
            df = (
                df.sort_values(["ticker", "date"])
                .assign(_rk=lambda d: d.groupby("ticker").cumcount())
                .query("_rk < @per_ticker")
                .drop(columns="_rk")
                .sort_values(["date", "ticker"])
                .reset_index(drop=True)
            )

    rows_after = len(df)
    return FnspidSlice(
        df=df,
        bytes_downloaded=blob_path.stat().st_size,
        rows_raw=rows_raw,
        rows_after_filter=rows_after,
        date_min=df["date"].min() if rows_after else lo,
        date_max=df["date"].max() if rows_after else hi,
        tickers=tuple(sorted(df["ticker"].unique())),
    )


def load_fnspid_full(
    full_csv_path: Path | None = None,
    date_from: str = "2014-01-01",
    date_to: str = "2020-06-01",
    ticker_whitelist: Iterable[str] | None = DEFAULT_TICKER_WHITELIST,
    max_per_ticker_per_day: int | None = None,
    max_per_ticker_total: int | None = 200,
    max_rows: int | None = None,
    chunk_size: int = 500_000,
) -> FnspidSlice:
    """Load + slice the cached full FNSPID CSV via chunked pandas reads.

    Parameters
    ----------
    full_csv_path
        Path to the cached 5.7 GB FNSPID file. Defaults to
        ``data/fnspid/all_external_full.csv`` under ``PROJECT_ROOT``.
    date_from, date_to
        ISO inclusive date bounds.
    ticker_whitelist
        Tickers to keep. Default is the broad S&P-500 universe in
        :mod:`mas.data.sp500_universe`.
    max_per_ticker_per_day
        Per-(ticker, day) cap. ``None`` keeps every row of the day --
        better for IC because multi-headline days are downweighted by
        averaging in the metrics step rather than thrown away.
    max_per_ticker_total
        Hard cap per ticker over the whole date range. Keeps the
        universe roughly balanced. Set to ``None`` for no cap.
    max_rows
        Optional final cap on the dataframe (after stratified
        balancing). ``None`` keeps everything.
    chunk_size
        Rows per pandas read chunk. 500k * a few hundred bytes per row
        is well within RAM budget.

    Returns
    -------
    FnspidSlice with ``df`` columns ``[text, ticker, date]`` plus
    provenance metadata.
    """
    csv_path = full_csv_path or FULL_CACHE_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} missing. Download with:\n" f"  curl -L -o {csv_path} {FNSPID_URL}"
        )

    lo = pd.to_datetime(date_from).date()
    hi = pd.to_datetime(date_to).date()
    wl = {t.upper() for t in ticker_whitelist} if ticker_whitelist else None

    rows_raw_total = 0
    keep: list[pd.DataFrame] = []
    print(f"  [FNSPID full] streaming {csv_path.name} in {chunk_size//1000}k-row chunks")
    chunks_done = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=["Date", "Article_title", "Stock_symbol"],
        dtype={"Article_title": "string", "Stock_symbol": "string"},
        chunksize=chunk_size,
        on_bad_lines="skip",
        engine="c",
    ):
        rows_raw_total += len(chunk)
        chunk = chunk.rename(
            columns={"Date": "date", "Article_title": "text", "Stock_symbol": "ticker"}
        )
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce", utc=True).dt.date
        chunk = chunk.dropna(subset=["date", "text", "ticker"])
        chunk["ticker"] = chunk["ticker"].astype(str).str.upper().str.strip()
        if wl is not None:
            chunk = chunk[chunk["ticker"].isin(wl)]
        chunk = chunk[(chunk["date"] >= lo) & (chunk["date"] <= hi)]
        chunk["text"] = chunk["text"].astype(str).str.strip()
        chunk = chunk[chunk["text"].str.len() >= 10]
        if not chunk.empty:
            keep.append(chunk)
        chunks_done += 1
        if chunks_done % 4 == 0:
            kept_so_far = sum(len(k) for k in keep)
            print(
                f"    [FNSPID full] {chunks_done * chunk_size // 1000}k rows scanned, "
                f"{kept_so_far} kept after filter"
            )

    if not keep:
        df = pd.DataFrame(columns=["date", "text", "ticker"])
    else:
        df = pd.concat(keep, ignore_index=True)
    print(f"  [FNSPID full] scan done: {rows_raw_total:,} raw rows, {len(df):,} pre-dedup")

    if max_per_ticker_per_day:
        df["_len"] = df["text"].str.len()
        df = (
            df.sort_values(["ticker", "date", "_len"], ascending=[True, True, False])
            .groupby(["ticker", "date"], as_index=False)
            .head(max_per_ticker_per_day)
            .drop(columns=["_len"])
        )

    df = df.drop_duplicates(subset=["ticker", "date", "text"])

    if max_per_ticker_total is not None:
        N = int(max_per_ticker_total)
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        def _evenly_spaced(g: pd.DataFrame) -> pd.DataFrame:
            if len(g) <= N:
                return g
            idx = np.linspace(0, len(g) - 1, N).round().astype(int)
            idx = np.unique(idx)
            return g.iloc[idx]

        df = (
            df.groupby("ticker", group_keys=False, sort=False)
            .apply(_evenly_spaced)
            .reset_index(drop=True)
        )

    if max_rows is not None and len(df) > max_rows:
        per_ticker = max_rows // df["ticker"].nunique()
        if per_ticker == 0:
            df = df.head(max_rows)
        else:
            df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

            def _evenly_spaced_global(g: pd.DataFrame) -> pd.DataFrame:
                if len(g) <= per_ticker:
                    return g
                idx = np.linspace(0, len(g) - 1, per_ticker).round().astype(int)
                idx = np.unique(idx)
                return g.iloc[idx]

            df = (
                df.groupby("ticker", group_keys=False, sort=False)
                .apply(_evenly_spaced_global)
                .reset_index(drop=True)
            )

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    rows_after = len(df)

    return FnspidSlice(
        df=df,
        bytes_downloaded=int(csv_path.stat().st_size),
        rows_raw=int(rows_raw_total),
        rows_after_filter=rows_after,
        date_min=df["date"].min() if rows_after else lo,
        date_max=df["date"].max() if rows_after else hi,
        tickers=tuple(sorted(df["ticker"].unique())),
    )


__all__ = [
    "FNSPID_URL",
    "DEFAULT_TICKER_WHITELIST",
    "LEGACY_AB_WHITELIST",
    "FnspidSlice",
    "load_fnspid_slice",
    "load_fnspid_full",
    "FULL_CACHE_PATH",
]
