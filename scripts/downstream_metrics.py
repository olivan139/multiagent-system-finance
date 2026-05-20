#!/usr/bin/env python3
"""Compute downstream-signal metrics for the FNSPID test (Option C).

Joins per-headline sentiment predictions with abnormal returns at multiple
forward horizons (1, 5, 30 trading days) using two benchmarks:

  - SPY (broad-market control)
  - sector-ETF (XLK / XLF / XLV / XLY / XLP / XLE / XLI / XLU / XLB / XLRE
    / XLC) — falls back to SPY when the sector ETF was not yet trading on
    the headline date (e.g. XLRE before 2015-10, XLC before 2018-06).

Per-system metrics (computed for every (horizon, benchmark) pair):
  1. Mean Information Coefficient (IC) — daily Spearman of signal vs
     forward H-day abnormal return, averaged over trading days.
  2. IC information ratio (IC_mean / IC_std).
  3. Pooled Spearman ρ + p-value.
  4. Hit rate — fraction of rows where sign(signal) == sign(abn_ret).
  5. Conditional-return Welch t-test on hard {pos, neg} predictions.
  6. Bucketed mean abnormal return per predicted-label bucket.

Plus a per-ticker IC table (top / bottom) for the headline view
(1d, sector benchmark, ensemble system).

Output: ``results/fnspid/downstream_metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind

sys.path.insert(0, str(Path(__file__).parent.parent))

from mas.config import LABELS, RESULTS_DIR
from mas.data import SECTOR_ETFS, sector_for

DEFAULT_OUT_DIR = RESULTS_DIR / "fnspid"

NEG_IDX = LABELS.index("negative")
POS_IDX = LABELS.index("positive")

SOFT_SYSTEMS = ("tfidf", "finbert", "llm", "lex", "ensemble", "fintwit", "ensemble5")
ALWAYS_PRESENT_SYSTEMS = ("tfidf", "finbert", "lex", "ensemble")
PRED_KEY = {
    "tfidf": "tfidf_pred",
    "finbert": "finbert_pred",
    "llm": "llm_pred",
    "lex": "lex_pred",
    "ensemble": "ensemble_pred",
    "fintwit": "fintwit_pred",
    "ensemble5": "ensemble5_pred",
}
PROBA_KEY = {
    "tfidf": "tfidf_proba",
    "finbert": "finbert_proba",
    "llm": "llm_proba",
    "lex": "lex_proba",
    "ensemble": "ensemble_proba",
    "fintwit": "fintwit_proba",
    "ensemble5": "ensemble5_proba",
}
DISPLAY_NAME = {
    "tfidf": "TF-IDF + LogReg",
    "finbert": "FinBERT (fine-tuned)",
    "llm": "Single LLM (zero-shot)",
    "lex": "Loughran-McDonald Lexicon",
    "ensemble": "4-Agent Stacking Ensemble",
    "fintwit": "FinTwitBERT (fine-tuned)",
    "ensemble5": "5-Agent Stacking Ensemble",
    "neutral": "Neutral Baseline",
    "random": "Random Baseline",
}

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 30)
BENCHMARKS: tuple[str, ...] = ("spy", "sector")


def _load_predictions(pred_path: Path) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(pred_path.read_text())
    rows = payload["rows"]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize().astype("datetime64[ns]")
    return df, payload["meta"]


def _load_prices(prices_path: Path) -> pd.DataFrame:
    px = pd.read_parquet(prices_path)
    px["date"] = pd.to_datetime(px["date"]).dt.normalize().astype("datetime64[ns]")
    return px.sort_values(["ticker", "date"]).reset_index(drop=True)


ENTRY_LAG_DAYS: int = 1


def _forward_returns(prices: pd.DataFrame, horizons: tuple[int, ...]) -> dict[int, pd.DataFrame]:
    """For each ticker and each horizon H, build a long table with
    columns ``ticker, date, ret_{H}d`` where ``ret_{H}d`` is the close-to-
    close H-day return realised between dates ``t + LAG`` and
    ``t + LAG + H`` trading days. ``LAG = ENTRY_LAG_DAYS`` is applied to
    avoid the same-day intraday/after-hours look-ahead -- see the
    module-level comment.
    Returns a dict mapping H -> DataFrame.
    """
    out: dict[int, pd.DataFrame] = {h: [] for h in horizons}
    for tk, g in prices.groupby("ticker"):
        g = g[["date", "close"]].sort_values("date").reset_index(drop=True)
        for h in horizons:
            g_h = g.copy()
            g_h[f"close_entry_{h}"] = g_h["close"].shift(-ENTRY_LAG_DAYS)
            g_h[f"close_exit_{h}"] = g_h["close"].shift(-(ENTRY_LAG_DAYS + h))
            g_h[f"ret_{h}d"] = g_h[f"close_exit_{h}"] / g_h[f"close_entry_{h}"] - 1.0
            g_h["ticker"] = tk
            out[h].append(g_h[["ticker", "date", f"ret_{h}d"]])
    return {h: pd.concat(out[h], ignore_index=True) for h in horizons}


def _build_abnormal_returns(prices: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Build a single long table with one row per (ticker, date) that
    carries every horizon's raw return, every horizon's abnormal return
    vs SPY, and every horizon's abnormal return vs the ticker's sector
    ETF.

    The output columns are:
        ticker, date,
        ret_1d, ret_5d, ret_30d,
        abn_spy_1d, abn_spy_5d, abn_spy_30d,
        abn_sector_1d, abn_sector_5d, abn_sector_30d,
        sector_etf
    Sector-relative columns fall back to SPY-relative when the sector ETF
    return is missing for that date (e.g. before XLRE's inception).
    """
    fwd_by_h = _forward_returns(prices, horizons)

    wide = None
    for h, df_h in fwd_by_h.items():
        if wide is None:
            wide = df_h
        else:
            wide = wide.merge(df_h, on=["ticker", "date"], how="outer")

    bench_set = {"SPY"} | set(SECTOR_ETFS)
    bench_wide = wide[wide["ticker"].isin(bench_set)].copy()
    asset_wide = wide[~wide["ticker"].isin(bench_set)].copy()

    spy = bench_wide[bench_wide["ticker"] == "SPY"][
        ["date"] + [f"ret_{h}d" for h in horizons]
    ].rename(columns={f"ret_{h}d": f"spy_ret_{h}d" for h in horizons})

    sector_long = bench_wide[bench_wide["ticker"].isin(set(SECTOR_ETFS))].rename(
        columns={"ticker": "sector_etf"}
    )
    sector_long = sector_long.rename(columns={f"ret_{h}d": f"sec_ret_{h}d" for h in horizons})[
        ["sector_etf", "date"] + [f"sec_ret_{h}d" for h in horizons]
    ]

    asset_wide["sector_etf"] = asset_wide["ticker"].map(sector_for)
    asset_wide = asset_wide.merge(spy, on="date", how="left")
    asset_wide = asset_wide.merge(sector_long, on=["sector_etf", "date"], how="left")

    for h in horizons:
        asset_wide[f"abn_spy_{h}d"] = asset_wide[f"ret_{h}d"] - asset_wide[f"spy_ret_{h}d"]
        sec = asset_wide[f"ret_{h}d"] - asset_wide[f"sec_ret_{h}d"]

        asset_wide[f"abn_sector_fallback_{h}d"] = sec.isna() & asset_wide[f"abn_spy_{h}d"].notna()

        asset_wide[f"abn_sector_{h}d"] = sec.where(sec.notna(), asset_wide[f"abn_spy_{h}d"])

    keep_cols = (
        ["ticker", "date", "sector_etf"]
        + [f"ret_{h}d" for h in horizons]
        + [f"abn_spy_{h}d" for h in horizons]
        + [f"abn_sector_{h}d" for h in horizons]
        + [f"abn_sector_fallback_{h}d" for h in horizons]
    )
    return asset_wide[keep_cols].sort_values(["ticker", "date"]).reset_index(drop=True)


def _attach_returns(
    news: pd.DataFrame, rets: pd.DataFrame, horizons: tuple[int, ...]
) -> pd.DataFrame:
    """Forward-snap each headline to the next available trading day for
    its ticker (so weekend headlines pick up the Monday open). Carries all
    return columns through.
    """
    news = news.copy()
    news["date"] = pd.to_datetime(news["date"]).dt.normalize().astype("datetime64[ns]")
    news = news.sort_values(["ticker", "date"]).reset_index(drop=True)

    rets = rets.copy()
    rets["date"] = pd.to_datetime(rets["date"]).dt.normalize().astype("datetime64[ns]")
    rets = rets.sort_values(["ticker", "date"]).reset_index(drop=True)

    out_frames: list[pd.DataFrame] = []
    ret_cols = (
        ["sector_etf"]
        + [f"ret_{h}d" for h in horizons]
        + [f"abn_spy_{h}d" for h in horizons]
        + [f"abn_sector_{h}d" for h in horizons]
    )
    for tk, g_news in news.groupby("ticker"):
        g_rets = rets[rets["ticker"] == tk]
        if g_rets.empty:
            continue
        merged = pd.merge_asof(
            g_news.sort_values("date"),
            g_rets[["date"] + ret_cols].sort_values("date"),
            on="date",
            direction="forward",
            tolerance=pd.Timedelta("5D"),
        )
        out_frames.append(merged)

    if not out_frames:
        return news.head(0)
    return pd.concat(out_frames, ignore_index=True)


def _signal_for(df: pd.DataFrame, system: str, seed: int = 42) -> np.ndarray:
    if system == "neutral":
        return np.zeros(len(df), dtype=np.float64)
    if system == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(-1.0, 1.0, size=len(df))
    proba_key = PROBA_KEY[system]
    p = np.array(df[proba_key].tolist(), dtype=np.float64)
    return p[:, POS_IDX] - p[:, NEG_IDX]


def _label_for(df: pd.DataFrame, system: str, seed: int = 42) -> np.ndarray:
    if system == "neutral":
        return np.array(["neutral"] * len(df))
    if system == "random":

        rng = np.random.default_rng(seed + 1000)
        return rng.choice(["negative", "neutral", "positive"], size=len(df))
    return df[PRED_KEY[system]].to_numpy()


N_RANDOM_SEEDS: int = 100


def _bootstrap_ci(
    values: np.ndarray, n_iter: int = 5000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_iter, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _compute_per_system(
    df_joined: pd.DataFrame, system: str, ret_col: str, min_per_day: int = 3, seed: int = 42
) -> dict:
    df_use = df_joined.dropna(subset=[ret_col]).copy()
    if df_use.empty:
        return {"system": system, "display": DISPLAY_NAME[system], "n_rows": 0}

    sig = _signal_for(df_use, system, seed=seed)
    df_use["_sig"] = sig
    df_use["_ret"] = df_use[ret_col]

    daily_ics: list[float] = []
    for _, g in df_use.groupby("date"):
        if len(g) < min_per_day:
            continue
        if g["_sig"].nunique() < 2:
            continue
        rho, _ = spearmanr(g["_sig"], g["_ret"])
        if not np.isnan(rho):
            daily_ics.append(float(rho))
    daily_arr = np.array(daily_ics)
    ic_mean = float(daily_arr.mean()) if len(daily_arr) else float("nan")
    ic_std = float(daily_arr.std(ddof=1)) if len(daily_arr) > 1 else float("nan")

    ic_t_stat = (ic_mean / ic_std) if ic_std and ic_std > 0 else float("nan")
    ic_ci_lo, ic_ci_hi = (
        _bootstrap_ci(daily_arr) if len(daily_arr) >= 5 else (float("nan"), float("nan"))
    )

    pooled_rho, pooled_p = spearmanr(df_use["_sig"], df_use["_ret"])
    pooled_rho = float(pooled_rho) if not np.isnan(pooled_rho) else float("nan")
    pooled_p = float(pooled_p) if not np.isnan(pooled_p) else float("nan")

    nz = (df_use["_sig"] != 0) & (df_use["_ret"] != 0)
    if nz.any():
        hit = float((np.sign(df_use.loc[nz, "_sig"]) == np.sign(df_use.loc[nz, "_ret"])).mean())
    else:
        hit = float("nan")

    labs = _label_for(df_use, system, seed=seed)
    pos_ret = df_use.loc[labs == "positive", "_ret"].to_numpy()
    neg_ret = df_use.loc[labs == "negative", "_ret"].to_numpy()
    pos_mean = float(pos_ret.mean()) if len(pos_ret) else float("nan")
    neg_mean = float(neg_ret.mean()) if len(neg_ret) else float("nan")
    pos_minus_neg = (
        pos_mean - neg_mean if not np.isnan(pos_mean) and not np.isnan(neg_mean) else float("nan")
    )
    if len(pos_ret) >= 5 and len(neg_ret) >= 5:
        t_stat, t_p = ttest_ind(pos_ret, neg_ret, equal_var=False)
        t_stat, t_p = float(t_stat), float(t_p)
    else:
        t_stat, t_p = float("nan"), float("nan")

    buckets: dict[str, dict] = {}
    for lab in ("positive", "neutral", "negative"):
        mask = labs == lab
        n = int(mask.sum())
        mean = float(df_use.loc[mask, "_ret"].mean()) if n else float("nan")
        buckets[lab] = {"n": n, "mean_ret": mean}

    return {
        "system": system,
        "display": DISPLAY_NAME[system],
        "n_rows": int(len(df_use)),
        "n_days_used": int(len(daily_arr)),
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_t_stat": ic_t_stat,
        "ic_ir": ic_t_stat,
        "ic_ci_lo": ic_ci_lo,
        "ic_ci_hi": ic_ci_hi,
        "pooled_spearman_rho": pooled_rho,
        "pooled_spearman_p": pooled_p,
        "hit_rate_signed": hit,
        "buckets_mean_ret": buckets,
        "pos_minus_neg_mean": pos_minus_neg,
        "welch_t_pos_vs_neg": t_stat,
        "welch_p_pos_vs_neg": t_p,
    }


def _per_ticker_ic(df_joined: pd.DataFrame, ret_col: str, system: str = "ensemble") -> dict:
    df_use = df_joined.dropna(subset=[ret_col]).copy()
    if df_use.empty:
        return {}
    df_use["_sig"] = _signal_for(df_use, system)
    out = {}
    for tk, g in df_use.groupby("ticker"):
        if len(g) < 20 or g["_sig"].nunique() < 2:
            continue
        rho, p = spearmanr(g["_sig"], g[ret_col])
        out[tk] = {"n": int(len(g)), "rho": float(rho), "p": float(p)}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--min-per-day", type=int, default=3, help="Minimum headlines/day to include the day in IC"
    )
    p.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_HORIZONS),
        help="Forward horizons in trading days",
    )
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Per-run results dir (reads predictions.json + "
        "prices.parquet, writes downstream_metrics.json)",
    )
    args = p.parse_args()
    horizons = tuple(args.horizons)

    out_dir = Path(args.out_dir)
    pred_path = out_dir / "predictions.json"
    prices_path = out_dir / "prices.parquet"
    metrics_path = out_dir / "downstream_metrics.json"

    print(f"  Loading {pred_path}")
    news, meta = _load_predictions(pred_path)
    print(
        f"    {len(news)} headlines, {news['ticker'].nunique()} tickers, "
        f"{news['date'].min().date()}..{news['date'].max().date()}"
    )

    print(f"  Loading {prices_path}")
    prices_raw = _load_prices(prices_path)
    print(f"    {len(prices_raw):,} OHLC rows over {prices_raw['ticker'].nunique()} symbols")

    print(f"  Building abnormal-return tables for horizons={horizons}")
    rets = _build_abnormal_returns(prices_raw, horizons)
    print(
        f"    {len(rets):,} (ticker, date) return rows  "
        f"(sector cov: {rets['abn_sector_1d'].notna().mean():.2%})"
    )

    joined = _attach_returns(news, rets, horizons)
    print(f"    joined: {len(joined)} headlines")

    has_llm = "llm_proba" in news.columns
    if not has_llm:
        print(
            "    note: predictions.json has no llm_proba column "
            "(--no-llm run); LLM system will be skipped"
        )
    soft_systems = SOFT_SYSTEMS if has_llm else ALWAYS_PRESENT_SYSTEMS
    systems = list(soft_systems) + ["neutral", "random"]
    by_view: dict[str, dict] = {}
    print()
    for h in horizons:
        for bench in BENCHMARKS:
            ret_col = f"abn_{bench}_{h}d"
            if ret_col not in joined.columns:
                continue
            n_with = int(joined[ret_col].notna().sum())
            print(
                f"  --- horizon={h}d, benchmark={bench.upper():6s} "
                f"(matched {n_with}/{len(joined)} headlines) ---"
            )
            per_system: list[dict] = []
            for s in systems:
                if s == "random":

                    runs = [
                        _compute_per_system(
                            joined, s, ret_col=ret_col, min_per_day=args.min_per_day, seed=42 + k
                        )
                        for k in range(N_RANDOM_SEEDS)
                    ]
                    runs = [r for r in runs if "ic_mean" in r]
                    if not runs:
                        per_system.append({"system": s, "display": DISPLAY_NAME[s], "n_rows": 0})
                        continue
                    keys_avg = (
                        "ic_mean",
                        "ic_std",
                        "ic_t_stat",
                        "ic_ci_lo",
                        "ic_ci_hi",
                        "pooled_spearman_rho",
                        "pooled_spearman_p",
                        "hit_rate_signed",
                        "pos_minus_neg_mean",
                        "welch_t_pos_vs_neg",
                        "welch_p_pos_vs_neg",
                    )
                    avg = {k: float(np.nanmean([r[k] for r in runs])) for k in keys_avg}
                    std = {
                        k + "_seed_std": float(np.nanstd([r[k] for r in runs], ddof=1))
                        for k in keys_avg
                    }
                    m = {**runs[0]}
                    m.update(avg)
                    m.update(std)
                    m["ic_ir"] = m["ic_t_stat"]
                    m["n_random_seeds"] = N_RANDOM_SEEDS
                    per_system.append(m)
                    print(
                        f"    {DISPLAY_NAME[s]:32s}  "
                        f"IC={m['ic_mean']:+.4f}±{std['ic_mean_seed_std']:.4f}  "
                        f"hit={m['hit_rate_signed']:.3f}  "
                        f"pos-neg={m['pos_minus_neg_mean']:+.4f} "
                        f"(p={m['welch_p_pos_vs_neg']:.3g})  "
                        f"[{N_RANDOM_SEEDS} seeds]"
                    )
                    continue
                m = _compute_per_system(joined, s, ret_col=ret_col, min_per_day=args.min_per_day)
                per_system.append(m)
                if "ic_mean" in m:
                    print(
                        f"    {DISPLAY_NAME[s]:32s}  IC={m['ic_mean']:+.4f}  "
                        f"hit={m['hit_rate_signed']:.3f}  "
                        f"pos-neg={m['pos_minus_neg_mean']:+.4f} "
                        f"(p={m['welch_p_pos_vs_neg']:.3g})"
                    )
            view_extra: dict = {}
            if bench == "sector":
                fb_col = f"abn_sector_fallback_{h}d"
                if fb_col in joined.columns:
                    mask = joined[ret_col].notna()
                    n_mask = int(mask.sum())
                    n_fb = int((joined.loc[mask, fb_col] == True).sum())
                    view_extra["sector_fallback_to_spy_pct"] = (
                        float(n_fb / n_mask) if n_mask else float("nan")
                    )
                    view_extra["sector_fallback_to_spy_n"] = n_fb
            by_view[f"{h}d_{bench}"] = {
                "horizon_days": h,
                "benchmark": bench,
                "n_with_return": n_with,
                "per_system": per_system,
                **view_extra,
            }

    primary_ret_col = f"abn_sector_{horizons[0]}d"
    per_ticker = _per_ticker_ic(joined, ret_col=primary_ret_col, system="ensemble")

    out = {
        "config": {
            "fnspid_meta": meta,
            "min_per_day": args.min_per_day,
            "horizons": list(horizons),
            "benchmarks": list(BENCHMARKS),
            "primary_view": f"{horizons[0]}d_sector",
        },
        "n_headlines": int(len(joined)),
        "by_view": by_view,
        "ensemble_per_ticker_ic": per_ticker,
    }
    metrics_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved metrics ({len(by_view)} views) to {metrics_path}")


if __name__ == "__main__":
    main()
