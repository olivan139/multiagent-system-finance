#!/usr/bin/env python3
"""Plots for Section 5.X (Downstream Signal Test, Option C: multi-horizon
+ sector-relative benchmarks).

Inputs (under ``--results-dir``)
    predictions.json            per-headline agent + ensemble probs
    prices.parquet              OHLC for the universe + SPY + sector ETFs
    downstream_metrics.json     output of scripts/downstream_metrics.py

Outputs into ``--fig-dir`` (defaults to ``thesis/images/``):
    downstream_ic_grid.png      mean IC bars per system, 3 horizons x 2 benchmarks
    downstream_returns.png      mean abnormal-return buckets at 1d / sector
    downstream_scatter.png      ensemble signal vs 5d sector-relative return
    downstream_horizon.png      ensemble IC vs horizon for SPY and sector
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from mas.config import LABELS, RESULTS_DIR

DISPLAY_ORDER = ("ensemble", "finbert", "llm", "tfidf", "lex", "neutral", "random")
DISPLAY_ORDER_NO_LLM = ("ensemble", "finbert", "tfidf", "lex", "neutral", "random")
DISPLAY_NAME = {
    "ensemble": "Ensemble",
    "finbert": "FinBERT FT",
    "llm": "Single LLM",
    "tfidf": "TF-IDF + LR",
    "lex": "LM Lexicon",
    "neutral": "Neutral baseline",
    "random": "Random baseline",
}
COLOR_MAP = {
    "ensemble": "#2c7fb8",
    "finbert": "#41ab5d",
    "llm": "#a6611a",
    "tfidf": "#878787",
    "lex": "#dfc27d",
    "neutral": "#cccccc",
    "random": "#999999",
}


def _load_metrics(metrics_path: Path) -> dict:
    return json.loads(metrics_path.read_text())


def _systems_from_view(view: dict) -> list[dict]:
    return view["per_system"]


def plot_ic_grid(metrics: dict, fig_dir: Path) -> Path:
    cfg = metrics["config"]
    horizons = cfg["horizons"]
    benchmarks = cfg["benchmarks"]
    nrows, ncols = len(horizons), len(benchmarks)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows), sharey=False)
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = np.array([[ax] for ax in axes])

    for r, h in enumerate(horizons):
        for c, bench in enumerate(benchmarks):
            view_key = f"{h}d_{bench}"
            view = metrics["by_view"].get(view_key)
            ax = axes[r, c]
            if view is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                ax.set_axis_off()
                continue
            rows = {m["system"]: m for m in _systems_from_view(view)}
            systems = [s for s in DISPLAY_ORDER if s in rows]
            means = [rows[s].get("ic_mean", float("nan")) or 0.0 for s in systems]
            ci_lo = [rows[s].get("ic_ci_lo", float("nan")) for s in systems]
            ci_hi = [rows[s].get("ic_ci_hi", float("nan")) for s in systems]
            n_days = [rows[s].get("n_days_used", 0) for s in systems]
            xs = np.arange(len(systems))
            err_lo = [m - lo if not np.isnan(lo) else 0 for m, lo in zip(means, ci_lo)]
            err_hi = [hi - m if not np.isnan(hi) else 0 for m, hi in zip(means, ci_hi)]
            ax.bar(
                xs,
                means,
                yerr=[err_lo, err_hi],
                capsize=3,
                alpha=0.85,
                color=[COLOR_MAP[s] for s in systems],
            )
            for x, m, n in zip(xs, means, n_days):
                ax.text(x, max(m + 0.002, 0.0005), f"{n}d", ha="center", fontsize=7, color="#444")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xticks(xs)
            ax.set_xticklabels(
                [DISPLAY_NAME[s] for s in systems], rotation=20, ha="right", fontsize=8
            )
            bench_label = "SPY" if bench == "spy" else "sector"
            ax.set_title(f"{h}-day horizon, vs {bench_label}", fontsize=10)
            ax.grid(axis="y", linestyle=":", alpha=0.35)
            if c == 0:
                ax.set_ylabel("Mean daily IC")

    fig.suptitle(
        f"Sentiment-signal Information Coefficient across horizons + benchmarks\n"
        f"FNSPID slice: {metrics['n_headlines']} headlines, "
        f"{len(metrics['ensemble_per_ticker_ic'])} tickers; bars are 95% bootstrap CI",
        fontsize=11,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = fig_dir / "downstream_ic_grid.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_return_buckets(metrics: dict, fig_dir: Path, view_key: str | None = None) -> Path:
    if view_key is None:
        view_key = metrics["config"]["primary_view"]
    view = metrics["by_view"][view_key]
    rows = {m["system"]: m for m in _systems_from_view(view)}
    systems = [s for s in ("ensemble", "finbert", "llm", "tfidf", "lex") if s in rows]

    bucket_order = ("negative", "neutral", "positive")
    colors = {"negative": "#d7191c", "neutral": "#bababa", "positive": "#1a9641"}

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.27
    xs = np.arange(len(systems))

    for j, b in enumerate(bucket_order):
        means_bp = [
            (rows[s].get("buckets_mean_ret", {}).get(b, {}).get("mean_ret") or 0.0) * 1e4
            for s in systems
        ]
        ns = [rows[s].get("buckets_mean_ret", {}).get(b, {}).get("n", 0) for s in systems]
        ax.bar(
            xs + (j - 1) * width,
            means_bp,
            width=width,
            color=colors[b],
            alpha=0.85,
            label=f"predicted {b}",
        )
        for x, m, n in zip(xs + (j - 1) * width, means_bp, ns):
            offset = 1.5 if m >= 0 else -3.5
            ax.text(x, m + offset, f"n={n}", ha="center", fontsize=7, color="#333")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY_NAME[s] for s in systems], rotation=15, ha="right")
    ax.set_ylabel("Mean abnormal return (basis points)")
    h = view["horizon_days"]
    bench_label = "SPY" if view["benchmark"] == "spy" else "sector ETF"
    ax.set_title(
        f"Mean {h}-day abnormal return ({h}d vs {bench_label}) by predicted-sentiment bucket\n"
        f"positive minus negative spread = signal value"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out = fig_dir / "downstream_returns.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_signal_scatter(
    metrics: dict, fig_dir: Path, results_dir: Path, view_key: str | None = None
) -> Path:
    if view_key is None:
        view_key = metrics["config"]["primary_view"]
    view = metrics["by_view"][view_key]
    h = view["horizon_days"]
    bench = view["benchmark"]
    abn_col = f"abn_{bench}_{h}d"

    payload = json.loads((results_dir / "predictions.json").read_text())
    df = pd.DataFrame(payload["rows"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize().astype("datetime64[ns]")

    from scripts.downstream_metrics import (
        _build_abnormal_returns,
        _attach_returns,
    )

    prices = pd.read_parquet(results_dir / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize().astype("datetime64[ns]")
    rets = _build_abnormal_returns(prices, tuple(metrics["config"]["horizons"]))
    joined = _attach_returns(df, rets, tuple(metrics["config"]["horizons"]))
    joined = joined.dropna(subset=[abn_col])

    NEG, POS = LABELS.index("negative"), LABELS.index("positive")
    if "ensemble_proba" not in joined.columns:
        raise RuntimeError("ensemble_proba missing from joined predictions")
    proba = np.array(joined["ensemble_proba"].tolist())
    sig = proba[:, POS] - proba[:, NEG]
    ret = joined[abn_col].to_numpy() * 100.0
    rho, p_val = spearmanr(sig, ret)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sig, ret, s=6, alpha=0.18, color="#2c7fb8")
    x_axis = np.linspace(-1, 1, 50)
    coef = np.polyfit(sig, ret, 1)
    ax.plot(
        x_axis,
        coef[0] * x_axis + coef[1],
        color="#d7191c",
        lw=1.8,
        label=f"OLS slope = {coef[0]:+.3f}%/unit signal",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Ensemble signal: P(positive) − P(negative)")
    ax.set_ylabel(f"{h}-day abnormal return (%, vs {bench.upper()})")
    ax.set_title(
        f"Ensemble sentiment signal vs {h}-day abnormal return\n"
        f"Spearman ρ = {rho:+.4f}  (p = {p_val:.3g}), n = {len(joined)} headlines"
    )
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(np.percentile(ret, 1), np.percentile(ret, 99))
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    out = fig_dir / "downstream_scatter.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_horizon_curve(metrics: dict, fig_dir: Path) -> Path:
    horizons = metrics["config"]["horizons"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for bench, color, marker in (("spy", "#a6611a", "o"), ("sector", "#2c7fb8", "s")):
        ys = []
        ci_los = []
        ci_his = []
        for h in horizons:
            view = metrics["by_view"].get(f"{h}d_{bench}")
            if view is None:
                ys.append(np.nan)
                ci_los.append(np.nan)
                ci_his.append(np.nan)
                continue
            rows = {m["system"]: m for m in _systems_from_view(view)}
            ens = rows.get("ensemble", {})
            ys.append(ens.get("ic_mean", np.nan))
            ci_los.append(ens.get("ic_ci_lo", np.nan))
            ci_his.append(ens.get("ic_ci_hi", np.nan))
        ax.plot(
            horizons,
            ys,
            marker=marker,
            color=color,
            label=f"vs {bench.upper() if bench=='spy' else 'sector ETF'}",
        )
        ax.fill_between(horizons, ci_los, ci_his, color=color, alpha=0.15)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Forward horizon (trading days)")
    ax.set_ylabel("Mean daily IC (Spearman)")
    ax.set_title("Ensemble signal IC vs forward-return horizon\n95% bootstrap CI shaded")
    ax.set_xticks(horizons)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = fig_dir / "downstream_horizon.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR / "fnspid"),
        help="Directory containing predictions.json + prices.parquet " "+ downstream_metrics.json",
    )
    p.add_argument("--fig-dir", default="figures")
    p.add_argument(
        "--figure", choices=["ic", "buckets", "scatter", "horizon", "all"], default="all"
    )
    p.add_argument(
        "--suffix",
        default="",
        help="Optional suffix to append before .png "
        "(e.g. '_30k' so figures land at downstream_ic_grid_30k.png)",
    )
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics(results_dir / "downstream_metrics.json")

    def _rename(out_path: Path) -> Path:
        if not args.suffix:
            return out_path
        new = out_path.with_name(out_path.stem + args.suffix + out_path.suffix)
        out_path.rename(new)
        return new

    if args.figure in ("ic", "all"):
        out = plot_ic_grid(metrics, fig_dir)
        out = _rename(out)
        print(f"  wrote {out}")
    if args.figure in ("buckets", "all"):
        out = plot_return_buckets(metrics, fig_dir)
        out = _rename(out)
        print(f"  wrote {out}")
    if args.figure in ("scatter", "all"):
        out = plot_signal_scatter(metrics, fig_dir, results_dir)
        out = _rename(out)
        print(f"  wrote {out}")
    if args.figure in ("horizon", "all"):
        out = plot_horizon_curve(metrics, fig_dir)
        out = _rename(out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
