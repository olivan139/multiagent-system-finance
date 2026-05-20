#!/usr/bin/env python3
"""Reliability-diagram + ECE analysis for the four base agents on the
Twitter test split, before and after a one-parameter temperature scaling
fitted on the validation split.

Inputs:
    python/results/twitter/_cache_agents.npz
        keys: tfidf_val_proba, tfidf_test_proba, lex_val_proba,
              lex_test_proba, llm_val_proba, llm_test_proba,
              val_labels, test_labels
    python/results/twitter/_cache_finbert.npz
        keys: val_proba, test_proba, val_labels, test_labels

Outputs:
    thesis/images/calibration_reliability.pdf  (4-panel figure)
    python/results/calibration_summary.json    (ECE table)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results" / "twitter"
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
SUMMARY = REPO_ROOT / "results" / "calibration_summary.json"

LABEL_ORDER = ["negative", "neutral", "positive"]
N_BINS = 10
EPS = 1e-12


def _to_indices(labels: np.ndarray) -> np.ndarray:
    """Convert string labels in LABEL_ORDER to integer class indices."""
    mapping = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
    return np.array([mapping[str(lbl)] for lbl in labels], dtype=np.int64)


def _safe_log(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1.0))


def temperature_scale(proba: np.ndarray, T: float) -> np.ndarray:
    """Re-softmax a probability matrix under temperature T.

    proba: (N, K) probability matrix that we treat as softmax(z). We
    reconstruct logits via z = log(p) and return softmax(z / T).
    """
    z = _safe_log(proba)
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(proba: np.ndarray, y_true: np.ndarray) -> float:
    """Multi-class NLL with clipped log."""
    p_true = proba[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(np.clip(p_true, EPS, 1.0))))


def fit_temperature(val_proba: np.ndarray, val_y: np.ndarray) -> float:
    """Minimise NLL on the validation set over T in (0.05, 10.0)."""

    def obj(T: float) -> float:
        return nll(temperature_scale(val_proba, T), val_y)

    res = minimize_scalar(obj, bounds=(0.4, 5.0), method="bounded")
    return float(res.x)


def expected_calibration_error(
    proba: np.ndarray, y_true: np.ndarray, n_bins: int = N_BINS
) -> tuple[float, list[tuple[float, float, int]]]:
    """Bin by top-class confidence, return weighted ECE and per-bin
    (mean_confidence, observed_accuracy, count) triples."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[tuple[float, float, int]] = []
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        n_in = int(mask.sum())
        if n_in == 0:
            bins.append((float((lo + hi) / 2), float("nan"), 0))
            continue
        mean_conf = float(conf[mask].mean())
        acc = float(correct[mask].mean())
        bins.append((mean_conf, acc, n_in))
        ece += (n_in / N) * abs(mean_conf - acc)
    return float(ece), bins


def plot_reliability(
    ax,
    bins_before: list[tuple[float, float, int]],
    bins_after: list[tuple[float, float, int]],
    title: str,
    ece_before: float,
    ece_after: float,
    annotation: str | None = None,
) -> None:
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = 1.0 / N_BINS

    counts_before = np.array([b[2] for b in bins_before], dtype=float)
    counts_after = np.array([b[2] for b in bins_after], dtype=float)
    acc_before = np.array([b[1] for b in bins_before], dtype=float)
    acc_after = np.array([b[1] for b in bins_after], dtype=float)

    n_before = counts_before.sum()
    n_after = counts_after.sum()
    weights_before = counts_before / max(n_before, 1.0)
    weights_after = counts_after / max(n_after, 1.0)

    ax.bar(
        centres - width * 0.22,
        weights_before * 0.4,
        width=width * 0.4,
        color="lightgray",
        edgecolor="gray",
        alpha=0.55,
        zorder=1,
        label="density (raw)",
    )
    ax.bar(
        centres + width * 0.22,
        weights_after * 0.4,
        width=width * 0.4,
        color="#d0d0d0",
        edgecolor="gray",
        alpha=0.55,
        zorder=1,
    )

    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--", zorder=2)
    mask_b = ~np.isnan(acc_before)
    mask_a = ~np.isnan(acc_after)
    ax.plot(
        centres[mask_b],
        acc_before[mask_b],
        marker="o",
        color="#1f77b4",
        linewidth=1.6,
        markersize=5,
        label=f"raw  (ECE={ece_before:.3f})",
        zorder=3,
    )
    ax.plot(
        centres[mask_a],
        acc_after[mask_a],
        marker="s",
        color="#d62728",
        linewidth=1.6,
        markersize=5,
        label=f"T-scaled  (ECE={ece_after:.3f})",
        zorder=3,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax.grid(True, linestyle=":", alpha=0.5)
    if annotation:
        ax.text(
            0.97,
            0.04,
            annotation,
            transform=ax.transAxes,
            fontsize=7.8,
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, pad=2.5),
        )


def main() -> int:
    cache_agents = np.load(RESULTS / "_cache_agents.npz", allow_pickle=True)
    cache_finbert = np.load(RESULTS / "_cache_finbert.npz", allow_pickle=True)

    val_y = _to_indices(cache_agents["val_labels"])
    test_y = _to_indices(cache_agents["test_labels"])
    val_y_fb = _to_indices(cache_finbert["val_labels"])
    test_y_fb = _to_indices(cache_finbert["test_labels"])

    agents = [
        (
            "TF-IDF + LogReg",
            np.asarray(cache_agents["tfidf_val_proba"]),
            np.asarray(cache_agents["tfidf_test_proba"]),
            val_y,
            test_y,
            None,
        ),
        (
            "FinBERT (fine-tuned)",
            np.asarray(cache_finbert["val_proba"]),
            np.asarray(cache_finbert["test_proba"]),
            val_y_fb,
            test_y_fb,
            None,
        ),
        (
            "GPT-4o-mini (zero-shot)",
            np.asarray(cache_agents["llm_val_proba"]),
            np.asarray(cache_agents["llm_test_proba"]),
            val_y,
            test_y,
            "constant self-reported\nconfidence (0.85)",
        ),
        (
            "Loughran-McDonald Lexicon",
            np.asarray(cache_agents["lex_val_proba"]),
            np.asarray(cache_agents["lex_test_proba"]),
            val_y,
            test_y,
            "two discrete confidence\nlevels (0.70 / 1.00)",
        ),
    ]

    summary: list[dict] = []
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.4), constrained_layout=True)
    axes = axes.flatten()

    for ax, (name, val_proba, test_proba, vy, ty, annot) in zip(axes, agents):
        T = fit_temperature(val_proba, vy)
        test_proba_T = temperature_scale(test_proba, T)
        ece_raw, bins_raw = expected_calibration_error(test_proba, ty)
        ece_T, bins_T = expected_calibration_error(test_proba_T, ty)
        nll_raw = nll(test_proba, ty)
        nll_T = nll(test_proba_T, ty)
        max_conf_raw = float(test_proba.max(axis=1).mean())
        max_conf_T = float(test_proba_T.max(axis=1).mean())

        plot_reliability(
            ax,
            bins_raw,
            bins_T,
            name,
            ece_raw,
            ece_T,
            annotation=annot,
        )
        summary.append(
            {
                "agent": name,
                "temperature": T,
                "ece_raw": ece_raw,
                "ece_t_scaled": ece_T,
                "nll_raw": nll_raw,
                "nll_t_scaled": nll_T,
                "mean_top_conf_raw": max_conf_raw,
                "mean_top_conf_t_scaled": max_conf_T,
            }
        )

    fig.suptitle(
        "Per-agent reliability diagrams, Twitter test set (N = 1,194)",
        fontsize=12,
    )
    out_pdf = FIG_DIR / "calibration_reliability.pdf"
    out_png = FIG_DIR / "calibration_reliability.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_pdf}")
    print(f"Wrote {SUMMARY}")
    print("\n=== Calibration summary (Twitter test set) ===")
    print(
        f"{'Agent':<28} {'T':>6} {'ECE_raw':>9} {'ECE_T':>8} {'NLL_raw':>9} {'NLL_T':>8} {'conf_raw':>9}"
    )
    for r in summary:
        print(
            f"{r['agent']:<28} {r['temperature']:>6.3f} {r['ece_raw']:>9.4f} "
            f"{r['ece_t_scaled']:>8.4f} {r['nll_raw']:>9.4f} "
            f"{r['nll_t_scaled']:>8.4f} {r['mean_top_conf_raw']:>9.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
